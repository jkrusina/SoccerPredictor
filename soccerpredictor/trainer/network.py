import numpy as np
from typing import Any, Dict, List, Tuple

from tensorflow.compat.v1 import Session

import keras.backend as K
from keras.initializers import glorot_uniform
from keras.layers import Input, concatenate, Dense, LSTM, Layer
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from soccerpredictor.util.config import SPConfig
from soccerpredictor.util.constants import *


class SPNetwork:
    """
    Assembles Keras Model and handles dynamic getting and setting of params of the layers.

    Network consists of two heads - main head and head2 which are concatenated into final output layer.
    The idea is that the main head contains weights for current team and the head2 weights are
    dynamically changed according to opponent the current team plays against.

    Both heads consist of single stateful LSTM layer and Dense layer. The Dense layer uses ELU
    activation. Both layers use L2 regularization.
    Output layer uses softmax activation with 2 output classes.

    Interacting with the Keras model itself needs to be done within its own session and graph.
    E.g.: by using "with self._session.as_default(), self._graph.as_default():".

    """

    def __init__(self, team_name: str, target_team: bool, session: Session, lenc_bitlen: int) -> None:
        """

        :param team_name: Team's name the network is created for.
        :param target_team: Whether the team is in test set.
        :param session: Current session used.
        :param lenc_bitlen: Bitlength needed to encode all teams names.
        """
        config = SPConfig()
        self._session = session
        self._graph = session.graph
        self._lenc_bitlen = lenc_bitlen
        self._lrdecay = config.lrdecay
        self._features = FEATURES_COMMON + FEATURES_WD
        self._team_name = team_name
        self._model = None
        self._seed = config.seed

        # Store names of main head and head2 layers for direct access to layers
        self._main_head_layers_names = []
        self._main_head_stateful_layers_names = []
        self._head2_layers_names = []
        self._head2_stateful_layers_names = []

        # Dropout and lr can differ for teams' models which are not used in testing
        if target_team:
            self._lr = LR
            self._dropout = DROPOUT
        else:
            self._lr = NONTEST_LR
            self._dropout = NONTEST_DROPOUT

    def build(self) -> None:
        """
        Builds actual network as Keras Model.

        """
        with self._session.as_default(), self._graph.as_default():
            self._model = self._assemble_network()

            # Store layers names and stateful layers names
            for layer in self._model.layers:
                if len(layer.get_weights()) > 0:
                    if "head2" not in layer.name:
                        self._main_head_layers_names.append(layer.name)
                        if self._isstateful(layer):
                            self._main_head_stateful_layers_names.append(layer.name)
                    else:
                        self._head2_layers_names.append(layer.name)
                        if self._isstateful(layer):
                            self._head2_stateful_layers_names.append(layer.name)

    def _assemble_network(self) -> Model:
        """
        Assembles the network as Keras Model.

        Every layer needs to be named correctly:
        Input layers must be named: "input_team[1|2]_<feature_name>".
        Intermediate layers must be named "head[1|2]_<layer_name>".
        Output layer must be named: "output".

        Head2 is basically just mirrored main head.

        Seed is set for each initializer to increase reproducibility.

        :return: Keras model.
        """
        head1_inputs = []
        head2_inputs = []

        for f in self._features:
            if f in FEATURES_TO_LENC:
                head1_inputs.append(Input(batch_shape=(BATCH_SIZE, None, self._lenc_bitlen),
                                          name=f"input_team1_{f}"))
                head2_inputs.append(Input(batch_shape=(BATCH_SIZE, None, self._lenc_bitlen),
                                          name=f"input_team2_{f}"))
            else:
                head1_inputs.append(Input(batch_shape=(BATCH_SIZE, None, 1), name=f"input_team1_{f}"))
                head2_inputs.append(Input(batch_shape=(BATCH_SIZE, None, 1), name=f"input_team2_{f}"))

        # Main head
        head1_input_concat = concatenate(inputs=head1_inputs, name="head1_input_concat")
        head1_rnn1 = LSTM(35,
                          dropout=self._dropout,
                          stateful=STATEFUL,
                          return_sequences=False,
                          kernel_regularizer=l2(0.01),
                          kernel_initializer=glorot_uniform(self._seed),
                          name="head1_rnn1")(head1_input_concat)
        head1_fc1 = Dense(15,
                          activation="elu",
                          kernel_regularizer=l2(0.01),
                          kernel_initializer=glorot_uniform(self._seed),
                          name="head1_fc1")(head1_rnn1)

        # Head2
        head2_input_concat = concatenate(inputs=head2_inputs, name="head2_input_concat")
        head2_rnn1 = LSTM(35,
                          dropout=self._dropout,
                          stateful=STATEFUL,
                          return_sequences=False,
                          kernel_regularizer=l2(0.01),
                          trainable=False,
                          kernel_initializer=glorot_uniform(self._seed),
                          name="head2_rnn1")(head2_input_concat)
        head2_fc1 = Dense(15,
                          activation="elu",
                          kernel_regularizer=l2(0.01),
                          trainable=False,
                          kernel_initializer=glorot_uniform(self._seed),
                          name="head2_fc1")(head2_rnn1)

        joint_concat = concatenate([head1_fc1, head2_fc1], name="joint_concat")
        output = Dense(2,
                       activation="softmax",
                       kernel_initializer=glorot_uniform(self._seed),
                       name="output")(joint_concat)

        model = Model(inputs=head1_inputs+head2_inputs,
                      outputs=output,
                      name=self._team_name)

        model.compile(optimizer=Adam(learning_rate=self._lr, clipvalue=0.5, epsilon=1e-7),
                      loss=SparseCategoricalCrossentropy(),
                      metrics=["acc"])

        return model

    def train_on_batch(self,
                       x_input: Dict[str, np.ndarray],
                       y_input: Dict[str, np.ndarray],
                       class_weights: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for Keras train_on_batch.

        :param x_input: X input values.
        :param y_input: Y target value.
        :param class_weights: Class weights.
        :return: Training loss and metrics.
        """
        with self._session.as_default(), self._graph.as_default():
            loss, acc = self._model.train_on_batch(x_input, y_input, class_weight=class_weights)

        return loss, acc

    def test_on_batch(self,
                      x_input: Dict[str, np.ndarray],
                      y_input: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for Keras test_on_batch.

        :param x_input: X input values.
        :param y_input: Y target value.
        :return: Test loss and metrics.
        """
        with self._session.as_default(), self._graph.as_default():
            loss, acc = self._model.test_on_batch(x_input, y_input)

        return loss, acc

    def predict_on_batch(self, x_input: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Wrapper for Keras predict_on_batch.

        :param x_input: X input values.
        :return: Predictions probabilities.
        """
        with self._session.as_default(), self._graph.as_default():
            preds = self._model.predict_on_batch(x_input)

        return preds.flatten()

    def set_head2_params(self, params: Dict[str, Any]) -> None:
        """
        Sets head2 params with given params.
        Given params should contain only head2 layers.

        :param params: Params to set head2 layers with.
        """
        with self._session.as_default(), self._graph.as_default():
            for lname, w in params["weights"].items():
                self._model.get_layer(lname).set_weights(w)

            for lname, s in params["states"].items():
                K.set_value(self._model.get_layer(lname).states[0], s[0])
                K.set_value(self._model.get_layer(lname).states[1], s[1])

    def get_head2_states(self) -> Dict[str, Any]:
        """
        Gets states of head2 layers.

        :return: Head2 layers states.
        """
        states = {}

        with self._session.as_default(), self._graph.as_default():
            for lname in self._head2_stateful_layers_names:
                layer = self._model.get_layer(lname)
                states[lname] = [K.get_value(layer.states[0]), K.get_value(layer.states[1])]

        return states

    def set_main_head_params(self, params: Dict[str, Any], include_optimizer: bool) -> None:
        """
        Sets params of the main head layers with given params.

        :param params: Params to set.
        :param include_optimizer: Whether to set optimizer.
        """
        with self._session.as_default(), self._graph.as_default():
            for lname, w in params["weights"].items():
                self._model.get_layer(lname).set_weights(w)

            for lname, s in params["states"].items():
                K.set_value(self._model.get_layer(lname).states[0], s[0])
                K.set_value(self._model.get_layer(lname).states[1], s[1])

            if include_optimizer:
                K.set_value(self._model.optimizer.lr, params["optimizer_lr"])
                if len(params["optimizer"]) > 0:
                    self._model.optimizer.set_weights(params["optimizer"])

    def set_main_head_states(self, states: Dict[str, Any]) -> None:
        """
        Sets states of the main head with given states.

        :param states: States to set.
        """
        with self._session.as_default(), self._graph.as_default():
            for lname, s in states.items():
                K.set_value(self._model.get_layer(lname).states[0], s[0])
                K.set_value(self._model.get_layer(lname).states[1], s[1])

    def get_main_head_params(self, include_optimizer: bool) -> Dict[str, Any]:
        """
        Gets weights, states of main head layers, and optimizer.

        :param include_optimizer: Whether to get optimizer state.
        :return: Weights, states of main head layers, and optimizer.
        """
        weights = {}
        states = {}
        optimizer = {}
        optimizer_lr = None

        with self._session.as_default(), self._graph.as_default():
            for lname in self._main_head_layers_names:
                weights[lname] = self._model.get_layer(lname).get_weights()

            for lname in self._main_head_stateful_layers_names:
                layer = self._model.get_layer(lname)
                states[lname] = [K.get_value(layer.states[0]), K.get_value(layer.states[1])]

            if include_optimizer:
                optimizer = self._model.optimizer.get_weights()
                optimizer_lr = K.get_value(self._model.optimizer.lr)

        return {
            "weights": weights,
            "states": states,
            "optimizer": optimizer,
            "optimizer_lr": optimizer_lr
        }

    def get_main_head_weights(self) -> Dict[str, Any]:
        """
        Gets weights of main head layers.

        :return: Main head weights.
        """
        weights = {}

        with self._session.as_default(), self._graph.as_default():
            for lname in self._main_head_layers_names:
                weights[lname] = self._model.get_layer(lname).get_weights()

        return weights

    def get_main_head_states(self) -> Dict[str, List[np.array]]:
        """
        Gets states of main head layers.

        :return: Main head states.
        """
        states = {}

        with self._session.as_default(), self._graph.as_default():
            for lname in self._main_head_stateful_layers_names:
                layer = self._model.get_layer(lname)
                states[lname] = [K.get_value(layer.states[0]), K.get_value(layer.states[1])]

        return states

    def decay_learning_rate(self) -> None:
        """
        Decays learning rate by <current_lr * decay>.

        """
        with self._session.as_default(), self._graph.as_default():
            current_lr = K.get_value(self._model.optimizer.lr)
            K.set_value(self._model.optimizer.lr, current_lr * self._lrdecay)

    def reset_states(self) -> None:
        """
        Resets states of the RNNs.

        """
        with self._session.as_default(), self._graph.as_default():
            self._model.reset_states()

    def _isstateful(self, layer: Layer) -> bool:
        """
        Checks whether given layer is stateful.
        Copied from Keras code.

        :param layer: Layer to be checked.
        :return: Whether the layer is stateful.
        """
        return hasattr(layer, "reset_states") and getattr(layer, "stateful", False)
