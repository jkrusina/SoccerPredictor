from collections import Counter
import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
from typing import Any, Dict, List, Optional, Tuple

from tensorflow.compat.v1 import Session, Graph, ConfigProto, set_random_seed

from soccerpredictor.trainer.network import SPNetwork
from soccerpredictor.trainer.snapshot import SPSnapshot
from soccerpredictor.trainer.tensorboard import SPTensorboard
from soccerpredictor.util.config import SPConfig
from soccerpredictor.util.constants import *
from soccerpredictor.util.enums import Dataset, TargetVariable


class SPModel:
    """
    Represents a single team's model which encapsulates network.
    Handles data preparation and its formatting, and manages the network.

    Attributes:
        class_weights: Class weights for training.
        matches_data: Contains prepared input data to network for each dataset. Index is pointing
                      to current data that will be used. The index is incremented after every
                      match (and is individual for each dataset).
        states_after_training: Contains copy of the best states of the main head right after training
                               (before testing).

    """

    def __init__(self, team_name: str, test_teams: List[str], lenc_bitlen: int, folder_prefix: str) -> None:
        """

        :param team_name: Current model's team name.
        :param test_teams: Teams that participate in test dataset.
        :param lenc_bitlen: Bitlength required to encode teams names.
        :param folder_prefix: Current output folder prefix.
        """
        config = SPConfig()
        self._team_name = team_name
        self._lenc_bitlen = lenc_bitlen
        self._lr_decay = config.lrdecay
        self._lr_patience = config.lrpatience
        self._verbose = config.verbose
        self._target_variable = TargetVariable.FutureWD
        self._timesteps = config.timesteps
        self._features = FEATURES_COMMON + FEATURES_WD
        self._target_team = team_name in test_teams

        self.class_weights = {}
        self.states_after_training = None
        self.matches_data = {d: {"idx": 0, "data": {}} for d in Dataset}

        # Tracks best stats
        self.epochs_since_improvement = 0
        self.best_epoch = None
        self.best_loss = float("inf")
        self.best_acc = float("-inf")
        self.best_params = {}

        # Create model's own session and graph
        self.graph = Graph()

        if FORCE_SINGLE_THREADS:
            with self.graph.as_default():
                set_random_seed(config.seed)
                if FORCE_SINGLE_CPU:
                    cp = ConfigProto(intra_op_parallelism_threads=1,
                                     inter_op_parallelism_threads=1,
                                     device_count={"CPU": 1})
                else:
                    cp = ConfigProto(intra_op_parallelism_threads=1,
                                     inter_op_parallelism_threads=1)

                self.session = Session(graph=self.graph, config=cp)
        else:
            with self.graph.as_default():
                set_random_seed(config.seed)
                self.session = Session(graph=self.graph)

        self.tensorboard = SPTensorboard(self._team_name, self._target_team, self.session, folder_prefix)
        self.network = SPNetwork(self._team_name, self._target_team, self.session, lenc_bitlen)
        self.snapshot = SPSnapshot()

    def build_model(self) -> None:
        """
        Builds model individually.

        """
        self.network.build()
        self.best_params = self.network.get_main_head_params(include_optimizer=False)

    def build_model_from(self, team2_model: "SPModel") -> None:
        """
        Builds model based on another model.
        Copies params from the second model into the current one.
        Useful for initializing both models with same weights and states.

        Should be used only during init if we do not resume training.

        Optimizer does not need to be copied during first run because all models
        should be compiled identically anyway.

        :param team2_model: Model to copy from.
        """
        self.network.build()

        params = team2_model.network.get_main_head_params(include_optimizer=False)
        self.network.set_main_head_params(params, include_optimizer=False)
        self.best_params = params

    def update_performance(self,
                           stats: pd.DataFrame,
                           best_stats: pd.DataFrame,
                           epoch: int,
                           metrics: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, bool]:
        """
        Checks whether performance of the model improved at the end of current epoch.
        Either accuracy must increase, or loss must decrease without accuracy decreasing to
        acknowledge that model has improved.

        If there are no improvements for more epochs than threshold given by lrpatience then
        model's learning rate is decreased (multiplied by 0.95 by default).

        Further, no tracking of performance is made for first few initial epochs until model's
        accuracy and loss settle down a bit - depending on initialization there can be some
        high accuracy spikes at the beggining which would be logged and model would not be able
        to improve later.

        :param stats: Df with train or test stats.
        :param best_stats: Df with best train or test stats.
        :param epoch: Current epoch.
        :param metrics: Current metrics.
        :return Modified best stats and whether model improved.
        """
        # Default values
        loss = metrics["loss"]
        acc = metrics["acc"]
        improved = True

        # Wait a few epochs until the loss/acc settles down a bit
        if epoch < TRACK_PERF_FROM_EPOCH:
            self._record_new_best_epoch(stats, epoch, metrics)
        # Loss or acc improved
        elif (acc > self.best_acc) or (np.isclose(acc, self.best_acc) and loss < self.best_loss):
            self._record_new_best_epoch(stats, epoch, metrics)
        # No improvement to loss nor acc
        else:
            self.epochs_since_improvement += 1
            # Set loss and acc to nan
            loss = np.nan
            acc = np.nan
            improved = False

            # Decay learning rate after every nth epoch if there was no improvement for given
            # patience threshold
            if self._lr_patience and (self.epochs_since_improvement % self._lr_patience) == 0:
                self.network.decay_learning_rate()

        best_stats.loc[epoch, (self._team_name, "loss")] = loss
        best_stats.loc[epoch, (self._team_name, "acc")] = acc

        return best_stats, improved

    def _record_new_best_epoch(self, stats: pd.DataFrame, epoch: int, metrics: Dict[str, np.ndarray]) -> None:
        """
        Records new best epoch.

        :param stats: Train or test stats.
        :param epoch: Current epoch.
        :param metrics: Current metrics.
        """
        if self._target_team:
            self.tensorboard.notify_best_test(self.best_epoch, epoch, stats)
        else:
            self.tensorboard.notify_best_train(self.best_epoch, epoch, stats)

        self.epochs_since_improvement = 0
        self.best_epoch = epoch
        self.best_loss = metrics["loss"]
        self.best_acc = metrics["acc"]

        # Store current weights as new best
        self.restore_states_after_training()
        self.best_params = self.network.get_main_head_params(include_optimizer=True)

    def train_on_batch(self,
                       x_input: Dict[str, np.ndarray],
                       y_input: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for network train_on_batch.

        :param x_input: X input values.
        :param y_input: Y target value.
        :return: Training loss and acc metrics.
        """
        return self.network.train_on_batch(x_input, y_input, self.class_weights)

    def test_on_batch(self,
                      x_input: Dict[str, np.ndarray],
                      y_input: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for network test_on_batch.

        :param x_input: X input values.
        :param y_input: Y target value.
        :return: Test loss and acc metrics.
        """
        return self.network.test_on_batch(x_input, y_input)

    def predict_on_batch(self, x_input: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Wrapper for network predict_on_batch.

        :param x_input: X input values.
        :return: Predictions probabilities.
        """
        return self.network.predict_on_batch(x_input)

    def warm_up(self) -> None:
        """
        Warms up model before loading weights from file to properly initialize optimizer
        weights. This must be done in order to restore optimizer.
        Warming up is done by training on a single batch with arbitrary weights because they will
        be overwritten with the weights from file anyway.

        """
        x_input, y_input = self.form_input(Dataset.Train, team2_model=self)
        self.network.train_on_batch(x_input, y_input, self.class_weights)

    def save_states_after_training(self) -> None:
        """
        Saves states of the main head after training.

        """
        self.states_after_training = self.network.get_main_head_states()

    def restore_states_after_training(self) -> None:
        """
        Restores previously saved states of the main head after training.

        """
        self.network.set_main_head_states(self.states_after_training)

    def revert_to_best_params(self, include_optimizer: bool) -> None:
        """
        Restores params of the main head back to best params.

        :param include_optimizer: Whether to restore optimizer.
        """
        self.network.set_main_head_params(self.best_params, include_optimizer)

    def set_network_head2_params(self, team: str) -> None:
        """
        Sets head2 params with given team's params from snapshot.

        :param team: Team params to be set.
        """
        self.network.set_head2_params(self.snapshot.params[team])

    def store_network_head2_states(self, team2: str) -> None:
        """
        Saves head2 states of given team's model in snapshot.

        :param team2: Team params to be saved.
        """
        states = self.network.get_head2_states()
        self.snapshot.update_states(team2, states)

    def prepare_matches_data(self, dataset: Dataset, matches_data: pd.DataFrame) -> None:
        """
        Loops over given dataset to create timestep-sized windows of data.
        Data are properly formed (reshaped, encoded) to be ready to be passed as input to model.

        The datasets are quite small so they can be prepared this way before particular chunk of data is
        actually needed. This is helpful in order to avoid preparing data every time when looping
        over datasets.

        :param dataset: Dataset to prepare matches for.
        :param matches_data: Df of matches data for current team.
        """
        i = 0

        while True:
            iend = i + self._timesteps - 1
            subset = matches_data.loc[i:iend]
            if subset.empty:
                break

            # Default x, y values are none.
            # If model encounters none values during training it will skip them
            x_input = None
            y_input = None

            # Consider only chunks which length is equal to number of timesteps and which are not the
            # last chunk of data in the dataset (the last chunk is skipped to properly align match sequences)
            if len(subset) == self._timesteps and len(matches_data.loc[i:iend+1]) != self._timesteps:
                x_input = {}

                # Reshape features
                for f in self._features:
                    # Teams names are stored as lists so they need to be stacked
                    if f in FEATURES_TO_LENC:
                        team1_names = np.vstack(subset.loc[:, f].values).reshape((1, -1, self._lenc_bitlen))
                        x_input[f"input_team1_{f}"] = team1_names
                    else:
                        x_input[f"input_team1_{f}"] = subset.loc[:, f].values.reshape(1, -1, 1)

                # Get target variable from last row and ignore it if it is none
                y = subset.loc[iend, self._target_variable.value]
                if y is not None and not np.isnan(y):
                    y_input = {"output": y.reshape(-1, 1)}

            self.matches_data[dataset]["data"][i] = {"x_input": x_input, "y_input": y_input}
            i += 1

    def form_input(self,
                   dataset: Dataset,
                   team2_model: "SPModel") -> Tuple[Optional[Dict[str, np.array]],
                                                    Optional[Dict[str, np.array]]]:
        """
        Selects current chunk of data for the model's input.
        Names of input data from team2's model needs to be renamed from "team1" to "team2"
        to correctly match team1's model second head inputs.

        :param dataset: Dataset type.
        :param team2_model: Second team's model.
        :return: X and Y inputs for model.
        """
        # Get current data chunk based on index position for both models
        i = self.matches_data[dataset]["idx"]
        d1 = self.matches_data[dataset]["data"][i]
        j = team2_model.matches_data[dataset]["idx"]
        d2 = team2_model.matches_data[dataset]["data"][j]

        x_input = None
        d2_as_team2 = {}

        if d1["x_input"] and d2["x_input"]:
            for k, v in d2["x_input"].items():
                d2_as_team2[k.replace("team1", "team2")] = v

            # Unpack both inputs into a single dict
            x_input = {**d1["x_input"], **d2_as_team2}

        return x_input, d1["y_input"]

    def compute_class_weights(self,
                              team_matches_data: pd.DataFrame,
                              fixtures_ids: List[int],
                              verbose: bool = False) -> None:
        """
        Computes class weights which will be used to handle imbalances in the target classes.
        Usable for train set only.

        :param team_matches_data: Team matches data to count class weights from.
        :param fixtures_ids: Fixtures ids to select from the dataset.
        :param verbose: Whether to print computed class weights.
        """
        values = team_matches_data.loc[team_matches_data["id"].isin(fixtures_ids),
                                       self._target_variable.value].dropna().values

        # Add a single class sample to values if there is none (usually should not happen)
        if 0 not in values and not self.class_weights:
            values = np.append(values, 0)
        if 1 not in values and not self.class_weights:
            values = np.append(values, 1)

        cnt_total = {int(k): v for k, v in dict(sorted(Counter(values).items())).items()}
        cnt_ratio = {k: v / sum(cnt_total.values()) for k, v in cnt_total.items()}
        class_weights = compute_class_weight("balanced", np.unique(values), values)
        class_weights = dict(enumerate(class_weights))

        if verbose:
            print(f"cnt_total: {cnt_total}")
            print(f"cnt_ratio: {cnt_ratio}")
            print(f"class_weights: {class_weights}")

        if not self.class_weights:
            self.class_weights = class_weights

    def load_data_from_file(self, save_data: Dict[str, Any], load_optimizer: bool) -> None:
        """
        Loads previously saved model data.

        :param save_data: Previously saved data.
        :param load_optimizer: Whether to load optimzier state as well.
        """
        self.snapshot.load_params_from_file(save_data["snapshot_params"], save_data["snapshot_best_params"])
        self.snapshot.load_states_from_file(save_data["snapshot_states_after_training"])
        self.network.set_main_head_params(save_data["current_params"], include_optimizer=load_optimizer)
        self.best_params = save_data["best_params"]
        self.states_after_training = save_data["states_after_training"]
        self.epochs_since_improvement = save_data["epochs_since_improvement"]
        self.best_epoch = save_data["best_epoch"]
        self.best_acc = save_data["best_acc"]
        self.best_loss = save_data["best_loss"]

    def get_save_data(self) -> Dict[str, Any]:
        """
        Gathers data neeeded to properly save model.

        :return: Model data to save.
        """
        params, best_params = self.snapshot.serialize_params()

        return {
            "snapshot_params": params,
            "snapshot_best_params": best_params,
            "snapshot_states_after_training": self.snapshot.serialize_states(),
            "current_params": self.network.get_main_head_params(include_optimizer=True),
            "best_params": self.best_params,
            "states_after_training": self.states_after_training,
            "epochs_since_improvement": self.epochs_since_improvement,
            "best_epoch": self.best_epoch,
            "best_acc": self.best_acc,
            "best_loss": self.best_loss,
        }
