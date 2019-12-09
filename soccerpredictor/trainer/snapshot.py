from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Tuple


class SPSnapshot:
    """
    Holds main head params of all other teams' networks from viewpoint of current team.
    Every time some team's model performance improves, the new weights and states (params) are propagated
    into snapshot of every model. This ensures that the current team will always operate with the best
    weights of other models. Otherwise bad weight updates of other models could impair weight updates of
    the current model.
    So, the weights of teams in snapshot remains fixed, but states are properly updated during each call to
    train, test or predict.
    When current model improves, the snapshot params are stored as best_params, so the model can revert
    to that snapshot if needed.

    Attributes:
        params: The best weights and states of main head of each team. Renamed as "head2_*" so they
                are ready to be loaded into the second head of current team's network.
        best_params: Contains copy of the best recorded params so far.
        states_after_training: Copy of RNNs' states right after the last training (before testing).

    """

    def __init__(self) -> None:
        """
        Parameters contain only "head2" weights.

        """
        self.params = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        self.best_params = None
        self.states_after_training = defaultdict(lambda: defaultdict())

    def reset_states(self) -> None:
        """
        Resets states of all teams (by zeroing out arrays).

        """
        for team in self.params.keys():
            for lname in self.params[team]["states"].keys():
                self.params[team]["states"][lname][0][:] = 0
                self.params[team]["states"][lname][1][:] = 0

    def save_states_after_training(self) -> None:
        """
        Saves exact head2 states of all teams after finishing training in the current epoch.
        Should be called directly after training (before any testing or predicting).

        """
        for team in self.params.keys():
            self.states_after_training[team] = self.params[team]["states"]

    def restore_states_after_training(self) -> None:
        """
        Restores previously saved head2 states of all teams after the training.

        """
        for team in self.states_after_training.keys():
            self.params[team]["states"] = self.states_after_training[team]

    def revert_to_best_params(self) -> None:
        """
        Restores the best head2 params recorded so far.

        """
        self.params = deepcopy(self.best_params)

    def record_best_params(self) -> None:
        """
        First, restores head2 states after training (because they have changed due to testing),
        then saves them.
        Should be called only after testing.

        """
        self.restore_states_after_training()
        self.best_params = deepcopy(self.params)

    def set_initial_params(self, params_all_teams: Dict[str, Dict[str, Any]]) -> None:
        """
        Sets default weights of *all* teams.
        Sets weights and states from main head (named head1_*) as head2 (named head2_*).
        Used during first run only.

        Params should have following structure:
            params[<team_name>]["weights"]["head2_*"] = ...
                               ["states"]["head2_*"] = ...
                      ...

        :param params_all_teams: Params of all teams.
        """
        for team, params in params_all_teams.items():
            for lname, w in params["weights"].items():
                if "head1" in lname:
                    self.params[team]["weights"][lname.replace("head1", "head2")] = w
            for lname, s in params["states"].items():
                if "head1" in lname:
                    self.params[team]["states"][lname.replace("head1", "head2")] = s

        self.best_params = params_all_teams

    def load_params_from_file(self,
                              params: Dict[str, Dict[str, Any]],
                              best_params: Dict[str, Dict[str, Any]]) -> None:
        """
        Loads params from file.

        :param params: Parameters to load.
        :param best_params: The best parameters to load.
        """
        self.params = params
        self.best_params = best_params

    def load_states_from_file(self, states: Dict[str, Any]) -> None:
        """
        Loads states (from the moment after training) from file.

        :param states: States to load.
        """
        self.states_after_training = states

    def update_states(self, team: str, states: Dict[str, Any]) -> None:
        """
        Updates head2 states of given team.

        :param team: Team to be updated.
        :param states: States to update.
        """
        self.params[team]["states"] = states

    def update_weights(self, team: str, weights: Dict[str, Any]) -> None:
        """
        Updates head2 weights of given team.
        Weights given are from the main head and their prefix needs to be renamed from "head1" to "head2".

        :param team: Team to be updated.
        :param weights: Weights to update.
        """
        for lname, w in weights.items():
            self.params[team]["weights"][lname.replace("head1", "head2")] = w

    def serialize_states(self) -> Dict[str, Any]:
        """
        Serializes states from nested defaultdict into dict.

        :return: States as dict.
        """
        return {team: dict(states) for team, states in self.states_after_training.items()}

    def serialize_params(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Serializes params from nested defaultdicts into dicts.

        :return: Params and the best params as dicts.
        """
        params_dict = {team: {} for team in self.params.keys()}
        best_params_dict = {team: {} for team in self.best_params.keys()}

        for team, p in self.params.items():
            params_dict[team]["weights"] = p["weights"]
            params_dict[team]["states"] = p["states"]

        for team, p in self.best_params.items():
            best_params_dict[team]["weights"] = p["weights"]
            best_params_dict[team]["states"] = p["states"]

        return params_dict, best_params_dict
