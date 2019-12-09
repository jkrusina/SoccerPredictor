from collections import Counter
from itertools import chain
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Any, Dict, List, Tuple

from soccerpredictor.trainer.dbmanager import SPDBManager
from soccerpredictor.util.common import get_unique_teams, get_last_season_unique_teams, check_season_gaps, \
    get_mismatched_teams
from soccerpredictor.util.config import SPConfig
from soccerpredictor.util.constants import *
from soccerpredictor.util.enums import Dataset


class SPDataLoader:
    """
    Loads fixtures data and teams data needed to run model.
    Performs basic checks for any nan values and preprocesses data properly (creates new columns,
    encodes teams names, scales features, etc.).

    Attributes:
        teams_names_bitlen: Bitlength required to encode all teams names.
        teams: All teams names.
        train_teams: Teams names occurring in train dataset.
        test_teams: Teams names occurring in test dataset.
        predict_teams: Teams names occurring in predict dataset.
        last_season_teams: Teams playing last season.
        train_teams_exclusively: Only train teams not appearing in test dataset.
        train_fixtures_ids: Fixtures ids belonging to train dataset.
        test_fixtures_ids: Fixtures ids belonging to test dataset.
        predict_fixtures_ids: Fixtures ids belonging to predict dataset.
        max_season: Max season number in dataset.
        max_ntest_len: Max number of test samples.
        max_npredict_len: Max number of predict samples.

    """

    def __init__(self, dbmanager: SPDBManager, seasons: List[int], model_settings: Dict[str, Any]) -> None:
        """

        :param dbmanager: Database manager.
        :param seasons: Seasons used.
        :param model_settings: Previous model settings.
        """
        config = SPConfig()
        self._dbmanager = dbmanager
        self._model_settings = {} if not model_settings else model_settings

        self._resume = config.resume
        self._seasons = seasons
        self._timesteps = config.timesteps
        self._verbose = config.verbose
        self._ntest = config.ntest
        self._ndiscard = config.ndiscard

        self._features = FEATURES_COMMON + FEATURES_WD
        self._scalers = {f: MinMaxScaler(feature_range=(0, 1)) for f in FEATURES_TO_SCALE if f in self._features}
        self._teams_names_lenc = self._fit_teams_names_lencoder()

        self.teams_names_bitlen = len(self._teams_names_lenc.classes_).bit_length()
        self.teams = []
        self.train_teams = []
        self.test_teams = []
        self.predict_teams = []
        self.last_season_teams = []
        self.train_teams_exclusively = []
        self.train_fixtures_ids = {}
        self.test_fixtures_ids = {}
        self.predict_fixtures_ids = {}
        self.max_season = None
        self.max_ntest_len = 0
        self.max_npredict_len = 0

    def load_and_process_fixtures_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads fixtures data ordered by date.

        The split guarantees that there will be at least N samples for predict dataset given by
        NPREDICT. Also, there will approx. the same number on test samples given by ntest argument.

        The exact number of required samples for test/predict dataset cannot be guaranteed due to
        different number of predict/test/discard samples because matches are not always played
        successively, so some teams can play more often within a certain period of time.

        Thus, in order to avoid overlapping datasets, it is probably impossible to ensure the exact
        numbers of samples when doing backtesting in general.

        :return: Train, test, and predict fixtures datasets.
        """
        df = self._dbmanager.query_fixtures_data(self._seasons)
        if df.empty:
            raise ValueError("Empty fixtures dataframe.")

        df = self._drop_last_season_championship_matches(df)

        self.teams = get_unique_teams(df)
        self.last_season_teams = get_last_season_unique_teams(df)
        # Get fixtures ids for each team
        teams_fixtures_ids = {t: df[(df["home"] == t) | (df["away"] == t)].loc[:, "id"].tolist() for t in self.teams}

        self._check_missing_columns(df)
        df = self._check_nan_values(df, teams_fixtures_ids)
        teams_fixtures_ids = self._discard_matches(df, teams_fixtures_ids)

        if not self._resume:
            # Use last n ids for predictions and last m ids for testing
            for t in self.last_season_teams:
                self.predict_fixtures_ids[t] = teams_fixtures_ids[t][-NPREDICT:]
                teams_fixtures_ids[t] = teams_fixtures_ids[t][:-NPREDICT]

                self.test_fixtures_ids[t] = teams_fixtures_ids[t][-self._ntest:]
                teams_fixtures_ids[t] = teams_fixtures_ids[t][:-self._ntest]

            # Rest of ids is counted as train set
            self.train_fixtures_ids = teams_fixtures_ids
        else:
            if self.teams_names_bitlen != self._model_settings["teams_names_bitlen"]:
                raise ValueError("Current bitlength required to encode all teams names is higher than previous one.")

            # Check whether teams has not changed
            if self.teams != self._model_settings["teams"]:
                raise ValueError("Teams differ from previous run. \n"
                                 f"New: {self.teams} \n"
                                 f"Old: {self._model_settings['teams']}")

            if self.last_season_teams != self._model_settings["last_season_teams"]:
                raise ValueError("Last season teams differ from previous run. \n"
                                 f"New: {self.last_season_teams} \n"
                                 f"Old: {self._model_settings['last_season_teams']}")

            # Check whether fixtures ids match from previous run
            for t in self.last_season_teams:
                predict_fixtures_ids = teams_fixtures_ids[t][-NPREDICT:]
                teams_fixtures_ids[t] = teams_fixtures_ids[t][:-NPREDICT]

                test_fixtures_ids = teams_fixtures_ids[t][-self._ntest:]
                teams_fixtures_ids[t] = teams_fixtures_ids[t][:-self._ntest]

                if predict_fixtures_ids != self._model_settings["predict_fixtures_ids"][t]:
                    raise ValueError(f"{t} predict fixtures ids differ from previous run. \n"
                                     f"New: {predict_fixtures_ids} \n"
                                     f"Old: {self._model_settings['predict_fixtures_ids'][t]}")
                if test_fixtures_ids != self._model_settings['test_fixtures_ids'][t]:
                    raise ValueError(f"{t} test fixtures ids differ from previous run. \n"
                                     f"New: {test_fixtures_ids} \n"
                                     f"Old: {self._model_settings['test_fixtures_ids'][t]}")
                if teams_fixtures_ids[t] != self._model_settings['train_fixtures_ids'][t]:
                    raise ValueError(f"{t} train fixtures ids differ from previous run. \n"
                                     f"New: {teams_fixtures_ids[t]} \n"
                                     f"Old: {self._model_settings['train_fixtures_ids'][t]}")

            # Checks passed, load previously saved data
            self.teams = self._model_settings["teams"]
            self.last_season_teams = self._model_settings["last_season_teams"]
            self.train_fixtures_ids = self._model_settings["train_fixtures_ids"]
            self.test_fixtures_ids = self._model_settings["test_fixtures_ids"]
            self.predict_fixtures_ids = self._model_settings["predict_fixtures_ids"]

        self._check_season_gaps_in_teams_matches(df)

        # Split original dataset into train, test, and predict datasets
        df_train, df_test, df_predict = self._mask_out_dataset(df)
        self._get_unique_teams_from_datasets(df_train, df_test, df_predict)

        self._check_changes_in_teams()
        self._count_samples(df_train, df_test, df_predict)

        return df_train, df_test, df_predict

    def _drop_last_season_championship_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops all matches from CH league for last season to speed up training a little bit.
        They wont be needed since we care only about PL teams.

        :param df: Original dataset.
        :return: Dataset without last season CH teams fixtures.
        """
        max_season = df["season"].max()

        if self._resume:
            # Check if max season changed
            if max_season != self._model_settings["max_season"]:
                raise ValueError("Current max season is different than previous one. Model must be retrained.")
            else:
                self.max_season = self._model_settings["max_season"]
        else:
            self.max_season = max_season

        return df.loc[~((df["season"] == max_season) & (df["league"] == "CH"))]

    def _discard_matches(self,
                         df: pd.DataFrame,
                         teams_fixtures_ids: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """
        Discards fixtures for each team if ndiscard arg specified.
        Will discard at least N given fixtures ids from each team. There may be cases that more matches
        will be discarded to proper align matches sequences.

        :param df: Original dataframe.
        :param teams_fixtures_ids: Original fixtures ids.
        :return: Modified fixtures ids.
        """
        discarded_fixtures_ids = {}

        if self._ndiscard:
            for t in self.last_season_teams:
                discarded_fixtures_ids[t] = teams_fixtures_ids[t][-self._ndiscard:]
                teams_fixtures_ids[t] = teams_fixtures_ids[t][:-self._ndiscard]

            # Check if all discarded matches are within a single season
            unique_discarded_fixtures_ids = set(chain(*discarded_fixtures_ids.values()))
            mask_discarded = df["id"].isin(unique_discarded_fixtures_ids)
            seasons_in_discarded_df = df[mask_discarded].loc[:, "season"].unique()

            if any(s != self.max_season for s in seasons_in_discarded_df):
                raise ValueError("Discarded matches must be within the last season. \n"
                                 "Please lower the number of matches to discard.")

            # Some teams may have played more than once so we need to be sure that we discarded
            # all matches correctly.
            for t in self.last_season_teams:
                teams_fixtures_ids[t] = [j for j in teams_fixtures_ids[t] if j not in unique_discarded_fixtures_ids]

        return teams_fixtures_ids

    def _mask_out_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits original dataset into train, test, and predict datasets by masking the original dataset
        with fixtures ids belonging to corresponding dataset types.

        Each dataset will contain only matches which belong to particular dataset.
        In order to use timesteps sequences correctly, <timesteps> matches of previous dataset
        needs to be copied into the subsequent dataset (i.e. from train to test dataset, and from
        test to predict dataset). This is done later in SPTrainer's _preload method.

        :param df: Original dataset.
        :return: Whole dataset split into train, test, and predict datasets.
        """
        mask_train = df["id"].isin(set(chain(*self.train_fixtures_ids.values())))
        mask_test = df["id"].isin(set(chain(*self.test_fixtures_ids.values())))
        mask_predict = df["id"].isin(set(chain(*self.predict_fixtures_ids.values())))

        df_train = df[mask_train & ~mask_test & ~mask_predict].copy()
        df_test = df[mask_test & ~mask_predict].copy()
        df_predict = df[mask_predict].copy()

        if self._verbose > 0:
            print("Teams used for testing:")
            print(sorted(self.test_fixtures_ids.keys()))
            print("Teams used for prediction:")
            print(sorted(self.predict_fixtures_ids.keys()))
            print(f"Train dataset total samples: {len(df_train)}")
            print(f"Test dataset total samples: {len(df_test)}")
            print(f"Predict dataset total samples: {len(df_predict)}")

        # Additional argument checking which can be done only after computing individual datasets.
        # Check if datasets are empty (there is a check during parsing arguments that test/predict
        # split samples must be at least 1, but some dataset may become empty in the end due to some
        # restrictions and filtering).
        emsg = "Maybe try to specify the split of test and/or prediction samples more reasonably?"
        if df_train.empty:
            raise ValueError(f"Train dataset is empty. {emsg}")
        if df_test.empty:
            raise ValueError(f"Test dataset is empty. {emsg}")
        if df_predict.empty:
            raise ValueError(f"Predict dataset is empty. {emsg}")
        # Check if datasets are too large (e.g. predict dataset is larger than the rest, etc.)
        if len(df_predict) > len(df_test) + len(df_train):
            raise ValueError(f"Number of samples in predict dataset is too large. {emsg}")
        if len(df_test) > len(df_train):
            raise ValueError(f"Number of samples in test dataset is too large. {emsg}")

        return df_train, df_test, df_predict

    def _check_season_gaps_in_teams_matches(self, df: pd.DataFrame):
        """
        Checks whether to omit some teams from either test or predict datasets due to season gaps.

        :param df: Original dataframe.
        """
        if self._verbose > 0:
            print("Teams:")
            print(self.teams)
            print("Teams playing last season:")
            print(self.last_season_teams)

        omitted_teams_test = check_season_gaps(df, self.test_fixtures_ids)
        omitted_teams_predict = check_season_gaps(df, self.predict_fixtures_ids)

        if omitted_teams_predict or omitted_teams_test:
            print("Warning: There are some teams with season gaps in data.")
            print("Teams omitted based on predict dataset due to season gap:")
            print(omitted_teams_predict)
            print("Teams omitted based on test dataset due to season gap:")
            print(omitted_teams_test)

            # Delete these teams from datasets
            for team in set(omitted_teams_test + omitted_teams_predict):
                del self.test_fixtures_ids[team]
                del self.predict_fixtures_ids[team]

            print("You should consider lowering required number of test or predict data samples.")

    def _get_unique_teams_from_datasets(self,
                                        df_train: pd.DataFrame,
                                        df_test: pd.DataFrame,
                                        df_predict: pd.DataFrame) -> None:
        """
        Gets unique teams names from dataframes for each dataset type.

        :param df_train: Train dataset.
        :param df_test: Test dataset.
        :param df_predict: Predict dataset.
        """
        self.train_teams = get_unique_teams(df_train)
        self.test_teams = get_unique_teams(df_test)
        self.predict_teams = get_unique_teams(df_predict)
        self.train_teams_exclusively = sorted(list(set(self.train_teams) - set(self.test_teams)))

    def _count_samples(self, df_train: pd.DataFrame, df_test: pd.DataFrame, df_predict: pd.DataFrame) -> pd.DataFrame:
        """
        Counts number of samples for each team in each dataset.

        :param df_train: Train dataset.
        :param df_test: Test dataset.
        :param df_predict: Predict dataset.
        :return: Number of samples for each in datasets.
        """
        cols = ["home", "away"]
        cnt_samples_train = Counter(df_train[cols].values.ravel("K"))
        cnt_samples_test = Counter(df_test[cols].values.ravel("K"))
        cnt_samples_predict = Counter(df_predict[cols].values.ravel("K"))

        # Get max samples for test and predict datasets
        self.max_ntest_len = cnt_samples_test.most_common(1)[0][1]
        self.max_npredict_len = cnt_samples_predict.most_common(1)[0][1]

        counters = [cnt_samples_train.items(), cnt_samples_test.items(), cnt_samples_predict.items()]
        cols_names = ["train", "test", "predict"]

        # Fill value with "-" if the team is not in dataset
        df_samples = pd.DataFrame()
        for cnt, col in zip(counters, cols_names):
            df_samples = pd.concat((df_samples,
                                    pd.DataFrame.from_dict(dict(cnt), orient="index", columns=[col])),
                                   axis=1, sort=True).fillna("-")

        if self._verbose > 0:
            print("Counter of dataset samples for each team:")
            print(df_samples)

        # Check if required number of timesteps is not greater than max number of datapoints for any team
        if self._timesteps >= df_samples["train"].min():
            raise ValueError(f"Number of timesteps is >= than lowest number of samples in train dataset for "
                             f"some teams. \n"
                             f"Timesteps required: {self._timesteps}, max allowable: "
                             f"{df_samples['train'].min()}")

        return df_samples

    def load_and_process_team_data(self,
                                   dataset: Dataset,
                                   teamid: int,
                                   team_fixtures_idx: List[int]) -> pd.DataFrame:
        """
        Loads team data for the specified team where the team played either as home or away.
        Loads all data for particular team regardless of dataset.

        :param dataset: Current dataset to load team data for.
        :param teamid: Selected team's id.
        :param team_fixtures_idx: List of fixtures indices in which the team played.
        :return: Loaded data.
        """
        df = self._dbmanager.query_team_data(self._seasons, params=(teamid, teamid, teamid, *self._seasons))

        # Current win-or-draw target from team's POV
        df["wd"] = np.select(
            condlist=[
                (df["homeTeamID"] == teamid) & (df["winner"] == 1),
                (df["homeTeamID"] == teamid) & (df["winner"] == 2),
                (df["awayTeamID"] == teamid) & (df["winner"] == 2),
                (df["awayTeamID"] == teamid) & (df["winner"] == 1),
                df["winner"] == 0
            ],
            choicelist=[
                1,
                0,
                1,
                0,
                1
            ],
            default=np.nan
        )
        # Odds for wd from team's POV
        df["odds_wd"] = np.select(
            condlist=[
                df["homeTeamID"] == teamid,
                df["awayTeamID"] == teamid
            ],
            choicelist=[
                df["oddsDC_1X"],
                df["oddsDC_X2"]
            ],
            default=np.nan
        )
        # `team` is just name of current team
        df["team"] = np.select(
            condlist=[
                df["homeTeamID"] == teamid,
                df["awayTeamID"] == teamid
            ],
            choicelist=[
                df["home"],
                df["away"]
            ],
            default=np.nan
        )
        # `opponent` is the opposite team to `team`
        df["opponent"] = np.select(
            condlist=[
                df["homeTeamID"] == teamid,
                df["awayTeamID"] == teamid
            ],
            choicelist=[
                df["away"],
                df["home"]
            ],
            default=np.nan
        )
        # Whether the current team plays as home = 1, or away = 0
        df["ashome"] = np.select(
            condlist=[
                df["homeTeamID"] == teamid,
                df["awayTeamID"] == teamid,
            ],
            choicelist=[
                1,
                0
            ],
            default=np.nan
        )
        # Encode league: PL = 1, CH = 0
        df["league"] = np.select(
            condlist=[
                df["league"] == "PL",
                df["league"] == "CH"
            ],
            choicelist=[
                1,
                0
            ],
            default=np.nan
        )

        # Create future values of features which are known in advance.
        # They are created by shifting original values by one row.
        df["future_season"] = df["season"].shift(-1)
        df["future_league"] = df["league"].shift(-1)
        df["future_ashome"] = df["ashome"].shift(-1)
        df["future_opponent"] = df["opponent"].shift(-1)
        df["future_wd"] = df["wd"].shift(-1)
        df["future_odds_wd"] = df["odds_wd"].shift(-1)

        # Dataframe needs to be filtered according to current dataset to contain only corresponding
        # matches. Also index needs to be reset (to correctly access data for test and predict sets).
        df = df.loc[df["id"].isin(team_fixtures_idx)].copy()
        df.reset_index(inplace=True, drop=True)

        # If there are nans for train dataset, use ffill method to fill them.
        # This can happen if team occurs only in train dataset and does not have more data in later
        # seasons, thus it does not matter that we fill these nans with incorrect data as they wo not be
        # used for testing anyway.
        if dataset == Dataset.Train:
            # Fill nan values of future opponent with empty string
            df["future_opponent"].fillna("", inplace=True)
            df.fillna(method="ffill", inplace=True)

        df = self._scale_team_data(df)

        return df

    def _scale_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scales values in given dataframe and encodes team names.

        Transforms only non-nan values, nan values should only occur for predict dataset and
        we do not need these values for prediction.

        :param df: Dataframe to scale.
        :return: Dataframe with scaled values.
        """
        for f in FEATURES_TO_LENC:
            nonnull_index = ~df[f].isnull()
            df.loc[nonnull_index, f] = df.loc[nonnull_index, f].apply(self._transform_team_name)

        for f, scaler in self._scalers.items():
            nonnull_index = ~df[f].isnull()
            df.loc[nonnull_index, f] = scaler.transform(df.loc[nonnull_index, f].values.reshape(-1, 1))

            # Also transform columns with "future_" prefix with the same scalers
            futuref = f"future_{f}"
            if futuref in df.columns:
                nonnull_index = ~df[futuref].isnull()
                df.loc[nonnull_index, futuref] = scaler.transform(df.loc[nonnull_index, futuref].values.reshape(-1, 1))

        return df

    def inverse_transform(self, feature_name: str, value: Any) -> Any:
        """
        Rescales given feature value(s).

        :param feature_name: Feature name to rescale.
        :param value: Feature value(s) to rescale.
        :return: Rescaled feature value(s).
        """
        return self._scalers[feature_name].inverse_transform(value.reshape(-1, 1)).flatten()[0]

    def fit_scalers(self, df: pd.DataFrame) -> None:
        """
        Fits scalers on columns with the same name.
        Features specified in FEATURES_TO_SCALE are fit on columns with both "home_" and "away_" prefixes.

        :param df: Dataframe on which scalers should be fit.
        """
        for feature, scaler in self._scalers.items():
            if feature == "season":
                scaler.fit(df["season"].unique().reshape(-1, 1))
            elif feature in FEATURES_TO_SCALE:
                values = np.concatenate((df[f"home_{feature}"].values, df[f"away_{feature}"].values))
                scaler.fit(np.unique(values).reshape(-1, 1))
            else:
                scaler.fit(df[feature].unique().reshape(-1, 1))

    def _fit_teams_names_lencoder(self) -> LabelEncoder:
        """
        Fits a LabelEncoder for teams names.
        Queries teams names from db and fits encoder to map the names from strings to integers
        starting from 0 (including empty string for possible unknown/missing team name).

        :return: Fit LabelEncoder.
        """
        df = self._dbmanager.query_teams_names()
        values = [""] + df["name"].values.tolist()
        return LabelEncoder().fit(values)

    def _transform_team_name(self, team_name: str) -> np.ndarray:
        """
        Transforms team name from string representation to binary represenatation as
        numpy array. Uses already fit LabelEncoder.
        Width is set to min number of bits needed to encode all teams.

        E.g.: "Arsenal" -> 38 -> np.array([1, 0, 0, 1, 1, 0])

        :param team_name: Team name to transform.
        :return: Team name as binary array.
        """
        return np.array(list(np.binary_repr(self._teams_names_lenc.transform([team_name])[0],
                                            width=len(self._teams_names_lenc.classes_).bit_length())),
                        dtype=int)

    def _check_missing_columns(self, df: pd.DataFrame) -> None:
        """
        Basic check whether any of required columns is missing in the dataframe.

        :param df: Dataframe to be checked.
        """
        if any([c not in df.columns for c in REQUIRED_COLUMNS]):
            raise ValueError("Missing columns in dataset."
                             f"Columns: {df.columns}"
                             f"Required: {REQUIRED_COLUMNS}")

    def _check_nan_values(self, df: pd.DataFrame, teams_fixtures_ids: Dict[str, List[int]]) -> pd.DataFrame:
        """
        Checks whether there are any missing values in the dataset.

        Also fills columns with zeros for unanvailable features "rating" and "errors" if
        these columns are empty.

        :param df: Dataframe to be checked.
        :param teams_fixtures_ids: List of fixtures ids for each team.
        :return: Whether dataset contains any problematic nan values.
        """
        teams_fixtures_lastid = {k: v[-1] for k, v in teams_fixtures_ids.items()}

        mask_lastids = df["id"].isin(set(teams_fixtures_lastid.values()))
        df_except_lastid = df[~mask_lastids].copy()
        df_lastid_only = df[mask_lastids][REQUIRED_TARGET_COLUMNS]

        # Check missing values in data except for teams' last matches (targets)
        if df_except_lastid.isna().any().any():
            print("Missing values found in the dataset:")
            print(df_except_lastid.isna().sum())

            # If there are still nans then there are another missing data
            if df_except_lastid.isna().any().any():
                print("There are still missing values in the dataset:")
                print(df_except_lastid.isna().sum())
                raise ValueError("Dataset contains some missing values.")

        # Check if last matches (targets) contain any nans for required columns
        if df_lastid_only.isna().any().any():
            print(df_lastid_only.isna().sum())
            raise ValueError("Dataset contains some missing target values.")

        return df

    def _check_changes_in_teams(self) -> None:
        """
        Checks whether the current teams loaded from dataframe are the same as from the last run.

        If they do not match then the model has to be retrained, mainly because we dropped all teams from CH league
        for last season and the models are not trained on them.
        This can happen when new season starts and teams playing in that season differ.

        """
        if not self._resume:
            return

        if self.train_teams != self._model_settings["train_teams"]:
            mismatched_teams = get_mismatched_teams(self.train_teams, self._model_settings["train_teams"])
            raise ValueError(f"Teams in train dataset differ from previous ones. Model must be retrained. \n"
                             f"Teams that mismatch: {mismatched_teams}")

        if self.test_teams != self._model_settings["test_teams"]:
            mismatched_teams = get_mismatched_teams(self.test_teams, self._model_settings["test_teams"])
            raise ValueError(f"Teams in test dataset differ from previous ones. Model must be retrained. \n"
                             f"Teams that mismatch: {mismatched_teams}")

        if self.predict_teams != self._model_settings["predict_teams"]:
            mismatched_teams = get_mismatched_teams(self.predict_teams, self._model_settings["predict_teams"])
            raise ValueError(f"Teams in predict dataset differ from previous ones. Model must be retrained. \n"
                             f"Teams that mismatch: {mismatched_teams}")
