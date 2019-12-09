import json
import os
import pandas as pd
from pathlib import Path
import re
from typing import Any, Dict, List

from soccerpredictor.util.constants import *
from soccerpredictor.util.enums import Dataset


def compressed_df_format(df: pd.DataFrame, teams_names_len: int = 5) -> pd.DataFrame:
    """
    Prints dataframe in a more compressed format.
    Shortens teams names and formats loss and acc values. Some teams names can have the same prefix
    when shortened (e.g. West_Ham, West_Brom -> West_) and pandas prints only one of them.

    :param df: Dataframe to be compressed.
    :param teams_names_len: Length to shorten team names to.
    :return: Compressed dataframe.
    """
    compressed_df = df.copy()
    teams = pd.unique(df.columns.get_level_values("team"))

    for t in teams:
        subset = ~compressed_df.loc[:, (t, "loss")].isnull()
        compressed_df.loc[subset, (t, "loss")] = compressed_df.loc[subset, (t, "loss")].map("{:.3f}".format)
        compressed_df.loc[subset, (t, "acc")] = compressed_df.loc[subset, (t, "acc")].map("{:.0%}".format)

    return compressed_df.rename(columns=lambda x: x[:teams_names_len], level="team")


def get_model_settings_file(path: Path) -> Dict[str, Any]:
    """
    Loads a previously saved model settings file.

    :param path: Path to folder.
    :return: Model settings file.
    """
    p = path.joinpath(MODEL_SETTINGS_FILE)

    if p.is_file():
        print("Loading previous model settings file.")
        with open(str(p), "r") as fh:
            f = json.load(fh)
        if not f:
            raise FileNotFoundError("Model setttings file is empty.")

        return f
    else:
        raise FileNotFoundError("Could not load model settings file.")


def get_prediction_file(path: Path, predict_dataset: Dataset) -> pd.DataFrame:
    """
    Loads a previously saved prediction file.

    :param path: Path to folder.
    :param predict_dataset: Type of prediction file.
    :return: Prediction file.
    """
    if predict_dataset in [Dataset.Test, Dataset.Predict]:
        p = path.joinpath(f"{predict_dataset.value}_dataset_{PREDICT_STATS_FILE}")
    else:
        raise ValueError("Unknown prediction dataset.")

    if p.is_file():
        print(f"Loading previous prediction file of '{predict_dataset.value}' dataset.")
        f = pd.read_pickle(str(p))
        if f.empty:
            raise FileNotFoundError(f"Prediction file for '{predict_dataset.value}' dataset is empty.")

        return f
    else:
        raise ValueError(f"No previous prediction file of '{predict_dataset.value}' dataset found.")


def get_best_stats_file(path: Path, dataset: Dataset) -> pd.DataFrame:
    """
    Loads a previously saved best stats file.

    :param path: Path to folder.
    :param dataset: Type of best stats file.
    :return: Best stats file.
    """
    if dataset == Dataset.Train:
        p = path.joinpath(BEST_TRAIN_STATS_FILE)
    elif dataset == Dataset.Test:
        p = path.joinpath(BEST_TEST_STATS_FILE)
    else:
        raise ValueError("Unknown best stats dataset.")

    if p.is_file():
        print(f"Loading previous '{dataset.value}' dataset best stats file.")
        f = pd.read_pickle(str(p))
        if f.empty:
            raise FileNotFoundError(f"Best stats file for '{dataset.value}' dataset is empty.")

        return f
    else:
        raise FileNotFoundError(f"No previous '{dataset.value}' dataset best stats file found.")


def get_stats_file(path: Path, dataset: Dataset) -> pd.DataFrame:
    """
    Loads a previously saved stats file.

    :param path: Path to folder.
    :param dataset: Type of stats file.
    :return: Stats file.
    """
    if dataset == Dataset.Train:
        p = path.joinpath(TRAIN_STATS_FILE)
    elif dataset == Dataset.Test:
        p = path.joinpath(TEST_STATS_FILE)
    else:
        raise ValueError("Unknown stats dataset.")

    if p.is_file():
        print(f"Loading previous '{dataset.value}' dataset stats file.")
        f = pd.read_pickle(str(p))
        if f.empty:
            raise FileNotFoundError(f"Stats file for '{dataset.value}' dataset is empty.")

        return f
    else:
        raise FileNotFoundError(f"No previous '{dataset.value}' dataset stats file found.")


def get_latest_models_dir(name: str = "") -> Path:
    """
    Gets name of latest model dir sorted by time of creation.

    Name argument can be specified to load exact folder or latest folder with given prefix.

    :param name: Exact name, or a part of name of the model to load.
    :return: Latest models dir path.
    """
    path = Path(os.getcwd()).joinpath(f"{DATA_DIR}{MODEL_DIR}")
    if name:
        name = Path(name).name

    print(f"Checking path: '{path.resolve()}'")

    # Name specified
    if name:
        # Check whether full name is specified and whether that name exists
        if re.match(FOLDER_NAME_PATTERN, name):
            p = path.joinpath(name)
            if p.is_dir():
                print(f"Specified folder found, loading from: '{p.name}'")
                return p
            else:
                raise FileNotFoundError("Could not find specified folder.")
        else:
            # Full name was not specified so try to load folders which starts with given
            # prefix and load the most current one
            folders = []

            for f in os.listdir(path):
                p = path.joinpath(f)
                if f.startswith(name.upper()) and p.is_dir():
                    folders.append(p)

            if folders:
                current_dir = max(folders, key=os.path.getctime)
                print(f"The most current folder with given name '{name}' found, loading from: '{current_dir.name}'.")
                return current_dir
            else:
                raise FileNotFoundError(f"Could not find any folders with given name '{name}'.")
    else:
        # Name unspecified, load the most current folder
        folders = []

        for f in os.listdir(path):
            p = path.joinpath(f)
            if p.is_dir():
                folders.append(p)

        if folders:
            current_dir = max(folders, key=os.path.getctime)
            print(f"The most current folder found, loading from: '{current_dir.name}'")
            return current_dir
        else:
            raise FileNotFoundError("Could not find any folders.")


def get_unique_teams(df: pd.DataFrame) -> List[str]:
    """
    Gets a list of unique teams names in the dataframe.

    :param df: Dataframe to be processed.
    :return: Sorted list of unique teams names.
    """
    if {"home", "away"}.issubset(df.columns):
        teams = sorted(pd.unique(df[["home", "away"]].values.ravel("K")))
        if not teams:
            raise FileNotFoundError("Teams columns from dataframe are empty.")

        return teams
    else:
        raise ValueError("Dataframe does not contain 'home' and 'away' columns.")


def get_last_season_unique_teams(df: pd.DataFrame) -> List[str]:
    """
    Gets a list of unique teams names in the dataframe from the last season of PL.

    :param df: Dataframe to be processed.
    :return: Sorted list of unique teams names from the last season.
    """
    df_filtered = df.loc[(df["season"] == df["season"].unique().max()) & (df["league"] == "PL")]
    teams_filtered = sorted(get_unique_teams(df_filtered))

    return teams_filtered


def get_fixtures_ids_from_df(df: pd.DataFrame, team: str) -> List[int]:
    """
    Gets fixtures ids from dataframe for particular team.
    Both home and away sides count.

    :param df: Dataframe to extract ids from.
    :param team: Team name.
    :return: List of fixtures ids where the team played in.
    """
    return df[(df["home"] == team) | (df["away"] == team)].loc[:, "id"].tolist()


def check_season_gaps(df: pd.DataFrame,
                      reserved_fixtures_ids: Dict[str, List[int]]) -> List[str]:
    """
    Checks whether there are any seasons gaps in datasets.
    Season gaps could be problematic (e.g. when team drops to lower league than we have
    data for and reappers later), it could lead to mixing matches between train and test set
    date periods. Hence, it is better to drop these teams completely from testing and prediction.

    This can happen if there are too many datapoints required for prediction and/or testing (they
    are spread across more than one season), or at the end of season/start of the new one.

    Any teams found will be omitted from prediction and testing.

    :param df: Dataframe to be checked.
    :param reserved_fixtures_ids: Fixtures ids reserved for testing or prediction.
    :return: List of omitted teams names.
    """
    omitted_teams = []

    for team, matches_ids in reserved_fixtures_ids.items():
        matches_ids_reversed = matches_ids[::-1]
        # Get season of the latest match
        current_season = int(df.loc[(df["id"] == matches_ids_reversed[0]), "season"])

        # Loop over matches
        for m_id in matches_ids_reversed[1:]:
            # Get season of the current match
            season = int(df.loc[(df["id"] == m_id), "season"])

            # Check if there is a season gap between last two matches, if yes then
            # omit this team, otherwise update current season
            if season != current_season:
                if season < current_season-1:
                    omitted_teams.append(team)
                    break

                current_season = season

    return omitted_teams


def align_fixtures_ids(df: pd.DataFrame,
                       team: str,
                       fixtures_ids: List[int],
                       timesteps: int) -> List[int]:
    """
    Gets last <timesteps> datapoints from previous dataset to correctly offset matches sequence.

    E.g.: If timesteps == 5 then we would copy last 5 datapoints from train dataset and insert them
    at the beggining of test dataset. So when testing is run then these 5 datapoints are used to make
    prediction on the first test datapoint.
    The copied datapoints are also excluded from training to avoid data leakage. This is done in the

    The same applies for aligning test dataset and predict dataset.

    :param df: Dataset to copy from.
    :param team: Current team.
    :param fixtures_ids: List of original fixtures ids.
    :param timesteps: Number of timesteps used.
    :return: Correctly aligned matches ids.
    """
    aligned_matches = df[(df["home"] == team) | (df["away"] == team)][-timesteps:]
    aligned_fixtures_ids = get_fixtures_ids_from_df(aligned_matches, team)

    return aligned_fixtures_ids + fixtures_ids


def form_model_settings_file_path(generic_path: Path) -> Path:
    """
    Forms path for model settings file.

    :param generic_path: Path to common output dir.
    :return: Output path for model settings file.
    """
    return generic_path.joinpath(MODEL_SETTINGS_FILE)


def form_prediction_file_path(generic_path: Path, predict_dataset: Dataset) -> Path:
    """
    Forms path for prediction file.

    :param generic_path: Path to common output dir.
    :param predict_dataset: Type of prediction dataset.
    :return: Output path for prediction file.
    """
    return generic_path.joinpath(f"{predict_dataset.value}_dataset_{PREDICT_STATS_FILE}")


def form_best_stats_file_path(generic_path: Path, dataset: Dataset) -> Path:
    """
    Forms path for best stats file.

    :param generic_path: Path to common output dir.
    :param dataset: Type of dataset for best stats file.
    :return: Output path for model's best stats.
    """
    if dataset == Dataset.Train:
        path = generic_path.joinpath(BEST_TRAIN_STATS_FILE)
    elif dataset == Dataset.Test:
        path = generic_path.joinpath(BEST_TEST_STATS_FILE)
    else:
        raise ValueError("Unknown dataset for best stats file.")

    return path


def form_stats_file_path(generic_path: Path, dataset: Dataset) -> Path:
    """
    Forms path for stats file.

    :param generic_path: Path to common output dir.
    :param dataset: Type of dataset for stats file.
    :return: Output path for model's stats.
    """
    if dataset == Dataset.Train:
        path = generic_path.joinpath(TRAIN_STATS_FILE)
    elif dataset == Dataset.Test:
        path = generic_path.joinpath(TEST_STATS_FILE)
    else:
        raise ValueError("Unknown dataset for stats file.")

    return path


def form_data_file_path(generic_path: Path, team: str) -> Path:
    """
    Forms path for team's model params.

    :param generic_path: Path to common output dir.
    :param team: Team name.
    :return: Output path for models' params.
    """
    return generic_path.joinpath(f"{team}_{DATA_FILE}")


def get_mismatched_teams(list1: List[str], list2: List[str]) -> List[str]:
    """
    Compares two lists of team names and finds teams that are not contained in both of them.

    :param list1: First list of teams.
    :param list2: Second list of teams.
    :return: List of teams names that are not contained in each of given lists.
    """
    return list((set(list1) | set(list2)) - (set(list1) & set(list2)))
