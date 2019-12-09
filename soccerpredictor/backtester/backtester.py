from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pandas as pd
from pathlib import Path
import re
from sklearn.metrics import accuracy_score
import sys
from typing import Any, Dict, List, Tuple

from soccerpredictor.util.common import get_prediction_file
from soccerpredictor.util.constants import *
from soccerpredictor.util.enums import Dataset


def load_predictions_files(path: str) -> List[Tuple[Path, pd.DataFrame, pd.DataFrame]]:
    """
    Searches all folders in given parent directory which match the models dir name pattern.
    If more folders with the same prefix are found then selects only the most current one (according to
    date of creation).

    E.g.: <parent_dir>: AILL_2019-11-26T02-20-22_400/
                        GNWJ_2019-11-26T02-13-16_400/
                        ...
                        PJKE_2019-11-26T11-13-12_400/

    However, folders might not be created in a way they should be going consecutive, so the
    function tries to sort them according to dates appearing in predict dataset prediction files.
    It is not guaranteed they will be sorted absolutely correctly due to possibly overlapping ranges of
    match dates.

    :param path: Parent directory where models dirs are searched for.
    :return: Test and prediction files found in all model dirs in given directory.
    """
    path = Path(path) if path else Path(os.getcwd()).joinpath(f"{DATA_DIR}{MODEL_DIR}")

    print("---")
    print(f"Loading from directory: '{path}'")
    print("---")
    print("Loading files...")
    # Select all folders which match the name pattern and group them by their prefixes
    folders_groups = defaultdict(list)
    for f in os.listdir(path):
        p = path.joinpath(f)
        if p.is_dir() and re.match(FOLDER_NAME_PATTERN, f):
            folders_groups[f[:FOLDER_PREFIX_LEN]].append(p)

    if not folders_groups:
        raise FileNotFoundError("Could not find any folders.")

    # Get most current folders within the same prefix group
    folders = [max(folders, key=os.path.getctime) for folders in folders_groups.values()]

    # Load test and predict dataset files in format (<folder>, <test_file>, <predict_file>)
    files = []
    for folder in folders:
        print(f"Folder: {folder}")
        files.append((folder, get_prediction_file(folder, Dataset.Test), get_prediction_file(folder, Dataset.Predict)))
    print("---")

    # Make list of tuples (i, <predict_file_min_match_date>)
    ordered_files_ids = []
    for i, (_, _, predict_file) in enumerate(files):
        ordered_files_ids.append((i, pd.to_datetime(predict_file.stack(0)["match_id"]).min()))

    # Try to sort tuples by match date
    print("Ordered files (time ascending):")
    ordered_files_ids = sorted(ordered_files_ids, key=operator.itemgetter(1))
    for i, _ in ordered_files_ids:
        print(files[i][0])

    return [files[i] for i, _ in ordered_files_ids]


def iterative_backtesting(files: List[Tuple[Path, pd.DataFrame, pd.DataFrame]], ignoreodds: float) -> None:
    """
    Performs consecutive backtesting for each pair of test and predict datasets.
    Plots performance curve when finished.

    :param files: Test and predict datasets.
    :param ignoreodds: Ignore odds less than given amount when predicting what team to bet on.
    """
    # All matches in predict datasets to bet on
    predict_matches_to_bet_on = pd.DataFrame([], columns=BET_ON_MATCHES_COLS)
    bmpredict_matches_to_bet_on = pd.DataFrame([], columns=BET_ON_MATCHES_COLS)
    # All matches ids in predict sets already processed
    predict_set_processed_ids = []
    predict_set_date_min = None
    predict_set_date_max = None

    # Loop over folders
    for folder, test_file, predict_file in files:
        print("---")
        print(f"Folder: '{folder}'")
        dfs = {Dataset.Test.value: test_file, Dataset.Predict.value: predict_file}

        # Get current min and low match date
        stacked_predict_df = dfs[Dataset.Predict.value].stack(0)
        max_date = pd.to_datetime(stacked_predict_df["match_date"]).max()
        min_date = pd.to_datetime(stacked_predict_df["match_date"]).min()

        # Set new max date if found
        if predict_set_date_max is None:
            predict_set_date_max = max_date
        elif predict_set_date_max < max_date:
            predict_set_date_max = max_date

        # Set new min date if found
        if predict_set_date_min is None:
            predict_set_date_min = min_date
        elif predict_set_date_min > min_date:
            predict_set_date_min = min_date

        output = determine_matches_to_bet_on(dfs, predict_set_processed_ids, ignoreodds)
        matches_to_bet_on, bmmatches_to_bet_on, predict_set_processed_ids = output

        predict_matches_to_bet_on = predict_matches_to_bet_on.append(matches_to_bet_on[Dataset.Predict.value],
                                                                     ignore_index=True)
        bmpredict_matches_to_bet_on = bmpredict_matches_to_bet_on.append(bmmatches_to_bet_on[Dataset.Predict.value],
                                                                         ignore_index=True)

    # Whole timeframe of predictions (not necessarily matching first and last predictions dates)
    date_range = f"{predict_set_date_min.strftime(DATE_FORMAT)}-{predict_set_date_max.strftime(DATE_FORMAT)}"

    print("---")
    print("All predict matches to bet on:")
    predict_matches_to_bet_on_print = predict_matches_to_bet_on.copy()
    predict_matches_to_bet_on_print["pred_perc"] = predict_matches_to_bet_on_print["pred_perc"].astype(float).map(
        "{:.1%}".format)
    print(predict_matches_to_bet_on_print)
    print("---")
    print("Backtesting summary:")
    print("---")
    print(f"Total number of matches: {len(predict_set_processed_ids)}")
    print(f"Days time span of tested period: {(predict_set_date_max - predict_set_date_min).days}")

    plot_backtest_performance_curve(predict_matches_to_bet_on, len(predict_set_processed_ids), date_range, ignoreodds,
                                    save=True)
    plot_backtest_performance_curve(bmpredict_matches_to_bet_on, len(predict_set_processed_ids), date_range, ignoreodds,
                                    bmpreds=True, save=True)


def determine_matches_to_bet_on(dfs: Dict[str, pd.DataFrame],
                                predict_set_processed_ids: List[int],
                                ignoreodds: float) -> Tuple[Dict[str, pd.DataFrame],
                                                            Dict[str, pd.DataFrame],
                                                            List[int]]:
    """
    Determines matches to bet on for test and predict datasets according to prespecified conditions.

    Selects only matches where both models (model for home team and model for away team) "agree" on
    outcome of the match (meaning they give opposite predictions), e.g. model1 predicts 1 (win-or-draw)
    and model2 predicts 0 (loss) and vice versa.

    To focus more on reducing risk while betting, the number of matches is narrowed down by selecting only
    matches where their prediction confidence level is above a particular threshold. The threshold is
    computed on test dataset and is computed by selecting lowest confidence level on predictions
    while achieving maximal accuracy.
    This measure is done because some predictions might be on the decision threshold edge (0.5), so
    basically preditions of 0.495 vs. 0.505 are useless for us. We want to be maximally "sure" about the
    outcome.

    Also ignores predictions where potential winning odds would be less than given amount (default 1.10).

    Bookmaker predictions for matches to bet on are also computed the same way as model's predictions for comparison.
    Probability of predicted outcome is calculated as 1 / odds for simplicity.

    :param dfs: Test and predict set dataframes.
    :param predict_set_processed_ids: Already processed predict set matches ids.
    :param ignoreodds: Ignore odds less than given amount when predicting what team to bet on.
    :return: Matches to bet on for both predict and test dataset, bookmaker matches to bet on, and modified
             list of processed ids for predict dataset.
    """
    datasets = [Dataset.Test.value, Dataset.Predict.value]
    matches_to_bet_on = {d: pd.DataFrame([], columns=BET_ON_MATCHES_COLS) for d in datasets}
    bmmatches_to_bet_on = {d: pd.DataFrame([], columns=BET_ON_MATCHES_COLS) for d in datasets}
    # Default threshold value, will be modified after parsing test predictions file
    threshold = 0.51
    bmthreshold = 0.51

    for dataset in datasets:
        # Keep track of processed ids to avoid processing a single match twice (from perspective
        # of home team, and then from away team)
        processed_matches_ids = []

        for team in dfs[dataset].columns.get_level_values("team").unique():
            df_subset = dfs[dataset].loc[:, team].dropna(subset=["pred"])

            # Loop over current team's matches rows
            for i, r in df_subset.iterrows():
                # Skip match if processed from opponent's viewpoint already or if the match
                # occurred already in some other predict dataset when iterating over prediction dataset
                if r["match_id"] in processed_matches_ids:
                    continue
                elif dataset == Dataset.Predict.value and r["match_id"] in predict_set_processed_ids:
                    continue

                # Find matching row
                opponent_row = dfs[dataset][r["opponent"]].dropna(subset=["pred"])
                opponent_row = opponent_row.loc[opponent_row["match_id"] == r["match_id"]]

                if int(opponent_row["match_id"]) in processed_matches_ids:
                    continue
                elif dataset == Dataset.Predict.value and int(opponent_row["match_id"]) in predict_set_processed_ids:
                    continue

                # Append to processed ids
                processed_matches_ids.append(r["match_id"])
                if dataset == Dataset.Predict.value:
                    predict_set_processed_ids.append(r["match_id"])

                # Format values
                team1_pred = int(r["pred"])
                team1_pred_perc = float(r["pred_perc"])
                team1_odds = float(r["odds_wd"][:4])
                team2_pred = int(opponent_row["pred"].iloc[0])
                team2_pred_perc = float(opponent_row["pred_perc"].iloc[0])
                team2_odds = float(r["odds_wd"][-4:])

                # Model1 predicting win-or-draw and model2 predicting loss
                if team1_pred == 1 and team2_pred == 0 and \
                        (team1_odds > ignoreodds or np.isclose(team1_odds, ignoreodds)):
                    # Include only matches where the probabilities are higher than threshold when
                    # iterating over predict dataset, otherwise include all matches meeting the
                    # conditions for test dataset (they will be used to find best threshold).
                    if dataset == Dataset.Predict.value and APPLY_THRESHOLD_SELECTION and \
                            team1_pred_perc <= threshold:
                        continue

                    target = r["target"]
                    won = np.nan if (target is None or np.isnan(target)) else target == 1
                    match_to_bet_on = {"id": r["match_id"],
                                       "date": r["match_date"],
                                       "home": team,
                                       "away": r["opponent"],
                                       "bet_on_team": team,
                                       "pred_perc": team1_pred_perc,
                                       "odds_wd": team1_odds,
                                       "bet_won": won}
                    matches_to_bet_on[dataset] = matches_to_bet_on[dataset].append(match_to_bet_on,
                                                                                   ignore_index=True)
                # Model2 predicting win-or-draw and model1 predicting loss
                elif team1_pred == 0 and team2_pred == 1 and \
                        (team2_odds > ignoreodds or np.isclose(team2_odds, ignoreodds)):
                    if dataset == Dataset.Predict.value and APPLY_THRESHOLD_SELECTION and \
                            team2_pred_perc <= threshold:
                        continue

                    target = opponent_row["target"].iloc[0]
                    won = np.nan if (target is None or np.isnan(target)) else target == 1
                    match_to_bet_on = {"id": r["match_id"],
                                       "date": r["match_date"],
                                       "home": r["opponent"],
                                       "away": team,
                                       "bet_on_team": r["opponent"],
                                       "pred_perc": team2_pred_perc,
                                       "odds_wd": team2_odds,
                                       "bet_won": won}
                    matches_to_bet_on[dataset] = matches_to_bet_on[dataset].append(match_to_bet_on,
                                                                                   ignore_index=True)

                # Compute bookmaker predictions the same way for comparison
                if r["bmpred"] == 1 and (team1_odds > ignoreodds or np.isclose(team1_odds, ignoreodds)):
                    if dataset == Dataset.Predict.value and APPLY_THRESHOLD_SELECTION and \
                            r["bmpred_perc"] <= bmthreshold:
                        continue

                    target = r["target"]
                    won = np.nan if (target is None or np.isnan(target)) else target == 1
                    match_to_bet_on = {"id": r["match_id"],
                                       "date": r["match_date"],
                                       "home": team,
                                       "away": r["opponent"],
                                       "bet_on_team": team,
                                       "pred_perc": r["bmpred_perc"],
                                       "odds_wd": team1_odds,
                                       "bet_won": won}
                    bmmatches_to_bet_on[dataset] = bmmatches_to_bet_on[dataset].append(match_to_bet_on,
                                                                                       ignore_index=True)
                elif opponent_row["bmpred"].iloc[0] == 1 and \
                        (team2_odds > ignoreodds or np.isclose(team2_odds, ignoreodds)):
                    if dataset == Dataset.Predict.value and APPLY_THRESHOLD_SELECTION and \
                            opponent_row["bmpred_perc"].iloc[0] <= bmthreshold:
                        continue

                    target = opponent_row["target"].iloc[0]
                    won = np.nan if (target is None or np.isnan(target)) else target == 1
                    match_to_bet_on = {"id": r["match_id"],
                                       "date": r["match_date"],
                                       "home": r["opponent"],
                                       "away": team,
                                       "bet_on_team": r["opponent"],
                                       "pred_perc": opponent_row["bmpred_perc"].iloc[0],
                                       "odds_wd": team2_odds,
                                       "bet_won": won}
                    bmmatches_to_bet_on[dataset] = bmmatches_to_bet_on[dataset].append(match_to_bet_on,
                                                                                       ignore_index=True)

        matches_to_bet_on[dataset].sort_values(by=["date", "id"], inplace=True)
        bmmatches_to_bet_on[dataset].sort_values(by=["date", "id"], inplace=True)

        if dataset == Dataset.Test.value:
            # Compute best threshold/accuracy ratio for current test set which will be used to filter
            # matches to bet on
            threshold = compute_testset_best_threshold(matches_to_bet_on[dataset])["threshold"]
            bmthreshold = compute_testset_best_threshold(bmmatches_to_bet_on[dataset], verbose=False)["threshold"]
            matches_to_bet_on[dataset] = matches_to_bet_on[dataset].loc[
                matches_to_bet_on[dataset]["pred_perc"] > threshold]
            bmmatches_to_bet_on[dataset] = bmmatches_to_bet_on[dataset].loc[
                bmmatches_to_bet_on[dataset]["pred_perc"] > bmthreshold]

        print("---")
        print(f"Matches to bet on [{dataset}]:")
        print("---")
        matches_to_bet_on_print = matches_to_bet_on[dataset].copy()
        matches_to_bet_on_print["pred_perc"] = matches_to_bet_on_print["pred_perc"].astype(float).map("{:.1%}".format)
        print(matches_to_bet_on_print)
        print("---")

        # Compute stats
        compute_matches_to_bet_on_stats(matches_to_bet_on[dataset], dataset)

    return matches_to_bet_on, bmmatches_to_bet_on, predict_set_processed_ids


def compute_matches_to_bet_on_stats(matches_to_bet_on: pd.DataFrame,
                                    dataset: Dataset,
                                    verbose: bool = True) -> Dict[str, Any]:
    """
    Computes statistics for given matches to bet on.
    Including won and lost coefficients, number of wins, losses, net gain and ROI.

    Assumes placing same amount on each bet for simplicity.

    Won coeff. is computed by taking odds and subtracting 1.
    Lost coeff. is computed by summing 1s for each loss.
    Net gain is then won coeff. - lost coeff.
    ROI is net gain / total investment.

    Stats are computed for the current dataset only. Final performance may slighty differ when doing
    a backtesting because some matches may occur multiple times in the predict datasets, depending on
    the matches sequence alignment. Only first time occurence of the match is taken into account.

    :param matches_to_bet_on: Dataframe with bet on matches.
    :param dataset: Dataset type of given dataframe.
    :param verbose: Whether to print stats.
    :return: Stats about given matches.
    """
    bet_won_dropna = matches_to_bet_on["bet_won"].dropna()

    nwon = int(bet_won_dropna.sum())
    nlost = int((~(bet_won_dropna.astype(bool))).sum())
    if not bet_won_dropna.empty:
        coeff_won = matches_to_bet_on["odds_wd"].loc[matches_to_bet_on["bet_won"]].sum()-nwon
    else:
        coeff_won = 0.
    coeff_lost = nlost
    acc = nwon / len(matches_to_bet_on) if len(matches_to_bet_on) else 0
    netgain = coeff_won - coeff_lost
    roi = netgain / len(matches_to_bet_on) if len(matches_to_bet_on) else 0

    if verbose:
        print(f"Dataset [{dataset}] summary:")
        print(f"Matches to bet on: {len(matches_to_bet_on)}")
        print(f"Matches won: {nwon}")
        print(f"Matches lost: {nlost}")
        print(f"Accuracy: {acc:.1%}")
        print(f"Coeff. won: {coeff_won:.2f}")
        print(f"Coeff. lost: {coeff_lost:.2f}")
        print(f"Net gain: {netgain:.1%}")
        print(f"ROI: {roi:.1%}")
        print("---")

    return {
        "nwon": nwon,
        "nlost": nlost,
        "coeff_won": coeff_won,
        "coeff_lost": coeff_lost,
        "acc": acc,
        "netgain": netgain,
        "roi": roi,
    }


def compute_testset_best_threshold(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    Computes accuracies for each threshold value in range [.51, .99] with step of 0.01 where the
    threshold is predicted probability of selected outcome.

    :param df: Dataset to analyse.
    :param verbose: Whether to print best threshold found.
    :return: Lowest threshold with highest accuracy.
    """
    thresholds = pd.DataFrame(data=[], columns=["threshold", "accuracy", "nbets"])

    for t in np.linspace(.51, .99, 49):
        # Filter only predictions above current threshold
        subset = df.loc[df["pred_perc"] >= t]
        if not subset.empty:
            # Use only subsets which actually contain a prediction prob. within current threshold interval
            # E.g. when current threshold is 0.74 then check if there is a prediction prob. between 0.74-0.75
            if subset["pred_perc"].min() >= (t + 0.01):
                continue

            accuracy = accuracy_score([1]*len(subset), subset["bet_won"].tolist())
            thresholds = thresholds.append({"threshold": t,
                                            "accuracy": accuracy,
                                            "nbets": len(subset)},
                                           ignore_index=True)

    # Get lowest threshold with highest accuracy
    best_threshold = thresholds.loc[thresholds.index == thresholds["accuracy"].idxmax()].iloc[0]

    if verbose:
        print("---")
        print(f"Best testset threshold: {best_threshold['threshold']:.1%}, "
              f"accuracy: {best_threshold['accuracy']:.1%}")

    return {"threshold": best_threshold["threshold"], "accuracy": best_threshold["accuracy"]}


def plot_backtest_performance_curve(df: pd.DataFrame,
                                    total_number_of_matches: int,
                                    date_range: str,
                                    ignoreodds: float,
                                    bmpreds: bool = False,
                                    save: bool = True) -> None:
    """
    Plots performance curve for selected matches to bet on included in all predict datasets.

    :param df: Dataframe with all predict matches to bet on.
    :param total_number_of_matches: Total number of matches in all predict datasets.
    :param date_range: Backtesting time period.
    :param ignoreodds: Ignore odds less than given amount when predicting what team to bet on.
    :param bmpreds: Whether the predicted matches are by bookmaker.
    :param save: Whether to save plot.
    """
    df.sort_values(by=["date", "id"], inplace=True)

    xaxis = list(range(len(df)+1))
    yaxis_netgain = np.select(condlist=[df["bet_won"].astype(bool),
                                        ~df["bet_won"].astype(bool)],
                              choicelist=[df["odds_wd"]-1,
                                          -1],
                              default=np.nan)
    yaxis_roi = np.cumsum([1] * len(yaxis_netgain))
    yaxis_netgain = yaxis_netgain.cumsum()
    yaxis_roi = yaxis_netgain / yaxis_roi

    yaxis_netgain = [0] + yaxis_netgain.tolist()
    yaxis_roi = [0] + yaxis_roi.tolist()

    bet_won_dropna = df["bet_won"].dropna()
    if not bet_won_dropna.empty:
        nwon = int(bet_won_dropna.sum())
        acc = nwon / len(bet_won_dropna)
    else:
        acc = 0.

    if not bmpreds:
        print(f"Total number of matches to bet on: {len(df)}")
        print(f"Total accuracy: {acc:.1%}")
        print(f"Average odds: {df['odds_wd'].mean():.2f}")
        print(f"Total net gain: {yaxis_netgain[-1]:.1%}")
        print(f"Total ROI: {yaxis_roi[-1]:.1%}")

    # Form labels for bottom and top xaxis
    xlabels_bottom = [""] + (df["home"] + " - " + df["away"] + " | " + df["date"]).tolist()
    xlabels_top = [""] + (df["bet_on_team"].astype(str) + " | " + df["odds_wd"].astype(str)).tolist()

    # Roughly adjust xlabels size based on number of matches
    if len(df) < 50:
        labelsize = 7
    elif len(df) < 80:
        labelsize = 6
    else:
        labelsize = 5

    if bmpreds:
        color = "C4" if APPLY_THRESHOLD_SELECTION else "C9"
    else:
        color = "C0" if APPLY_THRESHOLD_SELECTION else "C1"

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))

    # First axis to plot net gain
    ax1.plot(xaxis, yaxis_netgain, marker="o", color=color, label="Net gain")
    ax1.fill_between(xaxis, yaxis_netgain, color=color, alpha=.25)
    ax1.axhline(y=0, linestyle="--", color="k", label="Baseline")
    ax1.set_xticks(xaxis)
    ax1.set_xticklabels(xlabels_bottom, rotation=45, ha="right")
    ax1.tick_params(axis="x", labelsize=labelsize)
    ax1.margins(x=0)
    ax1.set_xlabel("Matches to bet on (top: chosen team and odds)")
    ax1.set_ylabel("Net gain")
    ax1.grid(True, linestyle="dashed")
    ax1.legend(loc="upper left")
    ax1.annotate(f"Net gain: {yaxis_netgain[-1]:.1%}", xy=(1, yaxis_netgain[-1]), xytext=(50, 0),
                 xycoords=("axes fraction", "data"), textcoords="offset points",
                 fontsize=11, arrowprops=dict(arrowstyle="->"))

    # Second axis to plot ROI
    ax2 = ax1.twinx()
    ax2.set_xbound(ax1.get_xbound())
    ax2.plot(xaxis, yaxis_roi, marker=".", color=color, linestyle=":", label="ROI")
    ax2.margins(x=0)
    ax2.tick_params(axis="x", labelsize=labelsize)
    ax2.set_xticks(xaxis)
    ax2.set_ylabel("ROI")
    ax2.legend(loc="lower left")
    ax2.annotate(f"ROI: {yaxis_roi[-1]:.1%}", xy=(1, yaxis_roi[-1]), xytext=(50, 0),
                 xycoords=("axes fraction", "data"), textcoords="offset points",
                 fontsize=11, arrowprops=dict(arrowstyle="->"))

    # Dummy third axis just to show additional xlabels on top
    ax3 = ax2.twiny()
    ax3.set_xbound(ax1.get_xbound())
    ax3.set_xticks(xaxis)
    ax3.set_xticklabels(xlabels_top, rotation=45, ha="left")
    ax3.tick_params(axis="x", labelsize=labelsize)

    title_prefix = "Bookmaker" if bmpreds else "Model"
    selection = "" if APPLY_THRESHOLD_SELECTION else "without threshold selection"
    selection_file = "" if APPLY_THRESHOLD_SELECTION else "withoutts_"
    plt.title(f"[{title_prefix}] Backtesting {selection} (assuming placing the same amount on each bet)\n"
              f"time period: {date_range}, ignoreodds: {ignoreodds}\n"
              f"Net gain: {yaxis_netgain[-1]:.1%}, ROI: {yaxis_roi[-1]:.1%}, "
              f"Total matches: {total_number_of_matches}, Total matches to bet on: {len(df)}, "
              f"Accuracy: {acc:.1%}, Average odds: {df['odds_wd'].mean():.2f}")
    plt.tight_layout()

    if save:
        path = f"{DATA_DIR}{IMG_DIR}"
        plt.savefig(f"{path}backtest_{title_prefix.lower()}_{selection_file}"
                    f"{date_range}_{str(ignoreodds).replace('.', '')}")
    plt.show()


def run(path: str, ignoreodds: float) -> None:
    """
    Runs iterative backtesting of the saved models.
    Expects non overlapping folders in given parent directory.
    Behavior of backtesting is not guaranteed to be correct when multiple folders overlap.

    E.g.: parent directory contains 10 different folders each saved from training with different
    ndiscard argument.

    :param path: Path to directory where to search for models.
    :param ignoreodds: Ignore odds less than given amount when predicting what team to bet on.
    """
    try:
        files = load_predictions_files(path)
    except (ValueError, FileNotFoundError) as e:
        print(e)
        sys.exit(1)

    iterative_backtesting(files, ignoreodds)
