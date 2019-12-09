from collections import defaultdict
import itertools as it
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Tuple

import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from soccerpredictor.backtester.backtester import determine_matches_to_bet_on, compute_matches_to_bet_on_stats, \
    compute_testset_best_threshold
from soccerpredictor.util.constants import DATA_DIR, ASSETS_DIR
from soccerpredictor.util.common import get_latest_models_dir, get_model_settings_file, get_prediction_file
from soccerpredictor.util.enums import Dataset


# Data as dataframes and their configs
dfs = {}
dfs_config = {}
# Model settings file
model_settings = html.Div()
# Model dir to load data from
model_dir = Path()
# Subplots to show
subplots = defaultdict(lambda: defaultdict())
# Layout of pages
layouts = html.Div()
# Predictions table
matches_to_bet_on_table = {}


def load_predictions_dataset(name: str) -> None:
    """
    Loads test and predict dataset predictions files.

    :param name: Name of the folder to search for.
    """
    global dfs
    global dfs_config
    global model_dir

    model_dir = get_latest_models_dir(name)
    for dataset in [Dataset.Test.value, Dataset.Predict.value]:
        pred_file = get_prediction_file(model_dir, Dataset(dataset))
        dfs[dataset] = pred_file
        dfs_config[dataset] = compute_vars(pred_file)


def make_predictions_subplots() -> None:
    """
    Creates subplots for predictions and fills them with data.

    """
    global subplots

    for dataset in dfs.keys():
        subplots[dataset] = make_subplots(rows=dfs_config[dataset]["nrows"],
                                          cols=dfs_config[dataset]["ncols"],
                                          shared_yaxes=True,
                                          x_title="Matches",
                                          y_title="Outcomes",
                                          subplot_titles=dfs_config[dataset]["teams_list"])

    # Fill prediction subplots with data
    for dataset in dfs.keys():
        # Loop over teams and assign them to appropriate row and col position
        for r in range(dfs_config[dataset]["nrows"]):
            for c in range(dfs_config[dataset]["ncols"]):
                t = next(dfs_config[dataset]["teams"], None)
                if t:
                    # Drop rows with nan predictions if any
                    df = dfs[dataset].loc[:, t].dropna(subset=["pred"])

                    # Make subplots for targets and predictions
                    # Row and col is indexed from 1 in plotly (thus we need to add 1)
                    subplots[dataset].add_trace(go.Scatter(x=df.index,
                                                           y=df.loc[:, "target"],
                                                           line=dict(color="#1f77b4")),
                                                row=r+1, col=c+1)
                    subplots[dataset].add_trace(go.Scatter(x=df.index,
                                                           y=df.loc[:, "pred"],
                                                           line=dict(color="#ff7f0e")),
                                                row=r+1, col=c+1)

        # Modify properties of each subplot
        team = dfs_config[dataset]["teams_list"][0]
        teams_iter = iter(dfs_config[dataset]["teams_list"])
        yaxis_range = list(range(len(dfs[dataset].loc[0, (team, "preds_all")])))

        subplots[dataset]["layout"]["showlegend"] = False
        subplots[dataset]["layout"]["title"] = "Model's predictions quick preview:"
        subplots[dataset]["layout"]["yaxis"]["tickvals"] = yaxis_range

        for i in subplots[dataset].layout:
            # Y axis tickvals common for all teams
            if i.startswith("yaxis"):
                subplots[dataset].layout[i]["tickvals"] = yaxis_range
            # X axis tickvals customized for each team
            if i.startswith("xaxis"):
                t = next(teams_iter, None)
                if t:
                    subplots[dataset].layout[i]["tickvals"] = list(range(0, len(dfs[dataset][t].dropna())))


def load_model_settings() -> None:
    """
    Loads model settings and shows some basic settings info.

    """
    global model_settings

    settings = get_model_settings_file(model_dir)
    last_run = settings["runs"][str(len(settings["runs"].keys())-1)]

    avg_runtime = last_run["avg_runtime_per_epoch_in_secs"]
    total_runtime = last_run["total_runtime_in_secs"]
    total_epochs = last_run["current_run_epochs"] + last_run["previous_epochs"]

    style = {"font-weight": "bold"}
    model_settings = html.Div([
        html.Span("Features: ", style=style),
        html.Span(f"{', '.join([f for f in settings['features']])}"), html.Br(),
        html.Span("Seasons: ", style=style),
        html.Span(f"{', '.join([str(f) for f in settings['seasons']])}"), html.Br(),
        html.Span("Config: ", style=style),
        html.Span(f"{', '.join([f'{k}: {v}' for k, v in settings['config'].items()])}"), html.Br(),
        html.Span("Epochs trained for: ", style=style),
        html.Span(f"{total_epochs}"), html.Br(),
        html.Span("Avg runtime per epoch: ", style=style),
        html.Span(f"~{avg_runtime:.1f} secs"), html.Br(),
        html.Span("Total runtime: ", style=style),
        html.Span(f"~{(total_runtime / 60):.1f} mins"), html.Br(),
    ])


def compute_vars(df: pd.DataFrame) -> Dict:
    """
    Computes number of rows and cols needed to make subplot for every team.

    :param df: Current dataframe.
    :return: Dict of computed values.
    """
    nteams = len(df.columns.get_level_values("team").unique())
    nrows = int(np.round(np.sqrt(nteams)))
    ncols = int(np.round(nteams / nrows))
    teams, teams_list = it.tee(sorted(df.columns.get_level_values("team").unique()))

    return {"nteams": nteams,
            "nrows": nrows,
            "ncols": ncols,
            "teams": teams,
            "teams_list": list(teams_list)}


def load_layouts() -> None:
    """
    Loads layouts for pages.

    """
    global layouts

    loaded_from = html.H6(f"Loaded from directory: \"{model_dir.name}\"")
    datasets = list(dfs.keys())

    layouts = html.Div([
        html.H2("Predictions"),
        loaded_from,
        html.Div([
            html.Div([
                html.H5("Choose dataset:"),
                dcc.Dropdown(id="pg-dd-dataset",
                             options=[{"label": i, "value": i} for i in datasets],
                             value=datasets[0],
                             clearable=False),
            ], style={"width": "48%", "display": "inline-block"}),
        ]),
        html.Div([
            html.H5("Choose team:"),
            dcc.Dropdown(id="pg-dd-team",
                         options=[],
                         clearable=False)
        ], style={"width": "48%"}),
        dcc.Graph(id="pg-gr-detail"),
        dcc.Graph(id="pg-gr-subplots"),
        html.Div(id="pg-matches-table"),
        html.Br(),
        html.Div(id="pg-dt-stats"),
        html.Br(),
        html.H6(f"Model settings:"),
        html.Div(model_settings),
        html.Br()
    ])


# Main layouts
app = dash.Dash(__name__, assets_folder=str(Path(f"{os.getcwd()}").joinpath(f"{DATA_DIR}{ASSETS_DIR}")))
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div([html.H1("Model's output dashboard", style={"text-align": "center"})]),
    html.Div(id="page-content")
])


@app.callback(Output("page-content", "children"),
              [Input("url", "pathname")])
def redirect_page(_: str) -> html.Div:
    """
    Displays corresponding page layout.

    :return: Layout of page.
    """
    return layouts


@app.callback(Output("pg-gr-detail", "figure"),
              [Input("pg-dd-dataset", "value"),
               Input("pg-dd-team", "value")])
def plot_graph_detail(dataset: str, team: str) -> Dict:
    """
    Plots graph detail of currently selected team.
    Depends on selected team and dataset.

    :param dataset: Selected dataset.
    :param team: Selected team.
    :return: Scatter plots.
    """
    # Return empty dict if values not loaded yet
    if not all([dataset, team]):
        return {}

    # Drop nan values for current team
    df_pred = dfs[dataset].loc[:, team].dropna(subset=["pred"]).copy()

    # Onhover detail for predictions
    onhover_detail_preds = df_pred.loc[:, "pred_perc"].astype(float).map("{:.1%}".format).astype(str) + "<br /><br />"

    # Onhover detail for targets
    features = ["match_goals", "rating", "red_cards", "errors", "season", "league", "odds_wd"]
    onhover_detail_targets = "vs. " + df_pred.loc[:, "opponent"] + "<br />"
    for f in features:
        onhover_detail_targets += f"{f}: " + df_pred.loc[:, f].astype(str) + "<br />"

    return {
        "data": [go.Scatter(x=df_pred.index,
                            y=df_pred.loc[:, "target"],
                            name="Targets",
                            mode="lines+markers",
                            text=onhover_detail_targets.tolist()),
                 go.Scatter(x=df_pred.index,
                            y=df_pred.loc[:, "pred"],
                            name=f"Predictions",
                            mode="lines+markers",
                            text=onhover_detail_preds.tolist())],
        "layout": go.Layout(title=f"{team} [Predictions]",
                            xaxis=dict(title="Matches",
                                       tickvals=df_pred.index,
                                       ticktext=df_pred.loc[:, "match_date"]),
                            yaxis=dict(title="Outcomes",
                                       tickvals=list(range(len(df_pred.loc[0, "preds_all"])))),
                            showlegend=True)
    }


@app.callback([Output("pg-dd-team", "options"),
               Output("pg-dd-team", "value"),
               Output("pg-gr-subplots", "figure")],
              [Input("pg-dd-dataset", "value")])
def set_dd_team(dataset: str) -> Tuple:
    """
    Sets options for team dropdown depending on the selected dataset, also sets initial
    value for the dropdown, and plots subplots.

    :param dataset: Selected dataset.
    :return: Tuple of dropdown values, initial value of dropdown, and subplots.
    """
    if not dataset:
        return [], None, {}

    o1 = [{"label": i, "value": i} for i in dfs_config[dataset]["teams_list"]]
    o2 = dfs_config[dataset]["teams_list"][0]
    o3 = subplots[dataset]

    return o1, o2, o3


@app.callback([Output("pg-dt-stats", "children"),
               Output("pg-matches-table", "children")],
              [Input("pg-dd-dataset", "value"),
               Input("pg-dd-team", "value")])
def set_dt_stats(dataset: str, team: str) -> Tuple[html.Div, html.Div]:
    """
    Creates a Datatable with predictions.
    Prediction is filtered by chosen dataset and team.

    :param dataset: Selected dataset.
    :param team: Selected team.
    :return: Datatable with predictions contained in a Div.d
    """
    if not all([dataset, team]):
        return html.Div(), html.Div()

    # Drop any potential nan prediction rows and reverse dataframe so recent matches are at the top
    df = dfs[dataset].loc[:, team].dropna(subset=["pred"]).copy()[::-1]
    df["pred_perc"] = df["pred_perc"].astype(float).map("{:.1%}".format)
    df["bmpred_perc"] = df["bmpred_perc"].astype(float).map("{:.1%}".format)

    predictions_table = html.Div([
        html.H6(f"Table view of all {team}'s predictions for [{dataset}] dataset:"),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict("records"),
            style_table={"overflowX": "scroll"},
            style_data_conditional=[{"if": {"row_index": "odd"},
                                     "backgroundColor": "rgb(248, 248, 248)"}],
            style_header={"backgroundColor": "rgb(230, 230, 230)",
                          "fontWeight": "bold"})
    ])

    return predictions_table, matches_to_bet_on_table[dataset]


def show_matches_to_bet_on(ignoreodds: float) -> None:
    """
    Shows matches to bet on in a table.

    :param ignoreodds: Ignore odds less than given amount when predicting what team to bet on.
    """
    global matches_to_bet_on_table

    matches_to_bet_on, _, _ = determine_matches_to_bet_on(dfs, [], ignoreodds)
    testset_bet_accuracy = compute_testset_best_threshold(matches_to_bet_on[Dataset.Test.value], verbose=False)
    testset_bet_accuracy_print = f"Best test set threshold: {testset_bet_accuracy['threshold']:.1%}, " \
                                 f"accuracy: {testset_bet_accuracy['accuracy']:.1%}"

    for dataset in [Dataset.Test.value, Dataset.Predict.value]:
        stats = compute_matches_to_bet_on_stats(matches_to_bet_on[dataset],
                                                Dataset(dataset),
                                                verbose=False)

        # Format columns
        matches_to_bet_on[dataset]["pred_perc"] = matches_to_bet_on[dataset]["pred_perc"].astype(float).map(
            "{:.1%}".format)
        matches_to_bet_on[dataset]["bet_won"] = matches_to_bet_on[dataset]["bet_won"].astype(str)

        style = {"font-weight": "bold"}
        matches_to_bet_on_table[dataset] = html.Div([
            html.H5(f"Bet-on predictions for [{dataset}] dataset:"),
            html.H6(f"{testset_bet_accuracy_print}"),
            html.Div([
                html.Span(f"Matches bet on: ", style=style),
                html.Span(f"{len(matches_to_bet_on[dataset])}"), html.Br(),
                html.Span(f"Matches won: ", style=style),
                html.Span(f"{stats['nwon']}"), html.Br(),
                html.Span(f"Matches lost: ", style=style),
                html.Span(f"{stats['nlost']}"), html.Br(),
                html.Span(f"Accuracy: ", style=style),
                html.Span(f"{stats['acc']:.1%}"), html.Br(),
                html.Span(f"Coeff. won: ", style=style),
                html.Span(f"{stats['coeff_won']:.2f}"), html.Br(),
                html.Span(f"Coeff. lost: ", style=style),
                html.Span(f"{stats['coeff_lost']:.2f}"), html.Br(),
                html.Span(f"Net gain: ", style=style),
                html.Span(f"{stats['netgain']:.1%}"), html.Br(),
                html.Span(f"ROI: ", style=style),
                html.Span(f"{stats['roi']:.1%}"), html.Br(),
            ]),
            html.Br(),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in matches_to_bet_on[dataset].columns.tolist()],
                data=matches_to_bet_on[dataset].to_dict("records"),
                style_table={"overflowX": "scroll"},
                style_data_conditional=[{"if": {"row_index": "odd"},
                                         "backgroundColor": "rgb(248, 248, 248)"}],
                style_header={"backgroundColor": "rgb(230, 230, 230)",
                              "fontWeight": "bold"},
            ),
            html.Br(),
            html.Br(),
        ])


def run(name: str, host: str, port: int, ignoreodds: float) -> None:
    """
    Loads data, layouts, and runs Dash server.

    Using global variables is not ideal when using Dash. However, they are set in such a way
    that they are not modified during the run so it should be okay. All dataframes and tables
    are stored as global variables before running the server and can be dynamically switched
    between depending on the dropdowns selection made by user.

    :param name: Exact name, or a part of name of the models dir to load.
    :param host: Host to run Dash on.
    :param port: Port to run Dash on.
    :param ignoreodds: Ignore odds less than given amount when predicting what team to bet on.
    """
    if host == "0":
        host = "0.0.0.0"

    try:
        load_predictions_dataset(name)
        load_model_settings()
    except (ValueError, FileNotFoundError) as e:
        print(e)
        sys.exit(1)

    make_predictions_subplots()
    show_matches_to_bet_on(ignoreodds)
    load_layouts()

    app.run_server(debug=False, host=host, port=port)
