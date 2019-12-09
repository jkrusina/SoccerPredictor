from collections import defaultdict
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Tuple, Optional

from soccerpredictor.trainer.dataloader import SPDataLoader
from soccerpredictor.trainer.dbmanager import SPDBManager
from soccerpredictor.trainer.model import SPModel
from soccerpredictor.util import common as spc
from soccerpredictor.util.config import SPConfig
from soccerpredictor.util.constants import *
from soccerpredictor.util.enums import Dataset, SaveLoad, TargetVariable


class SPTrainer:
    """
    Takes care of training, testing and prediction of each particular model.

    Attributes:
        models: Models used.
        train_stats: Train statistics.
        test_stats: Test statistics.
        best_train_stats: Best train statistics.
        best_test_stats: Best test statistics.
        predictions: Predictions of the model.

    """

    def __init__(self,
                 dbmanager: SPDBManager,
                 generated_folder_prefix: str = "",
                 model_settings: Optional[Dict] = None,
                 folder: Optional[Path] = None) -> None:
        """

        :param dbmanager: DB manager.
        :param generated_folder_prefix: Generated prefix for output folder's name.
        :param model_settings: Previously loaded model settings.
        :param folder: Previously loaded folder.
        """
        self._dbmanager = dbmanager
        self._config = SPConfig()

        # Variables used from config
        self._epochs = self._config.epochs
        self._predict = self._config.predict
        self._printfreq = self._config.printfreq
        self._resume = self._config.resume
        self._savefreq = self._config.savefreq
        self._seasons = list(range(MIN_SEASON, MAX_SEASON + 1))
        self._target_variable = TargetVariable.FutureWD
        self._timesteps = self._config.timesteps
        self._verbose = self._config.verbose

        self._folder_prefix = generated_folder_prefix
        self._timestamp = f"{datetime.now():{TIMESTAMP_FORMAT}}"
        self._previous_timestamp = ""
        self._previous_epochs = 0
        self._total_epochs_passed = 0
        self._model_settings = {} if not model_settings else model_settings
        self._runtimes_per_epoch = []
        self._teams_tuples = self._dbmanager.query_teams_ids_names_tuples()
        self._total_epochs = self._previous_epochs + self._epochs
        self._features = FEATURES_COMMON + FEATURES_WD

        self.models: Dict[str, SPModel] = {}
        self.train_stats = pd.DataFrame()
        self.test_stats = pd.DataFrame()
        self.best_train_stats = pd.DataFrame()
        self.best_test_stats = pd.DataFrame()
        self.predictions = {d: pd.DataFrame() for d in [Dataset.Test, Dataset.Predict]}
        self.data_loader = SPDataLoader(self._dbmanager, self._seasons, self._model_settings)

        # Load previous settings if we continue in training
        if self._resume:
            self._load_previous_settings(folder)

    def run(self) -> None:
        """
        Runs training of the model for given number of epochs.

        """
        st = time.time()
        df_train, df_test, df_predict = self._preload()

        # Just load models and make predictions
        if self._predict:
            teams = spc.get_unique_teams(df_train)
            self._load_models(teams, include_optimizer=False)

            self.predict(df_test,
                         Dataset.Test,
                         revert_to_best_params=True,
                         restore_states_after_training=True)
            self.predict(df_predict, Dataset.Predict)
            self.save(predict=True)
        else:
            self.train(df_train, df_test)
            self.save(models=True, train=True, test=True)

            # Predict both test and predict datasets with the best params after training
            self.predict(df_test,
                         Dataset.Test,
                         revert_to_best_params=True,
                         restore_states_after_training=True)
            self.predict(df_predict, Dataset.Predict)
            self.save(predict=True)

        print(f"Run time: {((time.time()-st)/60):.2f} mins.")

    def _preload(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads train, test, and predict teams data for each model, and build models.

        :return: Train, test, and predict datasets.
        """
        print("Loading data...")

        # Load fixtures for all three datasets
        df_train, df_test, df_predict = self.data_loader.load_and_process_fixtures_data()

        # Fit scalers on train dataset only
        self.data_loader.fit_scalers(df_train)

        # Build models for all teams
        all_teams = spc.get_unique_teams(pd.concat([df_train, df_test, df_predict]))
        for t in all_teams:
            self.models[t] = SPModel(t,
                                     self.data_loader.test_teams,
                                     self.data_loader.teams_names_bitlen,
                                     f"{self._folder_prefix}")

        # Get fixtures ids where each team played in (separately for each dataset) and store them
        # Ids for test and predict datasets are properly aligned to fit match sequences
        for t in self.data_loader.train_teams:
            fixtures_ids = spc.get_fixtures_ids_from_df(df_train, t)
            team_matches_data = self.data_loader.load_and_process_team_data(Dataset.Train,
                                                                            self._teams_tuples[t],
                                                                            fixtures_ids)
            self.models[t].prepare_matches_data(Dataset.Train, team_matches_data)

            # Compute class weights for train dataset (remove last id from each team's fixtures
            # which will not be used for training to properly offset test dataset)
            self.models[t].compute_class_weights(team_matches_data, fixtures_ids[:-1], verbose=False)

        for t in self.data_loader.test_teams:
            fixtures_ids = spc.get_fixtures_ids_from_df(df_test, t)
            aligned_fixtures_ids = spc.align_fixtures_ids(df_train, t, fixtures_ids, self._timesteps)
            team_matches_data = self.data_loader.load_and_process_team_data(Dataset.Test,
                                                                            self._teams_tuples[t],
                                                                            aligned_fixtures_ids)
            self.models[t].prepare_matches_data(Dataset.Test, team_matches_data)

        for t in self.data_loader.predict_teams:
            combined_df_train = pd.concat((df_train, df_test), ignore_index=True)
            fixtures_ids = spc.get_fixtures_ids_from_df(df_predict, t)
            # Use combined train+test dataset in case that there would be less test samples than timesteps
            # so the rest of sequence can be filled from train dataset
            aligned_fixtures_ids = spc.align_fixtures_ids(combined_df_train, t, fixtures_ids, self._timesteps)
            team_matches_data = self.data_loader.load_and_process_team_data(Dataset.Predict,
                                                                            self._teams_tuples[t],
                                                                            aligned_fixtures_ids)
            self.models[t].prepare_matches_data(Dataset.Predict, team_matches_data)

        # Assemble network for each model
        print(f"Assembling {len(self.models)} models...")
        t0 = all_teams[0]
        self.models[t0].build_model()

        for t in all_teams[1:]:
            if RANDOM_WEIGHTS and not self._resume:
                self.models[t].build_model()
            else:
                self.models[t].build_model_from(self.models[t0])

        if not self._resume:
            self._set_default_snapshots()
            self._create_stats_files()

        # Reset indices of dfs
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)
        df_predict.reset_index(inplace=True, drop=True)

        return df_train, df_test, df_predict

    def _set_default_snapshots(self) -> None:
        """
        Sets params of teams of each model's snapshot during first run.

        """
        params_all_teams = {t: self.models[t].network.get_main_head_params(include_optimizer=False)
                            for t in self.data_loader.teams}

        for t in self.data_loader.teams:
            self.models[t].snapshot.set_initial_params(params_all_teams)

    def _create_stats_files(self) -> None:
        """
        Creates stats files during first run.

        """
        levels_names = ["team", "metric"]

        # Create stats for training
        multiindex_train = pd.MultiIndex.from_product([self.data_loader.train_teams,
                                                       TRAINING_METRICS],
                                                      names=levels_names)
        self.train_stats = pd.DataFrame([], index=[], columns=multiindex_train)

        multiindex_train_best = pd.MultiIndex.from_product([self.data_loader.train_teams_exclusively,
                                                            TRAINING_METRICS],
                                                           names=levels_names)
        self.best_train_stats = pd.DataFrame([], index=[], columns=multiindex_train_best)

        # Create stats for testing
        multiindex_test = pd.MultiIndex.from_product([self.data_loader.test_teams,
                                                      TRAINING_METRICS],
                                                     names=levels_names)
        self.test_stats = pd.DataFrame([], index=[], columns=multiindex_test)
        self.best_test_stats = pd.DataFrame([], index=[], columns=multiindex_test)

    def train(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
        """
        Loops over train dataset for given number of epochs. Model's performance is evaluated
        against test dataset after each epoch.

        During each loop over matches within an epoch:
            1) Params of head2 for both current models are set - this needs to be done at the start to
               prevent using already updated weights when training second model.
            2) If input is correctly fetched then model of team1 is trained on the input and updated
               states of head2 are stored into snapshot. The same applies for the second model.
            3) Index to data is incremented for both models.
            4) Advance to next match and repeat.

        :param df_train: Portion of data used for training.
        :param df_test: Portion of data used for testing.
        """
        teams = spc.get_unique_teams(df_train)
        dataset = Dataset.Train
        # Load previously saved models if we continue in training
        # Optimizer is necessary for training
        if self._resume:
            self._load_models(teams, include_optimizer=True)

        for epoch in range(self._previous_epochs, self._total_epochs):
            st = time.time()
            print("---")
            print(f"Epoch: {epoch+1} of {self._total_epochs}")
            print("Training model...")
            # Verbose print only for first epoch
            verbose = (epoch == self._previous_epochs)
            train_metrics = defaultdict(lambda: defaultdict(list))
            # Reset states of RNNs at the beginning of the epoch, so they can be saved at the end
            # Also reset data index position
            for t in teams:
                self.models[t].snapshot.reset_states()
                self.models[t].network.reset_states()
                self.models[t].matches_data[dataset]["idx"] = 0

            # Loop over matches
            for i, r in df_train.iterrows():
                if verbose and self._verbose > 0:
                    print(f"{i:04d}: {r['id']:04d} {r['date']} {r['season']:02d} {r['league']} {r['home']} {r['away']}")
                team1 = r["home"]
                team2 = r["away"]

                # Set team2 weights for both teams to avoid using newly changed weights of home team for
                # the away team and vice versa
                self.models[team1].set_network_head2_params(team2)
                self.models[team2].set_network_head2_params(team1)

                # Train home model
                x_input, y_input = self.models[team1].form_input(dataset, self.models[team2])
                if x_input and y_input:
                    loss, acc = self.models[team1].train_on_batch(x_input, y_input)
                    self.models[team1].store_network_head2_states(team2)

                    train_metrics[team1]["loss"].append(loss)
                    train_metrics[team1]["acc"].append(acc)

                # Train away model
                x_input, y_input = self.models[team2].form_input(dataset, self.models[team1])
                if x_input and y_input:
                    loss, acc = self.models[team2].train_on_batch(x_input, y_input)
                    self.models[team2].store_network_head2_states(team1)

                    train_metrics[team2]["loss"].append(loss)
                    train_metrics[team2]["acc"].append(acc)

                # Increment index to data
                self.models[team1].matches_data[dataset]["idx"] += 1
                self.models[team2].matches_data[dataset]["idx"] += 1

            # Track epochs passed
            self._total_epochs_passed += 1

            # Append metrics per current epoch
            for t in teams:
                self.models[t].snapshot.save_states_after_training()
                self.models[t].save_states_after_training()

                self.train_stats.loc[epoch, (t, "loss")] = np.mean(train_metrics[t]["loss"])
                self.train_stats.loc[epoch, (t, "acc")] = np.mean(train_metrics[t]["acc"])

            # Test models after every epoch
            self.test(df_test, epoch, verbose)

            # Call on epoch end processing
            self._on_epoch_end(epoch)

            # Measure training time and approx remaining time to finish
            et = time.time() - st
            self._runtimes_per_epoch.append(et)
            estimate = et * (self._total_epochs - self._total_epochs_passed)
            runtime = f"{estimate/60:.2f} mins" if epoch else "<inaccurate at first epoch>"
            print(f"Epoch took: {et:.2f} secs. Estimated time to finish: {runtime}")

    def test(self, df_test: pd.DataFrame, epoch: int, verbose: bool) -> None:
        """
        Performs a single iteration of test_on_batch on every sample in given dataset.
        Logic of setting weights is same as for training.

        :param df_test: Portion of data used for testing.
        :param epoch: Current epoch after which the testing is performed.
        :param verbose: Whether to print match info when looping.
        """
        print("Testing model...")

        dataset = Dataset.Test
        test_metrics = defaultdict(lambda: defaultdict(list))
        for t in self.data_loader.test_teams:
            self.models[t].matches_data[dataset]["idx"] = 0

        # Loop over matches
        for i, r in df_test.iterrows():
            if verbose and self._verbose > 0:
                print(f"{i:04d}: {r['id']:04d} {r['date']} {r['season']:02d} {r['league']} {r['home']} {r['away']}")
            team1 = r["home"]
            team2 = r["away"]

            self.models[team1].set_network_head2_params(team2)
            self.models[team2].set_network_head2_params(team1)

            # Test home model
            # x_input should not be none for test dataset
            x_input, y_input = self.models[team1].form_input(dataset, self.models[team2])
            if x_input:
                loss, acc = self.models[team1].network.test_on_batch(x_input, y_input)
                self.models[team1].store_network_head2_states(team2)

                test_metrics[team1]["loss"].append(loss)
                test_metrics[team1]["acc"].append(acc)

            # Test away model
            x_input, y_input = self.models[team2].form_input(dataset, self.models[team1])
            if x_input:
                loss, acc = self.models[team2].network.test_on_batch(x_input, y_input)
                self.models[team2].store_network_head2_states(team1)

                test_metrics[team2]["loss"].append(loss)
                test_metrics[team2]["acc"].append(acc)

            self.models[team1].matches_data[dataset]["idx"] += 1
            self.models[team2].matches_data[dataset]["idx"] += 1

        # Append metrics for current epoch
        for t in self.data_loader.test_teams:
            self.test_stats.loc[epoch, (t, "loss")] = np.mean(test_metrics[t]["loss"])
            self.test_stats.loc[epoch, (t, "acc")] = np.mean(test_metrics[t]["acc"])

    def _on_epoch_end(self, epoch: int) -> None:
        """
        Checks whether models has improved, logs metrics for current epoch, and updates weights
        in snapshots for improved models.
        Called on each epoch end.

        :param epoch: Current epoch.
        """
        print("Epoch end summary:")

        improved_train_teams = []
        # For train-only teams we monitor improvement over the train dataset. Since the teams are not
        # evaluated on test dataset, we cannot measure whether models are capable to generalize or not.
        # But we do not care whether the model actually learns to generalize or just remembers input
        # output mapping. We care mainly about good predictions which we assume that model is able
        # to learn eventually.
        # Only best params are kept in case that the model's performance would actually degrade - which
        # can happen - model may overfit badly or its weights might start increasing rapidly, etc.
        for t in self.data_loader.train_teams_exclusively:
            train_metrics = {"loss": self.train_stats.loc[epoch, (t, "loss")],
                             "acc": self.train_stats.loc[epoch, (t, "acc")]}
            self.models[t].tensorboard.notify_train(epoch, train_metrics)
            self.best_train_stats, improved = self.models[t].update_performance(self.train_stats,
                                                                                self.best_train_stats,
                                                                                epoch,
                                                                                train_metrics)
            if improved:
                self.models[t].snapshot.record_best_params()
                improved_train_teams.append(t)

        improved_test_teams = []
        for t in self.data_loader.test_teams:
            self.models[t].tensorboard.notify_train(epoch,
                                                    {"loss": self.train_stats.loc[epoch, (t, "loss")],
                                                     "acc": self.train_stats.loc[epoch, (t, "acc")]})
            test_metrics = {"loss": self.test_stats.loc[epoch, (t, "loss")],
                            "acc": self.test_stats.loc[epoch, (t, "acc")]}
            self.models[t].tensorboard.notify_test(epoch, test_metrics)
            self.best_test_stats, improved = self.models[t].update_performance(self.test_stats,
                                                                               self.best_test_stats,
                                                                               epoch,
                                                                               test_metrics)
            # We need to propagate updated weights of team to snapshots in every model
            if improved:
                self.models[t].snapshot.record_best_params()
                improved_test_teams.append(t)

        # Update weights at once
        for t in improved_train_teams:
            improved_weights = self.models[t].network.get_main_head_weights()
            for t_ in self.data_loader.teams:
                self.models[t_].snapshot.update_weights(t, improved_weights)
        for t in improved_test_teams:
            improved_weights = self.models[t].network.get_main_head_weights()
            for t_ in self.data_loader.teams:
                self.models[t_].snapshot.update_weights(t, improved_weights)

        final_epoch = self._total_epochs_passed == self._total_epochs

        # Save every nth epoch (except last one to prevent saving twice)
        if self._savefreq and not final_epoch and (self._total_epochs_passed % self._savefreq) == 0:
            self.save(models=True, train=True, test=True)

        # Print actual loss/acc every nth epoch
        if final_epoch or (self._printfreq and (self._total_epochs_passed % self._printfreq) == 0):
            print("Test stats:")
            print(spc.compressed_df_format(self.test_stats.loc[[self.test_stats.index[-1]]]))
            print("Best test stats:")
            print(spc.compressed_df_format(self.best_test_stats.ffill().loc[[self.best_test_stats.index[-1]]]))

    def predict(self,
                df: pd.DataFrame,
                predict_dataset: Dataset,
                revert_to_best_params: bool = False,
                restore_states_after_training: bool = False,
                verbose: bool = False) -> None:
        """
        Performs a single iteration of predict_on_batch for every sample in given dataset.
        Logic of setting weights is same as for training.

        :param df: Portion of data used for prediction.
        :param predict_dataset: Which type of dataset is used for prediction.
        :param revert_to_best_params: Whether to revert back to best weights.
        :param restore_states_after_training: Whether to restore states to moment after training.
        :param verbose: Whether to print matches predicting.
        """
        print(f"Predicting dataset: {predict_dataset.value}...")

        teams = spc.get_unique_teams(df)
        predict_metrics = defaultdict(lambda: defaultdict(list))

        for t in teams:
            self.models[t].matches_data[predict_dataset]["idx"] = 0
            # Use only best params for prediction
            if revert_to_best_params:
                self.models[t].snapshot.revert_to_best_params()
                self.models[t].revert_to_best_params(include_optimizer=False)
            if restore_states_after_training:
                self.models[t].snapshot.restore_states_after_training()
                self.models[t].restore_states_after_training()

        # Loop over matches
        for i, r in df.iterrows():
            if verbose and self._verbose > 0:
                print(f"{i:04d}: {r['id']:04d} {r['date']} {r['season']:02d} {r['league']} {r['home']} {r['away']}")
            team1 = r["home"]
            team2 = r["away"]
            team1_preds = None
            team2_preds = None

            self.models[team1].set_network_head2_params(team2)
            self.models[team2].set_network_head2_params(team1)

            team1_xinput, team1_yinput = self.models[team1].form_input(predict_dataset, self.models[team2])
            if (predict_dataset == Dataset.Predict and team1_xinput) or (team1_xinput and team1_yinput):
                team1_preds = self.models[team1].network.predict_on_batch(team1_xinput)
                self.models[team1].store_network_head2_states(team2)

            team2_xinput, team2_yinput = self.models[team2].form_input(predict_dataset, self.models[team1])
            if (predict_dataset == Dataset.Predict and team2_xinput) or (team2_xinput and team2_yinput):
                team2_preds = self.models[team2].network.predict_on_batch(team2_xinput)
                self.models[team2].store_network_head2_states(team1)

            emsg = "There are probably some missing data in the dataset."
            if team1_preds is None:
                raise ValueError(f"Predictions for model1 are nan. \n{emsg}")
            elif team2_preds is None:
                raise ValueError(f"Predictions for model2 are nan. \n{emsg}")

            # Log mew metrics
            predict_metrics = self._log_predict_metrics(predict_metrics, r,
                                                        teams=(team1, team2),
                                                        x_inputs=(team1_xinput, team2_xinput),
                                                        y_inputs=(team1_yinput, team2_yinput),
                                                        preds=(team1_preds, team2_preds))

            self.models[team1].matches_data[predict_dataset]["idx"] += 1
            self.models[team2].matches_data[predict_dataset]["idx"] += 1

        # Get max number of indices depending on length of datasets
        if predict_dataset == Dataset.Test:
            max_range = self.data_loader.max_ntest_len
        else:
            max_range = self.data_loader.max_npredict_len
        # Create stats file for prediction
        metrics = list(predict_metrics[teams[0]].keys())
        multiindex = pd.MultiIndex.from_product([teams, metrics], names=["team", "metric"])
        self.predictions[predict_dataset] = pd.DataFrame([], index=range(0, max_range), columns=multiindex)

        # Save stats
        for t in teams:
            for m in metrics:
                self.predictions[predict_dataset].loc[0:len(predict_metrics[t]),
                                                      (t, m)] = pd.Series(predict_metrics[t][m])

    def _log_predict_metrics(self,
                             predict_metrics: Dict[str, Any],
                             r: pd.Series,
                             teams: Tuple[str, str],
                             x_inputs: Tuple[Any, Any],
                             y_inputs: Tuple[Any, Any],
                             preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Logs predictions stats and metrics.
        Prettier formating is applied to some values - mainly floats.

        Some stats are logged in format e.g.: <odds_wd> - <odds_wd> meaning that the first value
        is from the viewpoint of current team, and the second value is from viewpoint of the
        opponent.

        :param predict_metrics: Dict with current prediction metrics.
        :param r: Row of the dataframe.
        :param teams: Both teams names.
        :param x_inputs: X input values used for both models.
        :param y_inputs: Y input values used for both models.
        :param preds: Predictions of both models.
        :return: Updated prediction metrics dict.
        """
        team_sides = ["home", "away"]

        output = defaultdict(lambda: defaultdict())
        output[teams[0]]["sides"] = team_sides
        output[teams[0]]["target"] = y_inputs[0]
        output[teams[0]]["preds"] = preds[0]
        output[teams[0]]["ashome"] = x_inputs[0]["input_team1_future_ashome"].flatten()[-1]
        output[teams[1]]["sides"] = team_sides[::-1]
        output[teams[1]]["target"] = y_inputs[1]
        output[teams[1]]["preds"] = preds[1]
        output[teams[1]]["ashome"] = x_inputs[1]["input_team1_future_ashome"].flatten()[-1]

        # Get last value of odds from input for each team and rescale values
        odds_wd = [x_inputs[0]["input_team1_future_odds_wd"].flatten()[-1],
                   x_inputs[0]["input_team2_future_odds_wd"].flatten()[-1]]
        odds_wd = [self.data_loader.inverse_transform("odds_wd", k) for k in odds_wd]
        output[teams[0]]["odds_wd"] = odds_wd
        output[teams[1]]["odds_wd"] = odds_wd[::-1]
        output[teams[0]]["bmpred"] = 1 if odds_wd[0] <= odds_wd[1] else 0
        output[teams[1]]["bmpred"] = 1 if odds_wd[1] < odds_wd[0] else 0

        for team, o in output.items():
            t1 = o["sides"][0]
            t2 = o["sides"][1]

            # Set some attributes to None if results of the match are unknown
            if output[team]["target"] is None:
                target = None
                match_goals = None
                rating = None
                errors = None
                red_cards = None
            else:
                target = o["target"]["output"].flatten()[0]
                match_goals = f"{r[f'{t1}_goals']} - {r[f'{t2}_goals']}"
                rating = f"{r[f'{t1}_rating']:.2f} - {r[f'{t2}_rating']:.2f}"
                errors = f"{r[f'{t1}_errors']} - {r[f'{t2}_errors']}"
                red_cards = f"{r[f'{t1}_red_cards']} - {r[f'{t2}_red_cards']}"

            predict_metrics[team]["target"].append(target)
            predict_metrics[team]["pred"].append(output[team]["preds"].argmax())
            predict_metrics[team]["pred_perc"].append(f"{output[team]['preds'].max()}")
            predict_metrics[team]["preds_all"].append(", ".join([f"{i:.1%}" for i in output[team]["preds"].tolist()]))
            predict_metrics[team]["match_date"].append(r["date"])
            predict_metrics[team]["opponent"].append(r[f'{t2}'])
            predict_metrics[team]["match_goals"].append(match_goals)
            predict_metrics[team]["rating"].append(rating)
            predict_metrics[team]["ashome"].append(output[team]["ashome"])
            predict_metrics[team]["errors"].append(errors)
            predict_metrics[team]["red_cards"].append(red_cards)
            predict_metrics[team]["league"].append(r["league"])
            predict_metrics[team]["season"].append(r["season"])
            predict_metrics[team]["odds_wd"].append(" - ".join([f"{i:.2f}" for i in output[team]['odds_wd']]))
            predict_metrics[team]["match_id"].append(r["id"])
            predict_metrics[team]["bmpred"].append(output[team]["bmpred"])
            predict_metrics[team]["bmpred_perc"].append(1 / output[team]["odds_wd"][0])

        return predict_metrics

    def save(self,
             models: bool = False,
             train: bool = False,
             test: bool = False,
             predict: bool = False) -> None:
        """
        Saves models, stats files, predictions, and model settings.

        :param models: Whether to save models.
        :param train: Whether to save training stats.
        :param test: Whether to save test stats.
        :param predict: Whether to save prediction stats.
        """
        print("Saving...")

        mode = SaveLoad.Load if self._predict else SaveLoad.Save
        generic_path = self._form_generic_file_path(mode)

        if models:
            self._save_models(generic_path)
            self._save_model_settings(generic_path)

        # Save both train and best train stats
        if train:
            train_path = spc.form_stats_file_path(generic_path, Dataset.Train)
            self.train_stats.to_pickle(str(train_path))

            train_path_best = spc.form_best_stats_file_path(generic_path, Dataset.Train)
            self.best_train_stats.to_pickle(str(train_path_best))

        # Save both test and best test stats
        if test:
            test_path = spc.form_stats_file_path(generic_path, Dataset.Test)
            self.test_stats.to_pickle(str(test_path))

            test_path_best = spc.form_best_stats_file_path(generic_path, Dataset.Test)
            self.best_test_stats.to_pickle(str(test_path_best))

        # Save both prediction files - for test and predict datasets
        if predict:
            for dataset in [Dataset.Test, Dataset.Predict]:
                path = spc.form_prediction_file_path(generic_path, dataset)
                self.predictions[dataset].to_pickle(str(path))

    def _load_models(self, teams: List[str], include_optimizer: bool) -> None:
        """
        Loads all models data which was previously saved.

        :param teams: Teams to load models for.
        :param include_optimizer: Whether to load optimizer weights.
        """
        print("Loading models...")
        generic_path = self._form_generic_file_path(SaveLoad.Load)

        # Warms up models if we continue in training and are loading optimizer as well.
        # This is required in order to properly initialize optimizer weights before restoring them.
        # Otherwise they would have different shapes and could not be matched with the saved ones.
        if self._previous_epochs > 0 and include_optimizer:
            print("Warming up models...")
            for t in teams:
                self.models[t].warm_up()

        # Finally, restore all previous params - weights, states, optimizer, and other variables
        print("Restoring models weights...")
        for t in teams:
            save_data = np.load(str(spc.form_data_file_path(generic_path, t)), allow_pickle=True)[()]
            self.models[t].load_data_from_file(save_data, include_optimizer)

    def _save_models(self, generic_path: Path) -> None:
        """
        Saves all necessary data needed for correct continuation in training when
        reloaded again.

        :param generic_path: Generic output path.
        """
        for team, model in self.models.items():
            np.save(str(spc.form_data_file_path(generic_path, team)), model.get_save_data())

    def _save_model_settings(self, generic_path: Path) -> None:
        """
        Saves model settings.

        :param generic_path: Generic output path.
        """
        # General settings saved during first run only
        if not self._model_settings:
            self._model_settings = {
                "seasons": self._seasons,
                "folder_prefix": self._folder_prefix,
                "teams_names_bitlen": self.data_loader.teams_names_bitlen,
                "features": self._features,
                "teams": self.data_loader.teams,
                "last_season_teams": self.data_loader.last_season_teams,
                "train_teams": self.data_loader.train_teams,
                "test_teams": self.data_loader.test_teams,
                "predict_teams": self.data_loader.predict_teams,
                "train_fixtures_ids": self.data_loader.train_fixtures_ids,
                "test_fixtures_ids": self.data_loader.test_fixtures_ids,
                "predict_fixtures_ids": self.data_loader.predict_fixtures_ids,
                "max_season": int(self.data_loader.max_season),
                "class_weights": {team: model.class_weights for team, model in self.models.items()},
                "config": vars(self._config),
                "best_epochs": {t: m.best_epoch for t, m in self.models.items()},
                "runs": {},
            }

        # Settings saved each run
        self._model_settings["runs"][len(self._model_settings["runs"].keys())] = {
            "timestamp_start": self._timestamp,
            "previous_epochs": self._previous_epochs,
            "epochs": self._epochs,
            "current_run_epochs": self._total_epochs_passed - self._previous_epochs,
            "avg_runtime_per_epoch_in_secs": float(np.mean(self._runtimes_per_epoch)),
            "total_runtime_in_secs": float(np.sum(self._runtimes_per_epoch)),
        }

        with open(str(spc.form_model_settings_file_path(generic_path)), "w") as fh:
            json.dump(self._model_settings, fh)

    def _load_previous_settings(self, folder: Path) -> None:
        """
        Loads previously saved model settings.

        :param folder: Path to folder where the settings is saved.
        """
        # Extract name of the latest models dir
        name = re.search(FOLDER_NAME_PATTERN, folder.name)
        # Extract number of previous epochs from the name
        self._folder_prefix = name.group(1)
        self._previous_timestamp = name.group(2)

        if self._resume:
            self._previous_epochs = int(name.group(3))
            self._total_epochs_passed = self._previous_epochs
            self._total_epochs = self._previous_epochs + self._epochs

            # Load previous stats files
            self.train_stats = spc.get_stats_file(folder, Dataset.Train)
            self.test_stats = spc.get_stats_file(folder, Dataset.Test)
            self.best_test_stats = spc.get_best_stats_file(folder, Dataset.Test)

            # Do not load previous best train stats file in case of only one season because all
            # teams would be considered as test teams and the file would be empty which would
            # raise error
            if len(self._seasons) > 1:
                self.best_train_stats = spc.get_best_stats_file(folder, Dataset.Train)

        if self._verbose > 0:
            print(f"Previous epochs trained for: {self._previous_epochs}")

    def _form_generic_file_path(self, mode: SaveLoad) -> Path:
        """
        Properly forms generic output path (common for all outputs of the model).
        If mode is set to Save, then a new timestamp and epochs number are used, if it is set to Load,
        then previous timestamp and epochs number are used.

        Also creates folders recursively if the path does not exist.

        :param mode: Whether we are saving or loading model.
        :return: Generic output path.
        """
        path = Path(os.getcwd()).joinpath(f"{DATA_DIR}{MODEL_DIR}")

        if mode == SaveLoad.Save:
            path = path.joinpath(f"{self._folder_prefix}_{self._timestamp}_{self._total_epochs_passed}")
        elif mode == SaveLoad.Load:
            path = path.joinpath(f"{self._folder_prefix}_{self._previous_timestamp}_{self._previous_epochs}")

        if not path.exists():
            path.mkdir(parents=True)

        return path

    def cleanup(self) -> None:
        """
        Closes Tensorboard FileWriters for all models.

        """
        for model in self.models.values():
            model.tensorboard.close()
