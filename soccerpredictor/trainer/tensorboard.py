import numpy as np
import pandas as pd
from typing import Dict

from tensorflow.compat.v1 import Session
from tensorflow.compat.v1.summary import FileWriter, Summary

from soccerpredictor.util.config import SPConfig
from soccerpredictor.util.constants import TB_ROOT_DIR, TB_TRAIN_DIR, TB_MAX_QUEUE, TB_FLUSH_SECS, \
    TB_TEST_DIR, DATA_DIR


class SPTensorboard:
    """
    The logging needs to be done manually by FileWriters mainly due to using multiple sessions.
    If team is included in test dataset then additional FileWriter is used for tracking its performance
    on the test dataset. Otherwise only FileWriter for train dataset are used.
    Both train and test FileWriters log the best metrics recorded so far. They are writing into the same file
    so they can be viewed in the same plot in Tensorbaord.

    No logging is done when in prediction-only mode.

    Uses prefix "_target_" for teams in test dataset and "others_" for teams not in test dataset. This makes
    filtering in Tensorboard easier.

    """

    def __init__(self, team_name: str, target_team: bool, session: Session, folder_prefix: str) -> None:
        """

        :param team_name: Model's team name.
        :param target_team: Whether the team is included in test dataset.
        :param session: Model's session.
        :param folder_prefix: Current output folder prefix.
        """
        self._predict = SPConfig().predict
        self._session = session
        self._team_name = team_name
        self._target_team = target_team
        self._prefix = "_target_" if self._target_team else "others_"
        base_dir = f"{DATA_DIR}{TB_ROOT_DIR}{folder_prefix}/"

        if self._predict:
            return

        with self._session.as_default(), self._session.graph.as_default():
            # FileWriter for tracking train stats is used for every model
            self._filewriter_train = FileWriter(f"{base_dir}{TB_TRAIN_DIR}{self._team_name}",
                                                session=self._session,
                                                max_queue=TB_MAX_QUEUE,
                                                flush_secs=TB_FLUSH_SECS)
            self._filewriter_best_train = None
            self._filewriter_test = None
            self._filewriter_best_test = None

            # FileWriters for logging test stats are available only for teams included in test dataset
            if self._target_team:
                self._filewriter_test = FileWriter(f"{base_dir}{TB_TEST_DIR}{self._team_name}",
                                                   session=self._session,
                                                   max_queue=TB_MAX_QUEUE,
                                                   flush_secs=TB_FLUSH_SECS)

                self._filewriter_best_test = FileWriter(f"{base_dir}{TB_TEST_DIR}{self._team_name}",
                                                        session=self._session,
                                                        max_queue=TB_MAX_QUEUE,
                                                        flush_secs=TB_FLUSH_SECS)

            # If team is not in the test dataset then track its best stats on train dataset
            else:
                self._filewriter_best_train = FileWriter(f"{base_dir}{TB_TRAIN_DIR}{self._team_name}",
                                                         session=self._session,
                                                         max_queue=TB_MAX_QUEUE,
                                                         flush_secs=TB_FLUSH_SECS)

    def notify_train(self, epoch: int, metrics: Dict[str, np.ndarray]) -> None:
        """
        Logs training metrics.

        :param epoch: Current epoch.
        :param metrics: Metrics to log.
        """
        if self._predict:
            return

        with self._session.as_default(), self._session.graph.as_default():
            summary = Summary()
            summary.value.add(tag=f"{self._prefix}{self._team_name}/_loss", simple_value=metrics["loss"])
            summary.value.add(tag=f"{self._prefix}{self._team_name}/_acc", simple_value=metrics["acc"])
            self._filewriter_train.add_summary(summary, epoch)

    def notify_best_train(self, best_epoch: int, epoch: int, train_stats: pd.DataFrame) -> None:
        """
        Copies metrics since the last best train epoch recorded up to current epoch when the new best
        train epoch is found.

        :param best_epoch: The best train epoch recorded so far.
        :param epoch: Current epoch.
        :param train_stats: Df with train stats.
        """
        if self._predict:
            return

        if not self._target_team:
            with self._session.as_default(), self._session.graph.as_default():
                epochs_range = range(0, epoch+1) if best_epoch is None else range(best_epoch, epoch+1)
                for i in epochs_range:
                    summary = Summary()
                    for m in train_stats.columns.get_level_values("metric"):
                        summary.value.add(tag=f"{self._prefix}{self._team_name}/best_{m}",
                                          simple_value=train_stats.loc[i, (self._team_name, m)])
                    self._filewriter_best_train.add_summary(summary, i)

    def notify_test(self, epoch: int, metrics: Dict[str, np.ndarray]) -> None:
        """
        Logs test metrics.

        :param epoch: Current epoch.
        :param metrics: Metrics to log.
        """
        if self._predict:
            return

        with self._session.as_default(), self._session.graph.as_default():
            if self._target_team:
                summary = Summary()
                summary.value.add(tag=f"{self._prefix}{self._team_name}/_loss", simple_value=metrics["loss"])
                summary.value.add(tag=f"{self._prefix}{self._team_name}/_acc", simple_value=metrics["acc"])
                self._filewriter_test.add_summary(summary, epoch)

    def notify_best_test(self, best_epoch: int, epoch: int, test_stats: pd.DataFrame) -> None:
        """
        Copies metrics since the last best test epoch recorded up to current epoch when the new best
        test epoch is found.

        :param best_epoch: The best test epoch recorded so far.
        :param epoch: Current epoch.
        :param test_stats: Df with test stats.
        """
        if self._predict:
            return

        if self._target_team:
            with self._session.as_default(), self._session.graph.as_default():
                epochs_range = range(0, epoch+1) if best_epoch is None else range(best_epoch, epoch+1)
                for i in epochs_range:
                    summary = Summary()
                    for m in test_stats.columns.get_level_values("metric"):
                        summary.value.add(tag=f"{self._prefix}{self._team_name}/best_{m}",
                                          simple_value=test_stats.loc[i, (self._team_name, m)])
                    self._filewriter_best_test.add_summary(summary, i)

    def close(self) -> None:
        """
        Closes all opened FileWriters.
        FileWriters for the best stats are writing into the same file as regular ones so only one of them
        needs to be closed.

        """
        if self._predict:
            return

        with self._session.as_default(), self._session.graph.as_default():
            self._filewriter_train.flush()
            if self._filewriter_best_train:
                self._filewriter_best_train.flush()
            self._filewriter_train.close()

            if self._target_team:
                self._filewriter_test.flush()
                self._filewriter_best_test.flush()
                self._filewriter_test.close()
