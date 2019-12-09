#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import os
import pandas as pd
from sqlite3 import OperationalError
from string import ascii_uppercase
import sys
from typing import Optional

from soccerpredictor.trainer.dbmanager import SPDBManager
from soccerpredictor.util.common import get_latest_models_dir, get_model_settings_file
from soccerpredictor.util.constants import FOLDER_PREFIX_LEN, DATA_DIR, MODEL_DIR, VERBOSITY_LEVELS
from soccerpredictor.util.config import SPConfig
from soccerpredictor.util.enums import RunMode

# For printing stats during training
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", -1)


def main() -> None:
    """
    Runs model training or visualization.

    """
    parser = ArgumentParser(description="SoccerPredictor:", formatter_class=ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title="Modes to run", dest="command")

    # Trainer args
    trainer_parser = subparsers.add_parser(RunMode.Train.value, help="Trains model and makes predictions.",
                                           formatter_class=ArgumentDefaultsHelpFormatter)
    trainer_parser.add_argument("--resume", action="store_true", default=False,
                                help="Resumes training of previously saved model. "
                                     "Tries to load the latest model saved if no name or prefix specified via --name.")
    trainer_parser.add_argument("--epochs", type=int, action="store", default=1,
                                help="Number of epochs to train model for.")
    trainer_parser.add_argument("--ntest", type=int, action="store", default=10,
                                help="Number of last samples used for testing for each team.")
    trainer_parser.add_argument("--ndiscard", type=int, action="store", default=0,
                                help="Number of last samples to discard for each team.")
    trainer_parser.add_argument("--timesteps", type=int, action="store", default=40,
                                help="Number of timesteps to use as data window size for input to network.")
    trainer_parser.add_argument("--predict", action="store_true", default=False,
                                help="Whether to rerun predictions without any training.")
    trainer_parser.add_argument("--lrpatience", type=int, action="store", default=20,
                                help="How many epochs to tolerate before decaying learning rate if no improvement. "
                                     "Turned off if 0.")
    trainer_parser.add_argument("--lrdecay", type=float, action="store", default=0.95,
                                help="How much to decay learning rate after patience exceeded.")
    trainer_parser.add_argument("--seed", type=int, action="store",
                                help="Specifies seed for rng.")
    trainer_parser.add_argument("--savefreq", type=int, action="store", default=50,
                                help="How often (number of epochs) to save models. No intermediate saving if 0.")
    trainer_parser.add_argument("--printfreq", type=int, action="store", default=10,
                                help="How often (number of epochs) to print current summaries. "
                                     "No intermediate printing if 0.")
    trainer_parser.add_argument("--verbose", type=int, action="store", choices=VERBOSITY_LEVELS, default=1,
                                help="Level of verbosity.")

    # Visualizer args
    visualizer_parser = subparsers.add_parser(RunMode.Vis.value, help="Runs visualization of predictions.",
                                              formatter_class=ArgumentDefaultsHelpFormatter)
    visualizer_parser.add_argument("--port", type=int, action="store", default=8050,
                                   help="Custom port for Dash visualization.")
    visualizer_parser.add_argument("--host", type=str, action="store", default="127.0.0.1",
                                   help="Custom host for Dash visualization. Can use 0 for 0.0.0.0 shortcut.")

    # Backtester args
    backtester_parser = subparsers.add_parser(RunMode.Backtest.value, help="Runs backtesting on trained models.",
                                              formatter_class=ArgumentDefaultsHelpFormatter)
    backtester_parser.add_argument("--path", type=str, action="store", default=f"{DATA_DIR}{MODEL_DIR}",
                                   help="Path to folder where the trained models are saved.")

    # common args for trainer and visualizer
    for p in [trainer_parser, visualizer_parser]:
        p.add_argument("--name", type=str, action="store",
                       help="Tries to load the latest saved model with given name prefix. "
                            "Loads exact model if exact dir name specified.")

    # Common args for visualizer and backtester
    for p in [visualizer_parser, backtester_parser]:
        p.add_argument("--ignoreodds", type=float, action="store", default=1.10,
                       help="Ignores odds less than given amount when predicting which team to bet on.")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return
    elif args.command == RunMode.Vis.value:
        print("Visualizing...")
        vis_args, _ = visualizer_parser.parse_known_args()
        check_visualizer_args(visualizer_parser, vis_args)
        print(vis_args)

        from soccerpredictor.visualizer import visualizer
        try:
            visualizer.run(vis_args.name, vis_args.host, vis_args.port, vis_args.ignoreodds)
        except KeyboardInterrupt:
            print("> Received CTRL+C command. Exiting.")
    elif args.command == RunMode.Backtest.value:
        print("Backtesting...")
        backtest_args, _ = backtester_parser.parse_known_args()
        check_backtester_args(backtester_parser, backtest_args)
        print(backtest_args)

        from soccerpredictor.backtester import backtester
        try:
            backtester.run(backtest_args.path, backtest_args.ignoreodds)
        except KeyboardInterrupt:
            print("> Received CTRL+C command. Exiting.")
    elif args.command == RunMode.Train.value:
        print("Running model...")
        train_args, _ = trainer_parser.parse_known_args()
        config = SPConfig()

        # Implicitly set resume to true if we are predicting only
        if train_args.predict and not train_args.resume:
            train_args.resume = True

        check_trainer_args(trainer_parser, train_args)
        config.set_args(train_args)
        print(train_args)

        dbmanager = SPDBManager()
        try:
            dbmanager.connect()

            # Load previous settings if we resume training
            if train_args.resume:
                print("Resuming training, loading previous settings... "
                      "Any conflicting parameters will be ignored.")

                # Load previous settings
                folder = get_latest_models_dir(train_args.name)
                model_settings = get_model_settings_file(folder)
                # Restore original config
                config.restore_args(model_settings)
                set_rng_seed(config.seed)

                from soccerpredictor.trainer.trainer import SPTrainer
                trainer = SPTrainer(dbmanager, model_settings=model_settings, folder=folder)
            else:
                # Need to generate folder prefix before seeding random number generators
                import random
                generated_folder_prefix = "".join(random.choices(ascii_uppercase, k=FOLDER_PREFIX_LEN))
                print(f"New generated folder prefix: '{generated_folder_prefix}'")
                set_rng_seed(train_args.seed)

                from soccerpredictor.trainer.trainer import SPTrainer
                trainer = SPTrainer(dbmanager, generated_folder_prefix=generated_folder_prefix)

            try:
                trainer.run()
            finally:
                trainer.cleanup()

        except KeyboardInterrupt:
            print("> Received CTRL+C command. Exiting.")
        except (FileNotFoundError, ValueError, OperationalError) as e:
            print(e)
            sys.exit(1)
        finally:
            dbmanager.disconnect()


def set_rng_seed(seed: Optional[int]) -> None:
    """
    Sets seed of random number generators.
    Setting seeds will not ensure 100 % reproducibity but at least the same starting point.

    PYTHONHASHSEED should, ideally, be set before running program, e.g. by:
    PYTHONHASHSEED=0 python3 main.py train ...
    However, it seems it makes no difference.

    Seed for Tensorflow should be set before importing any Tensorflow or Keras modules.

    :param seed: Number to seed with.
    """
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(0)
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
        from tensorflow.compat.v1 import set_random_seed
        set_random_seed(seed)
        print(f"Running with seed: {seed}")


def check_trainer_args(parser: ArgumentParser, args: Namespace) -> None:
    """
    Checks trainer mode args.

    :param parser: Argument parser.
    :param args: Given arguments.
    """
    # Arguments checks
    if args.epochs <= 0:
        parser.error("Number of epochs must be >= 1.")
    if args.timesteps <= 0:
        parser.error("Number of timesteps must be >= 1.")
    if args.ntest <= 0:
        parser.error("Number of test samples must be >= 1")
    if args.ndiscard < 0:
        parser.error("Number of discarded samples must be >= 0")
    if args.savefreq < 0:
        parser.error("Epochs frequency of savefreq must be >= 0")
    if args.printfreq < 0:
        parser.error("Epochs frequency of printfreq must be >= 0")
    if args.seed is not None and args.seed < 0:
        parser.error("Rng must be seeded with number >= 0.")


def check_visualizer_args(parser: ArgumentParser, args: Namespace) -> None:
    """
    Checks visualizer mode args.

    :param parser: Argument parser.
    :param args: Given arguments.
    """
    if args.port < 0:
        parser.error("Port must be positive integer.")
    check_ignoreodds_arg(parser, args)


def check_backtester_args(parser: ArgumentParser, args: Namespace) -> None:
    """
    Checks backtester mode args.

    :param parser: Argument parser.
    :param args: Given arguments.
    """
    check_ignoreodds_arg(parser, args)


def check_ignoreodds_arg(parser: ArgumentParser, args: Namespace) -> None:
    """
    Checks whether ignoreodds arg is within limits.

    :param parser: Argument parser.
    :param args: Given arguments.
    """
    if args.ignoreodds < 1.01:
        parser.error("Ignored odds must be >= 1.01 at least.")


if __name__ == "__main__":
    main()
