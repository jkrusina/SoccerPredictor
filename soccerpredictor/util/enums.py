from enum import Enum, auto


class RunMode(Enum):
    """
    Available modes of program to run.

    """
    Train = "train"
    Vis = "vis"
    Backtest = "backtest"


class Dataset(Enum):
    """
    Available datasets types.

    """
    Train = "train"
    Test = "test"
    Predict = "predict"


class SaveLoad(Enum):
    """
    Save or Load toggle options.

    """
    Save = auto()
    Load = auto()


class TargetVariable(Enum):
    """
    Target variables available.

    """
    FutureWD = "future_wd"
