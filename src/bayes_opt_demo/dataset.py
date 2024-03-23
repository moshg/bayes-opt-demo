"""データの値と制約に関する型"""

from dataclasses import dataclass
from typing import Callable, TypeAlias, Union

import pandas as pd


class Dataset:
    """データの値と制約の集合"""

    parameters: "Parameters"
    """説明変数"""
    objectives: "Objectives"
    """目的変数"""

    _count: int
    """データ数"""

    def __init__(self, parameters: "Parameters", objectives: "Objectives"):
        count = None
        for column in parameters.values():
            if count is None:
                count = len(column.series)
            elif count != len(column.series):
                raise ValueError("seriesの長さは同じでなければなりません。")

        for column in objectives.values():
            if count is None:
                count = len(column.series)
            elif count != len(column.series):
                raise ValueError("seriesの長さは同じでなければなりません。")

        if count is None:
            raise ValueError("データがすべて空です。")

        self.parameters = parameters
        self.objectives = objectives
        self._count = count

    def experiments(self):
        """データセットから実験の列を生成する。"""
        for i in range(self._count):
            yield Experiment(
                parameters={
                    name: column.series[i] for name, column in self.parameters.items()
                },
                objectives={
                    name: column.series[i] for name, column in self.objectives.items()
                },
            )

    def map_series(self, func: Callable[[pd.Series], pd.Series]) -> "Dataset":
        """データセットの各列に関数を適用する。"""
        return Dataset(
            parameters={
                name: ParamFloatColumn(
                    func(column.series), column.lower_bound, column.upper_bound
                )
                if isinstance(column, ParamFloatColumn)
                else ParamCategoricalColumn(func(column.series), column.options)
                for name, column in self.parameters.items()
            },
            objectives={
                name: ObjFloatColumn(func(column.series), column.minimize)
                for name, column in self.objectives.items()
            },
        )


@dataclass
class Experiment:
    parameters: dict[str, float | str | None]
    objectives: dict[str, float]


Parameters: TypeAlias = dict[str, Union["ParamFloatColumn", "ParamCategoricalColumn"]]


@dataclass
class ParamFloatColumn:
    """数値型の説明変数の値と制約"""

    series: pd.Series
    lower_bound: float
    upper_bound: float


@dataclass
class ParamCategoricalColumn:
    """カテゴリカル型の説明変数の値と制約"""

    series: pd.Series
    options: list[str]


Objectives: TypeAlias = dict[str, "ObjFloatColumn"]


@dataclass
class ObjFloatColumn:
    """数値型の目的変数の値と制約"""

    series: pd.Series
    minimize: bool
