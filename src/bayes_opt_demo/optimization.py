"""ベイズ最適化を行うための関数"""

from typing import Any, Iterable, assert_never

import pandas as pd
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, ObjectiveProperties

from bayes_opt_demo.dataset import Dataset, ParamCategoricalColumn, ParamFloatColumn


def bayes_optimize(
    dataset: Dataset, max_trials: int
) -> tuple[list[dict[str, float | str | None]], AxClient]:
    """ベイズ最適化を行い、候補パラメータとAxClientを返す。"""

    # ref. https://ax.dev/tutorials/generation_strategy.html
    generation_strategy = GenerationStrategy(
        [
            GenerationStep(
                model=Models.BO_MIXED
                if dataset.has_categorical_parameters()
                else Models.GPEI,
                num_trials=-1,
            )
        ]
    )
    ax_client = AxClient(generation_strategy=generation_strategy)

    dataset = preprocess(dataset)

    # 探索空間を設定する
    create_experiment(ax_client, dataset)

    # モデルを学習させる
    optimize(ax_client, dataset)

    # 候補パラメータを取得する
    candidates, _ = ax_client.get_next_trials(max_trials)

    candidates = postprocess(dataset, candidates.values())

    return candidates, ax_client


def preprocess(dataset: Dataset) -> Dataset:
    """データセットを前処理する。

    欠損値を埋める。
    - 数値型の場合は平均値で埋める
    - カテゴリカル型の場合は最頻値で埋める
    """
    return dataset.map_series(
        lambda series: series.fillna(series.mean())
        if pd.api.types.is_float_dtype(series)
        else series.fillna(series.mode().iloc[0])
    )


def postprocess(
    dataset: Dataset, candidates: Iterable[dict[str, float | str | None]]
) -> list[dict[str, float | str | None]]:
    """候補パラメータを後処理する。

    x1, x3, x2のように、入力と異なる順序で候補パラメータが返される可能性があるので順序を揃える。
    Python 3.7以降では辞書の順序は挿入順になる: https://docs.python.org/ja/3.12/whatsnew/3.7.html
    """

    sorted_candidates: list[dict[str, float | str | None]] = []
    for candidate in candidates:
        sorted_candidate: dict[str, float | str | None] = {}
        for name in dataset.parameters.keys():
            sorted_candidate[name] = candidate[name]
        sorted_candidates.append(sorted_candidate)

    return sorted_candidates


def create_experiment(ax_client: AxClient, dataset: Dataset):
    """AxClientに実験を登録する。"""
    # 説明変数の制約をAXの形式に変換する
    parameters: list[dict[str, Any]] = []
    for name, column in dataset.parameters.items():
        match column:
            case ParamCategoricalColumn():
                parameters.append(
                    {
                        "name": name,
                        "type": "choice",
                        "values": column.options,
                    }
                )
            case ParamFloatColumn():
                parameters.append(
                    {
                        "name": name,
                        "type": "range",
                        "bounds": [column.lower_bound, column.upper_bound],
                        "value_type": "float",
                    }
                )
            case _:
                assert_never(column)

    objectives: dict[str, ObjectiveProperties] = {
        name: ObjectiveProperties(minimize=column.minimize)
        for name, column in dataset.objectives.items()
    }

    ax_client.create_experiment(
        name="bayesian optimization demo",
        parameters=parameters,
        objectives=objectives,
    )


def optimize(ax_client: AxClient, dataset: Dataset):
    for experiment in dataset.experiments():
        _, trial_index = ax_client.attach_trial(experiment.parameters)
        ax_client.complete_trial(
            trial_index=trial_index,
            # dict[str, float] は dict[str, float | str] に代入できないので
            # 型を合わせるためにshallow copyする
            raw_data={k: v for k, v in experiment.objectives.items()},
        )
