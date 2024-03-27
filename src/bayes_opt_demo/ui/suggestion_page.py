import pandas as pd
import streamlit as st
from ax.plot.slice import interact_slice_plotly
from ax.service.ax_client import AxClient

from bayes_opt_demo.dataset import (
    Dataset,
    Objectives,
    ObjFloatColumn,
    ParamCategoricalColumn,
    Parameters,
    ParamFloatColumn,
)
from bayes_opt_demo.math import round_range
from bayes_opt_demo.optimization import bayes_optimize


def suggestion_page(df: pd.DataFrame):
    """実験提案ページ"""
    # 目的変数の選択
    st.header("目的変数の選択")
    st.text(
        "チェックしていないものが説明変数、チェックしたものが目的変数として扱われます"
    )
    objective_columns = objective_columns_select(df)
    parameter_series = pd.DataFrame(
        {column: df[column] for column in df.columns if column not in objective_columns}
    )
    objective_series = pd.DataFrame(
        {column: df[column] for column in df.columns if column in objective_columns}
    )

    # 各変数の設定
    st.header("探索の設定")

    with st.expander("目標", expanded=True):
        if objective_series.empty:
            st.info("目的変数を選択してください", icon="⚠")
            objectives: Objectives = {}
        else:
            objectives: Objectives = objective_config_input(objective_series)

    with st.expander("探索空間", expanded=False):
        if parameter_series.empty:
            st.info("説明変数が存在しません", icon="⚠")
            parameters: Parameters = {}
        else:
            parameters = parameter_config_input(parameter_series)

    dataset = Dataset(
        parameters=parameters,
        objectives=objectives,
    )

    st.divider()

    # ベイズ最適化の開始
    st.header("実験候補を生成する")
    max_trials = st.number_input("生成数の上限", value=5, min_value=1)

    can_start = all(
        [
            len(parameters) > 0,
            len(objectives) > 0,
        ]
    )
    if not can_start:
        st.button(
            "実行",
            disabled=not can_start,
            help="説明変数と目的変数はそれぞれ1つ以上必要です",
        )
        return

    if not st.button("実行", disabled=not can_start):
        return

    # ベイズ最適化を開始する
    max_trials = max(int(max_trials), 1)
    candidates, ax_client = bayes_optimize(dataset, max_trials=max_trials)

    suggestion_tab, prediction_tab = st.tabs(["提案データ", "モデルの予測値"])

    # 実験候補を表示する
    with suggestion_tab:
        candidates = pd.DataFrame(candidates)
        st.dataframe(candidates)

    # ベイズ最適化の結果を表示する
    with prediction_tab:
        render_predictions(ax_client)


def objective_columns_select(df: pd.DataFrame) -> list[str]:
    """変数から目的変数を選択するウィジェット。"""
    objective_columns: list[str] = []
    for column in df.columns:
        # カテゴリカル変数は目的変数にできない
        is_str = df[column].dtype == "object"
        if st.checkbox(column, disabled=is_str):
            objective_columns.append(column)

    return objective_columns


def parameter_config_input(parameter_df: pd.DataFrame) -> Parameters:
    """説明変数の設定を行うウィジェット。"""

    parameters: Parameters = {}
    for column in parameter_df.columns:
        if parameter_df[column].dtype == "object":
            _options: list[str | None] = list(parameter_df[column].unique())
            options = [option for option in _options if option is not None]

            st.multiselect(
                f"{column}の選択肢 (固定)", options, default=options, disabled=True
            )

            parameters[column] = ParamCategoricalColumn(
                series=parameter_df[column],
                options=options,
            )
        else:
            col1, col2 = st.columns(2, gap="medium")
            default_lower, default_higher = round_range(
                float(parameter_df[column].min()), float(parameter_df[column].max())
            )
            lower_bound = col1.number_input(f"{column}の下限", value=default_lower)
            upper_bound = col2.number_input(f"{column}の上限", value=default_higher)

            parameters[column] = ParamFloatColumn(
                series=parameter_df[column],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

    return parameters


def objective_config_input(objective_df: pd.DataFrame) -> Objectives:
    """目的変数の設定を行うウィジェット。"""
    objectives: Objectives = {}
    for column in objective_df.columns:
        if not pd.api.types.is_float_dtype(objective_df[column]):
            raise ValueError(
                f"{column}は数値列ではないので目的変数として指定できません"
            )
            continue

        value = st.radio(
            f"{column}の目標", options=["最大化", "最小化"], horizontal=True
        )
        match value:
            case "最大化":
                minimize = False
            case "最小化":
                minimize = True
            case None:
                continue
            case _:
                raise ValueError(f"{column}の目標の値が想定外です: {value}")

        objectives[column] = ObjFloatColumn(
            series=objective_df[column],
            minimize=minimize,
        )

    return objectives


def render_predictions(ax_client: AxClient):
    """モデルの予測値を表示する。"""

    st.subheader("モデルの予測値")
    fig = interact_slice_plotly(
        model=ax_client.generation_strategy.model,  # type: ignore
    )
    st.plotly_chart(fig)

    # TODO: contourによる可視化
