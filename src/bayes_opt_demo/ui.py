import pandas as pd
import streamlit as st

from bayes_opt_demo.dataset import (
    Objectives,
    ObjFloatColumn,
    ParamCategoricalColumn,
    Parameters,
    ParamFloatColumn,
)


def upload_csv():
    """CSVファイルをアップロードし、DataFrameを返すウィジェット。
    アップロード前やCSVでないファイルがアップロードされた場合はNoneを返す。

    DataFrameはfloat, str, Noneのみを含む。
    """
    file = st.file_uploader("CSVファイルをアップロードしてください")

    if file is None:
        return

    # ファイルをCSVとしてパースする
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"指定されたファイルはCSVではありません: {e}", icon="⚠")
        return

    # intはfloatに変換する
    for column in list(df.columns):
        if pd.api.types.is_integer_dtype(df[column]):
            df[column] = df[column].astype(float)

    return df


debug_df = pd.DataFrame(
    {
        "x1": [1.0, 2.0, 3.0],
        "x2": ["foo", "bar", "foo"],
        "x3": [2.0, 1.0, None],
        "y": [3.0, 2.0, 3.5],
    }
)
"""デバッグ用のDataFrame"""


def select_objective_columns(df: pd.DataFrame) -> list[str]:
    """変数から目的変数を選択するウィジェット。"""
    objective_columns: list[str] = []
    for column in df.columns:
        # カテゴリカル変数は目的変数にできない
        is_str = df[column].dtype == "object"
        if st.checkbox(column, disabled=is_str):
            objective_columns.append(column)

    return objective_columns


def input_parameter_config(parameter_df: pd.DataFrame) -> Parameters:
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
            lower_bound = col1.number_input(
                f"{column}の下限", value=float(parameter_df[column].min())
            )
            upper_bound = col2.number_input(
                f"{column}の上限", value=float(parameter_df[column].max())
            )

            parameters[column] = ParamFloatColumn(
                series=parameter_df[column],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

    return parameters


def input_objective_config(objective_df: pd.DataFrame) -> Objectives:
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
