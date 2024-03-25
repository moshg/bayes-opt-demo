import pandas as pd
import streamlit as st
from ax.plot.contour import interact_contour_plotly
from ax.service.ax_client import AxClient
from bayes_opt_demo.dataset import (
    Dataset,
    Objectives,
    ObjFloatColumn,
    ParamCategoricalColumn,
    Parameters,
    ParamFloatColumn,
)
from pygwalker.api.streamlit import StreamlitRenderer


def csv_uploader():
    """CSVファイルをアップロードし、DataFrameを返すウィジェット。
    アップロード前やCSVでないファイルがアップロードされた場合はNoneを返す。

    DataFrameはfloat, str, Noneのみを含む。
    """
    file = st.file_uploader("ヘッダー付きCSVファイルをアップロードしてください")

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


def render_bayes_opt(dataset: Dataset, ax_client: AxClient):
    """ベイズ最適化の結果を表示する。"""

    # FIXME: クリックがfalseになるので、ラジオボタンの選択ができない
    # if len(dataset.objectives) > 1:
    #     objective_name = st.radio("目的変数", options=dataset.objectives.keys())

    #     if objective_name is None:
    #         return

    #     objective = dataset.objectives.get(objective_name)
    #     if objective is None:
    #         st.error(f"目的変数 {objective_name} が見つかりません", icon="⚠")
    #         return

    objective_name = next(iter(dataset.objectives.keys()))
    objective = next(iter(dataset.objectives.values()))
    if objective is None:
        st.error(f"目的変数 {objective_name} が存在しません", icon="⚠")
        return

    fig = interact_contour_plotly(
        model=ax_client.generation_strategy.model,  # type: ignore
        metric_name=objective_name,
        lower_is_better=objective.minimize,
    )
    st.plotly_chart(fig)

    st.info("目的変数の選択は未実装です", icon="⚠")


@st.cache_data(ttl="1h", max_entries=1)
def get_pyg_renderer(df: pd.DataFrame) -> StreamlitRenderer:
    """PygWalkerのStreamlitRendererを取得する。

    ref. https://docs.kanaries.net/ja/pygwalker/use-pygwalker-with-streamlit
    """
    return StreamlitRenderer(df, debug=False)
