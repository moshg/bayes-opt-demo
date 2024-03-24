import pandas as pd
import streamlit as st
from bayes_opt_demo.dataset import Dataset, Objectives
from bayes_opt_demo.example_data import debug_df, generate_hartmann6_sobol
from bayes_opt_demo.optimization import bayes_optimize
from bayes_opt_demo.ui.widgets import (
    csv_uploader,
    get_pyg_renderer,
    objective_columns_select,
    objective_config_input,
    parameter_config_input,
)
from pygwalker.api.streamlit import init_streamlit_comm


def upload_page():
    """教師データのアップロードページ"""
    st.header("データのアップロード")

    upload_option = "アップロードされたCSV"
    demo_option = "デモ (Hartmann 6 Sobol)"
    debug_option = "デバッグデータ"

    data_src = st.radio(
        "使用データ",
        options=[upload_option, demo_option, debug_option],
        horizontal=True,
    )

    df: pd.DataFrame | None = None
    if data_src == upload_option:
        df = csv_uploader()
    if data_src == demo_option:
        # 実行ごとに値が変わらないようにseedを固定
        df = generate_hartmann6_sobol(12, seed=13)
    elif data_src == debug_option:
        df = debug_df

    if df is None:
        return

    # 読み込んだCSVの表示
    st.subheader("アップロードされたデータ")
    st.dataframe(df)

    return df


def suggestion_page(df: pd.DataFrame):
    """実験提案ページ"""
    # 目的変数の選択
    st.header("目的変数の選択")
    objective_columns = objective_columns_select(df)
    parameter_series = pd.DataFrame(
        {column: df[column] for column in df.columns if column not in objective_columns}
    )
    objective_series = pd.DataFrame(
        {column: df[column] for column in df.columns if column in objective_columns}
    )

    # 各変数の設定
    st.header("設定")

    st.subheader("目的変数の目標")
    objectives: Objectives = objective_config_input(objective_series)

    st.subheader("説明変数の制約")
    parameters = parameter_config_input(parameter_series)

    dataset = Dataset(
        parameters=parameters,
        objectives=objectives,
    )

    st.divider()

    # ベイズ最適化の開始
    st.header("実験候補を生成する")
    max_trials = st.number_input("生成数の上限", value=10, min_value=1)

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
    candidates = bayes_optimize(dataset, max_trials=max_trials)

    # 提案されたデータを表示する
    candidates = pd.DataFrame(candidates)
    st.dataframe(candidates)


def visualization_page(df: pd.DataFrame):
    """データ可視化ページ"""
    init_streamlit_comm()

    st.page_link(
        "https://docs.kanaries.net/ja/graphic-walker/data-viz/create-data-viz",
        label="データ可視化のマニュアル (外部サイト)",
        icon="📚",
    )

    renderer = get_pyg_renderer(df)
    renderer.render_explore()
