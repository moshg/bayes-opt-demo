"""エントリーポイント"""

import pandas as pd
import streamlit as st

from bayes_opt_demo.dataset import (
    Dataset,
    Objectives,
)
from bayes_opt_demo.optimization import bayes_optimize
from bayes_opt_demo.ui import (
    debug_df,
    input_objective_config,
    input_parameter_config,
    select_objective_columns,
    upload_csv,
)


def main():
    st.title("ベイズ最適化デモ")

    # CSVの読み込み
    st.header("学習データのアップロード")
    df = upload_csv()

    data_src = st.radio(
        "使用データ",
        options=["アップロードされたCSV", "デバッグデータ"],
        horizontal=True,
    )
    if data_src == "デバッグデータ":
        df = debug_df

    # データが入力されるまで何も表示しない
    if df is None:
        return

    # 読み込んだCSVの表示
    st.subheader("アップロードされたデータ")
    st.dataframe(df)

    # 目的変数の選択
    st.header("目的変数の選択")
    objective_columns = select_objective_columns(df)
    parameter_series = pd.DataFrame(
        {column: df[column] for column in df.columns if column not in objective_columns}
    )
    objective_series = pd.DataFrame(
        {column: df[column] for column in df.columns if column in objective_columns}
    )

    # 各変数の設定
    st.header("設定")

    st.subheader("目的変数の目標")
    objectives: Objectives = input_objective_config(objective_series)

    st.subheader("説明変数の制約")
    parameters = input_parameter_config(parameter_series)

    dataset = Dataset(
        parameters=parameters,
        objectives=objectives,
    )

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
    candidates = bayes_optimize(dataset, max_trials=max_trials)

    # 提案されたデータを表示する
    candidates = pd.DataFrame(candidates)
    st.dataframe(candidates)


main()
