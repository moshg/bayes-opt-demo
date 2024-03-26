import pandas as pd
import streamlit as st
from bayes_opt_demo.example_data import debug_df, generate_hartmann6_sobol


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
