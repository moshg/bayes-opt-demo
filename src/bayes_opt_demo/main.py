"""エントリーポイント"""

import streamlit as st

from bayes_opt_demo.ui.pages import suggestion_page, upload_page


def main():
    st.title("ベイズ最適化デモ")

    upload_tab, suggestion_tab = st.tabs(["アップロード", "実験提案"])

    # CSVの読み込み
    with upload_tab:
        df = upload_page()

    # 実験の提案
    with suggestion_tab:
        if df is None:
            st.info("学習データをアップロードしてください")
        else:
            suggestion_page(df)


main()
