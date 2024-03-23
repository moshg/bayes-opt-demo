"""エントリーポイント"""

import streamlit as st

from bayes_opt_demo.ui.pages import suggestion_page, upload_page


def main():
    st.title("ベイズ最適化デモ")

    upload_tab, suggestion_tab = st.tabs(["アップロード", "実験提案"])
    # CSVの読み込み
    df = upload_page()

    # 物性の提案 (CSVが読み込まれてから表示する)
    if df is not None:
        suggestion_page(df)


main()
