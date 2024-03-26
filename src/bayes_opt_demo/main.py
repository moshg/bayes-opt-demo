"""エントリーポイント"""

import os
from pathlib import Path

import streamlit as st

from bayes_opt_demo.ui import suggestion_page, upload_page, visualization_page


def main():
    # Streamlit Community Cloudとローカルで同じカレントディレクトリになるようsrcに移動する
    main_path = Path(__file__).resolve()
    src_dir = main_path.parent.parent
    os.chdir(src_dir)
    st.set_page_config(page_title="ベイズ最適化デモ", layout="wide", page_icon="📊")

    st.title("ベイズ最適化デモ")

    upload_tab, visualization_tab, suggestion_tab = st.tabs(
        ["アップロード", "データ可視化", "実験提案"]
    )

    # CSVの読み込み
    with upload_tab:
        df = upload_page()

    with visualization_tab:
        if df is None:
            st.info("可視化するデータをアップロードしてください")
        else:
            visualization_page(df)

    # 実験の提案
    with suggestion_tab:
        if df is None:
            st.info("学習データをアップロードしてください")
        else:
            suggestion_page(df)


main()
