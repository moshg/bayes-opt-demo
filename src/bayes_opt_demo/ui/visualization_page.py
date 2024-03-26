import pandas as pd
import streamlit as st
from pygwalker import FieldSpec
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm


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


@st.cache_data(ttl="1h", max_entries=1)
def get_pyg_renderer(df: pd.DataFrame) -> StreamlitRenderer:
    """PygWalkerのStreamlitRendererを取得する。

    ref. https://docs.kanaries.net/ja/pygwalker/use-pygwalker-with-streamlit
    """
    # デフォルトで集計しないように設定
    field_specs = {col: FieldSpec(analyticType="dimension") for col in df.columns}
    return StreamlitRenderer(df, field_specs=field_specs, debug=False)
