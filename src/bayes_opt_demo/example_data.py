import pandas as pd
from streamlit import cache_data

debug_df = pd.DataFrame(
    {
        "x1": [1.0, 2.0, 3.0],
        "x2": ["foo", "bar", None],
        "x3": [2.0, 1.0, None],
        "y": [3.0, 2.0, 3.5],
    }
)
"""デバッグ用のDataFrame"""


@cache_data(ttl="1h", max_entries=1)
def get_hartmann6_bayes_opt() -> pd.DataFrame:
    """Hartmann6関数のベイズ最適化の結果を取得する。"""
    return pd.read_csv("bayes_opt_demo/data/hartmann6_bayes_opt.csv")
