import pandas as pd
from ax.models.random.sobol import SobolGenerator
from ax.utils.measurement.synthetic_functions import hartmann6
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


@cache_data(max_entries=1)
def generate_hartmann6_sobol(
    n: int, decimals: int = 2, seed: int | None = None
) -> pd.DataFrame:
    """Hartmann6関数のSobol列によるサンプルを生成する。

    Args:
        n: サンプル数
        precision: 丸める桁数
    """

    sobol = SobolGenerator(seed=seed)
    # 6次元のSobol列を生成
    samples, _ = sobol.gen(
        n,
        bounds=[(0.0, 1.0) for _ in range(6)],
        rounding_func=lambda x: x.round(decimals),
    )

    ys = hartmann6(samples)

    # 入力変数名をx1, x2, ..., x6とする
    df = pd.DataFrame(
        {f"x{i + 1}": x for i, x in enumerate(sample)} for sample in samples
    )
    df["hartmann6"] = ys

    return df
