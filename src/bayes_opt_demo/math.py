import math


def round_range(lower, higher):
    """2つの数値の範囲を丸める。

    >>> round_range(0.1, 1.8)
    (0, 2)
    >>> round_range(11, 28)
    (10, 30)
    >>> round_range(0.31, 0.58)
    (0.3, 0.6)
    """

    if lower > higher:
        lower, higher = higher, lower

    # xとyの差を計算
    diff = abs(lower - higher)

    # 差に応じて桁数を決定
    if diff < 1:
        # 小数点以下の桁で処理
        decimal_places = abs(math.floor(math.log10(diff)))
        factor = 10 ** (-decimal_places)
    elif diff < 10:
        # 1の位まで
        factor = 1
    else:
        # 10の位まで、またはそれ以上の場合は、差の桁数を基にする
        factor = 10 ** (math.floor(math.log10(diff)))

    # xを切り捨て、yを切り上げ
    x_rounded = math.floor(lower / factor) * factor
    y_rounded = math.ceil(higher / factor) * factor

    # 小数点以下の桁数を調整
    if factor < 1:
        decimal_places = -int(math.log10(factor))
        x_rounded = round(x_rounded, decimal_places)
        y_rounded = round(y_rounded, decimal_places)
    elif factor == 1:
        x_rounded = int(x_rounded)
        y_rounded = int(y_rounded)

    return (x_rounded, y_rounded)
