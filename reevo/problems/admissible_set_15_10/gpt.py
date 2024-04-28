import numpy as np
def heuristics_v2(el: tuple[int, ...], n: int, w: int) -> float:
    # Priority is based on the number of elements (w) and the deviation of the sum of the elements from the target average (w)
    penalty1 = (n - len(el)) ** 2 / (2 * (n - 1) * n)
    penalty2 = (sum(el) - w) ** 2 / (2 * (w ** 2))
    penalty3 = (sum(el) / w) ** 2 / n
    return -1 * penalty1 - 0.5 * penalty2 + 0.7 * penalty3
