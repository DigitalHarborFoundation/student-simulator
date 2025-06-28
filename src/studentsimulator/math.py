import numpy as np


def logit(p: float | np.ndarray) -> float | np.ndarray:
    """Compute the logit function, which is the
    inverse of the logistic function.

    Same as np.log(p/(1 - p)), but more stable for p near 1.

    input range: [0, 1]
    output_range: (-inf, inf)
    """
    p = np.asarray(p)
    p = np.clip(p, 0.01, 0.99)  # avoid log(0) and log(1)
    result = np.log(p) - np.log1p(-p)  # stable for p near 1
    # return as a single float if input is a single float
    if result.ndim == 0:
        result = result.item()  # convert to scalar if it's a single value
    return result


def logistic(x: float | np.ndarray) -> float | np.ndarray:
    """Compute the logistic sigmoid function (expit),
    which is the inverse of the logit function.

    input range: (-inf, inf)
    output_range: (0, 1)
    """
    x = np.asarray(x)
    z = np.exp(-x)
    result = 1 / (1 + z)  # stable for moderate x
    # return as a single float if input is a single float
    if result.ndim == 0:
        result = result.item()
    return result
