import numpy as np


def logit(p: float) -> float:
    """Compute the logit function, which is the
    inverse of the logistic function.

    Same as np.log(p/(1 - p)), but more stable for p near 1.

    input range: [0, 1]
    output_range: (-inf, inf)

    Note: this can be unstable for p near 0 or 1.
    """
    return float(np.log(p) - np.log1p(-p))


def logistic(x: float) -> float:
    """Compute the logistic sigmoid function (expit),
    which is the inverse of the logit function.

    input range: (-inf, inf)
    output_range: (0, 1)
    """
    # This is a standard and numerically stable implementation for moderate x.
    z = np.exp(-x)
    return float(1 / (1 + z))
