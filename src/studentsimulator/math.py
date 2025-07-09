import numpy as np


def logit(p: float) -> float:
    """Compute the logit function, which is the
    inverse of the logistic function.

    Same as np.log(p/(1 - p)), but more stable for p near 1.

    input range: [0, 1]
    output_range: (-inf, inf)
    """
    # This is a reasonable implementation for most practical purposes.
    # Clipping to [0.01, 0.99] avoids log(0) and log(1), which would be -inf/inf.
    p = np.clip(p, 0.01, 0.99)
    return np.log(p) - np.log1p(
        -p
    )  # np.log1p(-p) is log(1-p), more stable for p near 0


def logistic(x: float) -> float:
    """Compute the logistic sigmoid function (expit),
    which is the inverse of the logit function.

    input range: (-inf, inf)
    output_range: (0, 1)
    """
    # This is a standard and numerically stable implementation for moderate x.
    z = np.exp(-x)
    return 1 / (1 + z)
