import numpy as np


def sigmoid(z):
    """Compute the sigmoid of z.

    The sigmoid function is defined as 1 / (1 + exp(-z)).
    It is commonly used in machine learning and statistics
    as an activation function or to map any real value into
    the range (0, 1).

    Args:
        z (float or np.ndarray): The input value or array of values.

    Returns:
        float or np.ndarray: The sigmoid of the input value(s).
    """
    return 1 / (1 + np.exp(-z))
