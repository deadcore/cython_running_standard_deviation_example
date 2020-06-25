from libc.math cimport pow, sqrt

import numpy as np
cimport numpy as np

__all__ = ['calculate_running_std_dev']

cpdef calculate_running_std_dev(arr):
    out = np.zeros(arr.shape[0], dtype=np.double)
    rolling_sum = arr[0]
    squares_sum = pow(rolling_sum, 2)
    n = 1
    out[0] = np.NaN
    for value in arr[1:]:
        idx = n
        n = n + 1
        # ∑(x)
        rolling_sum = rolling_sum + value
        # ∑(x^2)
        squares_sum = squares_sum + pow(value, 2)
        # √(∑(x - μ)^2 / (n - 1))
        out[idx] = sqrt((squares_sum - (pow(rolling_sum, 2) / n)) / (n - 1))
    return out


