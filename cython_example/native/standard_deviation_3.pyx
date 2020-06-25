from libc.math cimport pow, sqrt

import numpy as np
cimport numpy as np

__all__ = ['calculate_running_std_dev']

def calculate_running_std_dev(long[:] vector):
    return calculate_running_std_dev_c(vector)

cdef double[:] calculate_running_std_dev_c(long[:] vector):
    cdef int length = vector.shape[0]
    cdef double[:] out = np.zeros(length, dtype=np.double)
    cdef double rolling_sum = vector[0]
    cdef double squares_sum = pow(rolling_sum, 2)
    cdef int n = 1
    cdef long value
    out[0] = np.nan

    for i in range(1, length):
        n = n + 1
        value = vector[i]
        # ∑(x)
        rolling_sum = rolling_sum + value
        # ∑(x^2)
        squares_sum = squares_sum + pow(value, 2)
        # √(∑(x - μ)^2 / (n - 1))
        out[i] = sqrt((squares_sum - (pow(rolling_sum, 2) / n) ) / (n-1))

    return out

