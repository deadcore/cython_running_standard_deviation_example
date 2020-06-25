import time
from math import sqrt

import numpy as np
from loguru import logger

import cython_example.native as native


def timerfunc(out_fn=print):
    """
    A timer decorator
    """

    def decorator(func):
        def function_timer(*args, **kwargs):
            """
            A nested function for timing other functions
            """
            start = time.time_ns()
            value = func(*args, **kwargs)
            end = time.time_ns()
            runtime = end - start
            out_fn(f"The runtime for [{func.__name__}] took {runtime:,}ns to complete")
            return value, runtime

        return function_timer

    return decorator


def time_native_function(fn):
    @timerfunc(logger.debug)
    def native_wrapper(arr):
        result = fn(arr)
        return np.asarray(result)

    return native_wrapper


@timerfunc(logger.debug)
def python_cumulative_standard_deviation(vector):
    out = np.zeros(vector.shape[0], dtype=np.double)
    rolling_sum = vector[0]
    squares_sum = pow(rolling_sum, 2)
    n = 1
    out[0] = np.NaN
    for value in vector[1:]:
        idx = n
        n = n + 1
        # ∑(x)
        rolling_sum = rolling_sum + value
        # ∑(x^2)
        squares_sum = squares_sum + pow(value, 2)
        # √(∑(x - μ)^2 / (n - 1))
        out[idx] = sqrt((squares_sum - (pow(rolling_sum, 2) / n)) / (n - 1))
    return out


def op_code_to_function(op):
    ops = {
        'python': python_cumulative_standard_deviation,
        'native1': time_native_function(native.sd1),
        'native2': time_native_function(native.sd2),
        'native3': time_native_function(native.sd3),
        'native4': time_native_function(native.sd4),
        'native5': time_native_function(native.sd5),
    }

    if op not in ops:
        raise ValueError(f'Unknown operation: [{op}]. Known operations are {list(ops.keys())}')

    return ops[op]
