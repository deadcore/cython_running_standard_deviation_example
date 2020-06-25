import argparse
import concurrent.futures

import numpy as np
from loguru import logger

from cython_example.operations import op_code_to_function


def job(column_idx, arr, op_code):
    logger.info('Starting to calculate on column: {}', column_idx)

    op = op_code_to_function(op_code)

    (result, runtime) = op(arr[:, column_idx])

    return column_idx, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', required=True, type=int)
    parser.add_argument('--max', required=True, type=int)
    parser.add_argument('--op', required=True, type=str)
    parser.add_argument('--dimensions', required=True, type=int)
    args, _ = parser.parse_known_args()

    logger.info('Passed arguments')

    arr = np.random.randint(1, args.max, size=(args.observations, args.dimensions))
    results = np.empty(shape=(args.observations, args.dimensions))

    logger.info('Created test data')

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_results = [executor.submit(job, column_idx, arr, args.op) for column_idx in range(0, args.dimensions)]
        concurrent.futures.wait(future_results)
        for future in future_results:
            (result_column_idx, result) = future.result()
            results[:, result_column_idx] = result

    logger.debug("results: {}", results)


if __name__ == '__main__':
    main()
