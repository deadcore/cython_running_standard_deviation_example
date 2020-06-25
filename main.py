import argparse

import numpy as np
from loguru import logger

from cython_example.operations import op_code_to_function


def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', required=True, type=int)
    parser.add_argument('--max', required=True, type=int)
    parser.add_argument('--op', required=True, type=str)
    args, _ = parser.parse_known_args()

    op = op_code_to_function(args.op)

    arr = np.random.randint(1, args.max, size=args.observations)
    logger.info('arr: {}', arr)

    (result, runtime) = op(arr)

    logger.info('Result: {}', result)


if __name__ == '__main__':
    main()
