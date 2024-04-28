import multiprocessing
import os.path
import sys
from typing import Any, Callable, Tuple

sys.path.append('../')
from my_task import _evaluator_accelerate

import numpy as np
import pickle


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray, priority: Callable
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


def _evaluate(instances: dict, priority: Callable) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []
    # Perform online binpacking for each instance.
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items, bins, priority)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    return np.mean(num_bins)


_data_path = os.path.join(os.path.dirname(__file__), 'weibull_train.pkl')
with open(_data_path, 'rb') as f:
    data = pickle.load(f)['weibull_5k_train']


def evaluate(func: Callable):
    return _evaluate(data, func)


test_code = '''def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    penalty = np.arange(len(bins), 0, -1)
    scores = bins / (bins - item) - penalty
    max_capacity_bins = np.where(bins == bins.max())[0]
    for idx in max_capacity_bins:
        scores[idx] = -np.inf
    return scores'''


class Sandbox:
    def __init__(self, verbose=False, numba_accelerate=True, timeout=10):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate
        self._timeout = timeout

    def run(self, function_to_evolve: str) -> Tuple[Any, bool]:
        try:
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=self._compile_and_run_function,
                args=(function_to_evolve, self._numba_accelerate, result_queue)
            )
            process.start()
            process.join(timeout=self._timeout)
            if process.is_alive():
                # if the process is not finished in time, we consider the program illegal
                process.terminate()
                process.join()
                results = None, False
            else:
                if not result_queue.empty():
                    results = result_queue.get_nowait()
                else:
                    results = None, False

            return results
        except Exception as e:
            # print(e)
            return None, False

    def _compile_and_run_function(self, function_to_evolve: str, _numba_accelerate, result_queue):
        try:
            func_name = _evaluator_accelerate._extract_function_name(function_to_evolve)
            function_to_evolve = _evaluator_accelerate.add_import_package_statement(function_to_evolve, 'numpy', 'np')
            if _numba_accelerate:
                function_to_evolve = _evaluator_accelerate.add_numba_decorator(
                    program=function_to_evolve,
                    function_name=[func_name]
                )
            # print(function_to_evolve)
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(function_to_evolve, all_globals_namespace)
            # get the pointer of 'function_to_run'
            func_pointer = all_globals_namespace[func_name]
            # return the execution results
            results = evaluate(func_pointer)
            if results is not None:
                if not isinstance(results, (int, float)):
                    results = (None, False)
                else:
                    # negation because our optimization objective is bigger, the better
                    results = (results, True)  # convert to FunSearch result format
            else:
                results = (None, False)
            result_queue.put(results)
        except Exception as e:
            results = (None, False)
            result_queue.put(results)


if __name__ == '__main__':
    sandbox = Sandbox()
    res = sandbox.run(test_code)
    print(res)
