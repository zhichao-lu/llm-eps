import json
import multiprocessing
import os.path
import pickle
import sys
from argparse import ArgumentParser

sys.path.append('../../../')
from funsearch_impl import evaluator_accelerate, code_manipulation

specification = r'''
import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray
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


def evaluate(instances: dict) -> float:
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
        _, bins_packed = online_binpack(items, bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    return -np.mean(num_bins)
'''


def _extract_function_name(function: str):
    func = code_manipulation.text_to_function(function)
    return func.name


def _evaluate_func_in_str(function: str, inputs, numba_accelerate, result_queue: multiprocessing.Queue):
    # np.random.seed(2024)
    # function = evaluator_accelerate.add_import_package_statement(function, 'numpy', 'np')
    # print(function)
    try:
        function_name = _extract_function_name(function)
        if numba_accelerate:
            function = evaluator_accelerate.add_numba_decorator(
                program=function,
                function_name=function_name
            )
        program = '\n'.join([specification, function])

        # set numpy random seed
        program = evaluator_accelerate.add_np_random_seed_below_numpy_import(program, seed=2024)

        try:
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(program, all_globals_namespace)
            # get the pointer of 'function_to_evolve', which will be sent to AEL's evaluation module later
            function_to_run = all_globals_namespace['evaluate']
            results = function_to_run(inputs)
            result_queue.put(results)
        except:
            result_queue.put(None)
    except:
        result_queue.put(None)


def evaluate_func_in_str(function: str, inputs, numba_accelerate: bool = True):
    try:
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_evaluate_func_in_str,
            args=(function, inputs, numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=5)
        if process.is_alive():
            # if the process is not finished in time, we consider the program illegal
            process.terminate()
            process.join()
            results = None
        else:
            if not result_queue.empty():
                results = result_queue.get_nowait()
            else:
                results = None

        return results

    except:
        return None


with open('../../bin_packing_or_datasets/or_train.pkl', 'rb') as f:
    data = pickle.load(f)['or_train']

from tqdm.auto import tqdm


def update_score(path):
    path = os.path.join(path, 'samples')
    for jsonf in tqdm(os.listdir(path), desc='Re-Evaluating'):
        jsonf = os.path.join(path, jsonf)
        with open(jsonf, 'r') as f:
            sample = json.load(f)
            f.close()

        if sample['score'] is None:
            continue

        func = sample['function']
        score = evaluate_func_in_str(func, data)
        sample['score'] = score
        # print(score)
        with open(jsonf, 'w') as f:
            json.dump(sample, f)
            f.close()


parser = ArgumentParser()
parser.add_argument('--run', type=int, default=4)
args = parser.parse_args()

if __name__ == '__main__':
    update_score(f'rand_search_bin_packing_or_run{args.run}')
