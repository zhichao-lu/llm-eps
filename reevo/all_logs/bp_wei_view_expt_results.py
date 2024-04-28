import json
import os
import pickle
import sys
from typing import List, Tuple
from tqdm.auto import tqdm

sys.path.append('../')
from bin_packing_weibull_datasets import dataset_utils
from funsearch_impl import evaluator_accelerate
from funsearch_impl import code_manipulation

import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray, func: callable
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = func(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


def evaluate(instances: dict, func: callable) -> float:
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
        _, bins_packed = online_binpack(items, bins, func)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    return -np.mean(num_bins)


def find_top_k_functions(log_file_path, k=1) -> Tuple[List[str], List[int]]:
    samples_path = log_file_path
    all_scores = {}
    for sample_file in os.listdir(samples_path):
        sample_file = os.path.join(samples_path, sample_file)
        with open(sample_file, 'r') as f:
            sample_json = json.load(f)
            f.close()
        all_scores[sample_file] = sample_json['score']
    sorted_keys = sorted(all_scores, key=lambda x: all_scores[x] if all_scores[x] is not None else float('inf'))
    top_k_func_files = sorted_keys[:k]
    top_k_func = []
    top_k_score = []
    for func_file in top_k_func_files:
        # sample_file = os.path.join(samples_path, func_file)
        with open(func_file, 'r') as f:
            sample_json = json.load(f)
            f.close()
        top_k_func.append(sample_json['function'])
        top_k_score.append(sample_json['score'])
    return top_k_func, top_k_score


def _extract_function_name(function: str):
    func = code_manipulation.text_to_function(function)
    return func.name


def evaluate_func_in_str(function: str, inputs, numba_accelerate: bool = True) -> float:
    # np.random.seed(2024)
    function = evaluator_accelerate.add_import_package_statement(function, 'numpy', 'np')
    # print(function)
    function_name = _extract_function_name(function)
    if numba_accelerate:
        function = evaluator_accelerate.add_numba_decorator(
            program=function,
            function_name=function_name
        )
    function, _ = evaluator_accelerate.replace_div_with_protected_div(function, numba_accelerate=numba_accelerate)
    function = evaluator_accelerate.add_numpy_random_seed_to_func(function, function_name)
    # print(function);sys.exit(0)
    # program = '\n'.join([specification, function])

    # set numpy random seed
    # program = evaluator_accelerate.add_np_random_seed_below_numpy_import(, seed=2024)

    try:
        # compile the program, and maps the global func/var/class name to its address
        all_globals_namespace = {}
        # execute the program, map func/var/class to global namespace
        exec(function, all_globals_namespace)
        # get the pointer of 'function_to_evolve', which will be sent to AEL's evaluation module later
        func = all_globals_namespace[function_name]
        results = evaluate(inputs, func)
        return results
    except Exception as e:
        print(e)
        return float('-inf')


def evaluate_top_k_func(log_file_path, inputs, k) -> list:
    """Returns the percentage of exceed bins"""
    optimal = dataset_utils._l1_bound_dataset(inputs)
    top_k_func, _ = find_top_k_functions(log_file_path, k)
    results = []
    for func in tqdm(top_k_func, desc='Evaluating Functions', leave=True):
        score = evaluate_func_in_str(func, inputs=inputs)
        excess_bins = (np.abs(score) - optimal) / optimal * 100
        results.append(excess_bins)
    return results


def _get_all_test_inputs():
    np.random.seed(2024)
    inputs0 = dataset_utils.generate_weibull_dataset(5, 5_000)
    inputs1 = dataset_utils.generate_weibull_dataset(5, 10_000)
    inputs2 = dataset_utils.generate_weibull_dataset(1, 100_000)
    return [inputs0, inputs1, inputs2]


def evaluate_all(log_file_path):
    with open('../bin_packing_weibull_datasets/weibull_train.pkl', 'rb') as f:
        train_data = pickle.load(f)['weibull_5k_train']
    _, train_score = find_top_k_functions(log_file_path, k=1)
    train_bounds = dataset_utils._l1_bound_dataset(train_data)
    print(train_score)
    print(train_bounds)
    train_exceeds = (np.abs(train_score[0]) - train_bounds) / train_bounds * 100
    print(f'train exceeds: {train_exceeds}%')

    all_inputs = _get_all_test_inputs()
    for name, inputs in zip(['5k', '10k', '100k'], all_inputs):
        res = evaluate_top_k_func(log_file_path, inputs, 10)  # TOP 10 !!!!!!!!
        print(f'weibull {name}: top-1: {res[0]}, top-5: {min(res[:5])}, top-10: {min(res)}')
    print('---------------------------------')


def evaluate_func_str_all(func: str):
    all_inputs = _get_all_test_inputs()
    res = []
    for inputs in all_inputs:
        res_ = evaluate_func_in_str(func, inputs, True)
        print(res_)

    print(f'Test Weibull 5k: {res[0]}, Test Weibull 10k: {res[1]}, Test Weibull 100k: {res[2]}')


if __name__ == '__main__':
    log_file_path = 'bin_packing_weibull_codellama_run1'
    evaluate_all(log_file_path)
    log_file_path = 'bin_packing_weibull_codellama_run1'
    evaluate_all(log_file_path)
    print('=================================================================================')
    # log_file_path = 'bin_packing_weibull_codellama_run2'
    # evaluate_all(log_file_path)
    # log_file_path = 'bin_packing_weibull_codellama_run2'
    # evaluate_all(log_file_path)
    # print('=================================================================================')
    # log_file_path = 'bin_packing_weibull_codellama_run3'
    # evaluate_all(log_file_path)
    # log_file_path = 'bin_packing_weibull_codellama_run3'
    # evaluate_all(log_file_path)
    # print('=================================================================================')
