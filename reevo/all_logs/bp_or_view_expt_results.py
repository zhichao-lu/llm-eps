from __future__ import annotations

import os
import json
import pickle
from typing import List, Tuple
import sys

import numpy as np

sys.path.append('../')
from funsearch_impl import code_manipulation, evaluator_accelerate
from bin_packing_or_datasets import dataset_utils

import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray,
        func: callable
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
    # function = evaluator_accelerate.add_import_package_statement(function, 'numpy', 'np')
    # print(function)
    function_name = _extract_function_name(function)
    function, protect_div_func_name = evaluator_accelerate.replace_div_with_protected_div(function)

    if numba_accelerate:
        function = evaluator_accelerate.add_numba_decorator(
            program=function,
            function_name=[function_name, protect_div_func_name]
        )
    function = evaluator_accelerate.add_import_package_statement(function, 'numpy', 'np')
    function = evaluator_accelerate.add_np_random_seed_below_numpy_import(function, 2024)
    function = evaluator_accelerate.add_numpy_random_seed_to_func(function, function_name, 2024)
    # compile the program, and /maps the global func/var/class name to its address
    all_globals_namespace = {}
    # execute the program, map func/var/class to global namespace
    exec(function, all_globals_namespace)
    # get the pointer of 'function_to_evolve', which will be sent to AEL's evaluation module later
    function_to_run = all_globals_namespace[function_name]
    try:
        results = evaluate(inputs, function_to_run)
    except Exception as e:
        print(e)
        results = float('-inf')
    return results


def evaluate_top_k_on_all_or_data(log_file_path, k: List[int] | int = 1):
    if isinstance(k, int):
        k = [k]

    for _k in k:
        res = []
        funcs, _ = find_top_k_functions(log_file_path, _k)
        for i in [1, 2, 3, 4]:
            data = dataset_utils.load_bin_packing_or_dataset(or_name=i)
            optimal = dataset_utils._l1_bound_dataset(data)
            k_res = []
            for func in funcs:
                score = evaluate_func_in_str(func, data)
                exceed_bins = (np.abs(score) - optimal) / optimal * 100
                k_res.append(exceed_bins)
            res.append(min(k_res))
        print(f'Best-{_k}: {res}%')


def __evaluate_on_train_data(func):
    with open('../bin_packing_or_datasets/or_train.pkl', 'rb') as f:
        data = pickle.load(f)['or_train']
    optimal = dataset_utils._l1_bound_dataset(data)
    score = evaluate_func_in_str(func, data)
    exceed_bins = (np.abs(score) - optimal) / optimal * 100
    return exceed_bins


if __name__ == '__main__':
    log_file = 'bin_packing_or_codellama_run1'
    func, score = find_top_k_functions(log_file, k=1)
    # print(func[0])
    print(f'train exceed bins: {__evaluate_on_train_data(func[0])}')
    evaluate_top_k_on_all_or_data(log_file_path=log_file, k=[1, 10])

    log_file = 'bin_packing_or_codellama_run2'
    func, score = find_top_k_functions(log_file, k=1)
    # print(func[0])
    print(f'train exceed bins: {__evaluate_on_train_data(func[0])}')
    evaluate_top_k_on_all_or_data(log_file_path=log_file, k=[1, 10])

    log_file = 'bin_packing_or_codellama_run3'
    func, score = find_top_k_functions(log_file, k=1)
    # print(func[0])
    print(f'train exceed bins: {__evaluate_on_train_data(func[0])}')
    evaluate_top_k_on_all_or_data(log_file_path=log_file, k=[1, 10])
