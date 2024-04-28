import ast
import json
import os
import sys
from typing import Callable, List, Tuple

from tqdm.auto import tqdm

sys.path.append('../')
from funsearch_impl import evaluator_accelerate
from funsearch_impl import code_manipulation
from tsp_eval_helper import ael_evaluation_test_tsplib

import numpy as np


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

def comment_print_invokes(program: str) -> str:
    """The 'print' function in sampled function may generate useless info to the log file.
    This function comments all print statement.
    This function may not consider all situations such as:
    -----------------------------------------------------------------------------------------
    def f(a: int):
        if a == 0:
            print('a is zero')
    -----------------------------------------------------------------------------------------
    In the above code, simply comment the print statement may cause some problems.
    """
    tree = ast.parse(program)
    program_lines = program.splitlines()
    print_call_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
            print_call_lines += [i for i in range(node.lineno, node.end_lineno + 1)]

    for line in print_call_lines:
        program_lines[line - 1] = f'# {program_lines[line - 1]}'
    return '\n'.join(program_lines)


def evaluate_func_in_str(function: str, numba_accelerate: bool = True):
    np.random.seed(2024)
    function = evaluator_accelerate.add_import_package_statement(function, 'numpy', 'np')
    # print(function)
    function_name = _extract_function_name(function)
    if numba_accelerate:
        function = evaluator_accelerate.add_numba_decorator(
            program=function,
            function_name=function_name
        )
    function = evaluator_accelerate.add_numpy_random_seed_to_func(function, function_name, 2024)
    function = comment_print_invokes(function)
    # compile the program, and maps the global func/var/class name to its address
    all_globals_namespace = {}
    # execute the program, map func/var/class to global namespace
    exec(function, all_globals_namespace)
    # get the pointer of 'function_to_evolve', which will be sent to AEL's evaluation module later
    function_to_evolve_pointer = all_globals_namespace[function_name]
    evaluator = ael_evaluation_test_tsplib.Evaluation()
    # do evaluate
    results = evaluator.evaluate(heuristic=function_to_evolve_pointer)
    return results


# def evaluate_func(function: Callable, instance_path: str = None):
#     evaluator = ael_evaluation.Evaluation(instance_path=instance_path)
#     results = evaluator.evaluate(heuristic_func=function)
#     return results


def evaluate_top_k(log_path, k=1):
    res = []
    top_k_func, _ = find_top_k_functions(log_path, k)
    for func in top_k_func:
        # print(func)
        score = evaluate_func_in_str(func, True)
        res.append(score)
    return min(res)


if __name__ == '__main__':
    logfile = 'gls_tsp_gpt35_run1'
    _, train_score = find_top_k_functions(logfile)
    print(f'Train score: {train_score}; ')

    top10, _ = find_top_k_functions(logfile, k=10)
    scores = []
    for func in tqdm(top10, desc='Evaluate TSP'):
        res_11_times = []
        for i in range(11):
            res = evaluate_func_in_str(func)
            res_11_times.append(res)
        res = np.mean(res_11_times)
        scores.append(res)

    print(f'top1: {scores[0]}')
    print(f'top5: {min(scores[:5])}')
    print(f'top10: {min(scores)}')
    # ================================================
    logfile = 'gls_tsp_gpt35_run2'
    _, train_score = find_top_k_functions(logfile)
    print(f'Train score: {train_score}; ')

    top10, _ = find_top_k_functions(logfile, k=10)
    scores = []
    for func in tqdm(top10, desc='Evaluate TSP'):
        res_11_times = []
        for i in range(11):
            res = evaluate_func_in_str(func)
            res_11_times.append(res)
        res = np.mean(res_11_times)
        scores.append(res)

    print(f'top1: {scores[0]}')
    print(f'top5: {min(scores[:5])}')
    print(f'top10: {min(scores)}')
    # ================================================
    logfile = 'gls_tsp_gpt35_run3'
    _, train_score = find_top_k_functions(logfile)
    print(f'Train score: {train_score}; ')

    top10, _ = find_top_k_functions(logfile, k=10)
    scores = []
    for func in tqdm(top10, desc='Evaluate TSP'):
        res_11_times = []
        for i in range(11):
            res = evaluate_func_in_str(func)
            res_11_times.append(res)
        res = np.mean(res_11_times)
        scores.append(res)

    print(f'top1: {scores[0]}')
    print(f'top5: {min(scores[:5])}')
    print(f'top10: {min(scores)}')
    # ================================================

