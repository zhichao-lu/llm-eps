import os
import json
from typing import Tuple

import numpy as np


def get_best_func(log_path) -> Tuple[str, float]:
    log_path = os.path.join(log_path, 'samples')
    best_score = float('-inf')
    best_specification = ...
    for dir in os.listdir(log_path):
        with open(os.path.join(log_path, dir), 'r') as f:
            sample_dict = json.load(f)
            if sample_dict['score'] is not None and sample_dict['score'] > best_score:
                best_score = sample_dict['score']
                best_specification = sample_dict['function']
    return best_specification, best_score


def get_last_samples_nums(log_path) -> int:
    log_path = os.path.join(log_path, 'samples')
    max_samples = -1
    for dir in os.listdir(log_path):
        order = int(dir.split('.')[0].split('_')[1])
        if order > max_samples:
            max_samples = order
    return max_samples


def get_sample_score_cur_best(log_file_path, num_scores: int):
    path = log_file_path
    scores = []
    all_samples = [int(i.split('.')[0].split('_')[1]) for i in os.listdir(path)]
    all_samples = sorted(all_samples)
    for i in all_samples:
        json_file_path = os.path.join(path, f'samples_{i}.json')
        with open(json_file_path, 'r') as json_file:
            json_file = json.load(json_file)

        scores.append(json_file['score'])
    scores_ = []
    for s in scores:
        if s is not None:
            scores_.append(s)
        else:
            scores_.append(float('inf'))
    scores = scores_
    scores = np.array(scores)
    scores = np.abs(scores)
    min = scores[0]
    for i in range(1, len(scores)):
        if scores[i] is None:
            scores[i] = min
        else:
            if scores[i] < min:
                min = scores[i]
            else:
                scores[i] = min

    if len(scores) > num_scores:
        scores = scores[:num_scores]
    elif len(scores) < num_scores:
        scores = list(scores)
        scores += [min] * (num_scores - len(scores))
    return scores


if __name__ == '__main__':
    log_path = 'admissible_set_15_10_codellama_run1'
    # ----------------------------------------------------------
    score = get_sample_score_cur_best(log_path, 10000)
    # print(best, '\n\n')
    print(score[-1])
    # ----------------------------------------------------------
    log_path = 'admissible_set_15_10_codellama_run2'
    # ----------------------------------------------------------
    score = get_sample_score_cur_best(log_path, 10000)
    # print(best, '\n\n')
    print(score[-1])
    # ----------------------------------------------------------
    log_path = 'admissible_set_15_10_codellama_run3'
    # ----------------------------------------------------------
    score = get_sample_score_cur_best(log_path, 10000)
    # print(best, '\n\n')
    print(score[-1])
    # ----------------------------------------------------------

    # last_samples = get_last_samples_nums(log_path)
    # print(last_samples)