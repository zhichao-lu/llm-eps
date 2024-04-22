import numpy as np

best_func = ''
best_score = float('-inf')
best_Func = ...


def update_best(func, score):
    global best_score, best_func
    if score is None:
        return
    if score > best_score:
        best_score = score
        best_func = func
