import json
import os

def get_valid_invalid(path):
    valid = 0
    invalid = 0
    for p in os.listdir(path):
        with open(os.path.join(path, p), 'r') as f:
            data = json.load(f)
        if data['score'] is not None:
            valid += 1
        else:
            invalid += 1
    return valid, invalid


for t in ['admissible_set_15_10_codellama_run', 'bin_packing_or_codellama_run', 'bin_packing_weibull_codellama_run', 'gls_tsp_codellama_run']:
    for r in [1, 2, 3]:
        path = t + str(r)
        valid, invalid = get_valid_invalid(path)
        orders = []
        for file in os.listdir(path):
            orders.append(int(file.split('.')[0].split('_')[1]))

        print(f'{path}: {len(list(os.listdir(path)))} valid: {valid} invalid: {invalid} orders: {max(orders)}')


