import json
import os
import pickle

path = 'gls_tsp_codellama_run1'
book = set()

for p in os.listdir(path):
    # samples_9991.json
    order = int(p.split('_')[1].split('.')[0])
    book.add(order)

for i in range(2, 10_000):
    if i not in book:
        copy_path = f'samples_{i - 1}.json'
        copy_path = os.path.join(path, copy_path)
        jpath = f'samples_{i}.json'
        jpath = os.path.join(path, jpath)
        with open(copy_path, 'r') as f:
            data = json.load(f)
            f.close()
        with open(jpath, 'w') as f:
            json.dump(data, f)
            f.close()
