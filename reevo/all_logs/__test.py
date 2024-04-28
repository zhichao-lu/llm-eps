import json
import os

# path = 'bin_packing_weibull_codellama_run2'
# scores = []
# for p in os.listdir(path):
#     jfile = os.path.join(path, p)
#     with open(jfile, 'r') as f:
#         data = json.load(f)
#         score = data['score']
#         if score:
#             scores.append(score)
#         if score and score < 500:
#             print(p)
#         f.close()
#
# print(min(scores), ' ', max(scores))

pp = ['bin_packing_weibull_codellama_run2/samples_25.json',
      'bin_packing_weibull_codellama_run2/samples_67.json',
      'bin_packing_weibull_codellama_run2/samples_42.json'
      ]
for p in pp:
    with open(p, 'r') as f:
        data = json.load(f)
        f.close()
    data['score'] = None
    with open(p, 'w') as f:
        json.dump(data, f)
