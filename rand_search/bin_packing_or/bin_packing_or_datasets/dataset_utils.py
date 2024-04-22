# Following the conventions of FunSearch paper:
#
# We evaluate FunSearch on the well-known OR-Library bin packing bench marks [32], using the binpack1, binpack2,
# binpack3 and binpack4 datasets, each containing 20 bin packing instances,
# with 120, 250, 500, and 1_000 items, respectively. These instances were generated by sampling item sizes
# uniformly from the interval [20, 100]. For all datasets, the capacity of the bins is set to 150.
# To evolve a heuristic with FunSearch, we generated a training dataset of 20 instances each with 120 items
# sampled from [20, 100] (similarly to the binpack1 instances). To evaluate our heuristic during training,
# we also generated a validation dataset of 20 instances of 250 items sampled from [20, 100]
# (similarly to the binpack2 instances). We then select the best heuristics with respect to the validation dataset
# and test them on the binpack1 to binpack4 instances.

import random
import numpy as np
import pickle


def _generate_or_instance(num_items):
    or_data = []
    for _ in range(num_items):
        or_data.append(random.uniform(20, 100))
    return or_data


def _generate_weibull_dataset(num_instances, num_items):
    ret = {}
    for i in range(num_instances):
        key = f'instance_{i}'
        instance = _generate_or_instance(num_items)
        instance = {'capacity': 150, 'num_items': num_items, 'items': instance}
        ret[key] = instance
    return ret


def _generate_dataset_and_save(dataset_name: str, num_instances: int, num_items: int, save_file: str):
    dataset = _generate_weibull_dataset(num_instances, num_items)
    dataset = {dataset_name: dataset}
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)


def load_bin_packing_or_dataset(or_name: int):
    with open(f'bin_packing_{or_name}.txt', 'r') as file:
        lines = file.readlines()

    dataset = {}
    num_test_problems = int(lines[0].strip())
    current_line = 1

    for problem_index in range(1, num_test_problems + 1):
        problem_data = {}
        problem_id = lines[current_line].strip()
        current_line += 1

        bin_capacity, num_items, num_bins = map(int, lines[current_line].split())
        problem_data['capacity'] = bin_capacity
        problem_data['num_items'] = num_items
        # problem_data['num_bins'] = num_bins
        current_line += 1

        items = []
        for _ in range(num_items):
            item_size = int(lines[current_line].strip())
            items.append(item_size)
            current_line += 1

        problem_data['items'] = items
        dataset[problem_id] = problem_data

    return dataset


def _l1_bound(items: tuple[int, ...], capacity: int) -> float:
    """Computes L1 lower bound on OPT for bin packing.

    Args:
      items: Tuple of items to pack into bins.
      capacity: Capacity of bins.

    Returns:
      Lower bound on number of bins required to pack items.
    """
    return np.ceil(np.sum(items) / capacity)


def _l1_bound_dataset(instances: dict) -> float:
    """Computes the mean L1 lower bound across a dataset of bin packing instances.

    Args:
      instances: Dictionary containing a set of bin packing instances.

    Returns:
      Average L1 lower bound on number of bins required to pack items.
    """
    l1_bounds = []
    for name in instances:
        instance = instances[name]
        l1_bounds.append(_l1_bound(instance['items'], instance['capacity']))
    return np.mean(l1_bounds)


def cal_bin_paking_bound_dataset():
    all_or_datset = []
    for or_name in [1, 2, 3, 4]:
        or_data = load_bin_packing_or_dataset(or_name)
        all_or_datset.append(_l1_bound_dataset(or_data))
    print(all_or_datset)


if __name__ == '__main__':
    # file_train = 'or_train.pkl'
    # _generate_dataset_and_save(dataset_name='or_train', num_instances=20, num_items=250, save_file=file_train)
    with open('../or_train.pkl', 'rb') as f:
        data = pickle.load(f)
        key = list(data.keys())[0]
        print(_l1_bound_dataset(data[key]))
    # cal_bin_paking_bound_dataset()
