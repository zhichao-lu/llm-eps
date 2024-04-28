import random
import sys
from typing import Callable

import numpy as np
import time
from joblib import Parallel, delayed
import os

sys.path.append('../')
from tsp_eval_helper import readTSPLib
from tsp_eval_helper.gls_run import solve_instance_tsplib
np.random.seed(2024)

class Evaluation():
    def __init__(self) -> None:
        self.time_limit = 60  # maximum 10 seconds for each instance
        self.ite_max = 1000  # maximum number of local searchs in GLS for each instance
        self.perturbation_moves = 1  # movers of each edge in each perturbation
        path = os.path.dirname(os.path.abspath(__file__))
        # self.instance_path = path+'/instance/TSPAEL64.pkl' #,instances=None,instances_name=None,instances_scale=None
        self.debug_mode = False

        # self.coords,self.instances,self.opt_costs = readTSPLib.read_instance_all(self.instance_path)
        m = 50  # neighborhood size
        instances_path = "instance/tsp_lib_200"
        coords = []
        instances = []
        neighbors = []
        instances_scale = []
        instances_name = []
        # RZ: modify path
        # ----------------------------------------------------
        path = os.path.dirname(os.path.abspath(__file__))
        instances_path = os.path.join(path, instances_path)
        # ----------------------------------------------------

        file_names = os.listdir(instances_path)
        for filename in file_names:
            coordinates, distance_matrix, neighbor_matrix, scale = readTSPLib.readinstance(
                instances_path + "/" + filename, m)
            coords.append(coordinates)
            instances.append(distance_matrix)
            neighbors.append(neighbor_matrix)
            instances_scale.append(scale)
            instances_name.append(filename)

        self.coords, self.instances, self.instances_name, self.instances_scale = coords, instances, instances_name, instances_scale

    def tour_cost(self, instance, solution, problem_size):
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def generate_neighborhood_matrix(self, instance):
        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            neighborhood_matrix[i] = sorted_indices

        return neighborhood_matrix

    def evaluateGLS(self, heuristic: Callable, temp_file_path):
        nins = len(self.instances_name)
        inputs = [(x, self.instances_name[x], self.instances_scale[x], self.instances[x], self.coords[x],
                   self.time_limit, self.ite_max, self.perturbation_moves, heuristic, temp_file_path) for x in range(nins)]

        gaps = Parallel(n_jobs=32, timeout=self.time_limit * 1.1)(
            delayed(solve_instance_tsplib)(*input) for input in inputs)

        if len(gaps) >= 50 and len(gaps) % 50 == 0:
            avg_gap = np.mean(gaps)
            print("Average Gap for", len(gaps), "instances:", avg_gap)

        return np.mean(gaps)

    def evaluate(self, heuristic: Callable):
        # 创建一个随机名字的临时文件
        temp_file_path = os.path.dirname(os.path.abspath(__file__))
        random_num = random.randint(0, 10000000)
        temp_file_path = os.path.join(temp_file_path, f'.tsp_lib_res{random_num}.txt')
        # 评估
        self.evaluateGLS(heuristic, temp_file_path)
        res = self.compare_results(temp_file_path)
        # 删除临时文件
        os.remove(temp_file_path)
        return res

    def compare_results(self, result_file):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        true_values_file = os.path.join(cur_path, 'opt_cost.txt')

        # 读取计算结果文件
        with open(result_file, 'r') as result_file:
            result_lines = result_file.readlines()

        # 读取真实值文件
        with open(true_values_file, 'r') as true_values_file:
            true_values_lines = true_values_file.readlines()

        # 创建一个字典来存储真实值
        true_values_dict = {}
        for line in true_values_lines:
            task_name, true_value = line.strip().split(',')
            true_values_dict[task_name] = float(true_value)

        gaps = []
        # 遍历计算结果并比较
        for line in result_lines:
            parts = line.strip().split(',')
            task_name = parts[1].split('.')[0]
            best_cost = float(parts[3])

            if task_name in true_values_dict:
                true_value = true_values_dict[task_name]
                # print(f'instance: {task_name}; my cost: {best_cost: .1f}; opt cost: {true_value}')

                absolute_difference = abs(true_value - best_cost)
                gaps.append(absolute_difference / true_value)
        # print(np.mean(gaps))
        return np.mean(gaps)


if __name__ == '__main__':
    def compare_results(result_file, true_values_file):
        # 读取计算结果文件
        with open(result_file, 'r') as result_file:
            result_lines = result_file.readlines()

        # 读取真实值文件
        with open(true_values_file, 'r') as true_values_file:
            true_values_lines = true_values_file.readlines()

        # 创建一个字典来存储真实值
        true_values_dict = {}
        for line in true_values_lines:
            task_name, true_value = line.strip().split(',')
            true_values_dict[task_name] = float(true_value)

        gaps = []
        # 遍历计算结果并比较
        for line in result_lines:
            parts = line.strip().split(',')
            task_name = parts[1].split('.')[0]
            best_cost = float(parts[3])

            if task_name in true_values_dict:
                true_value = true_values_dict[task_name]
                # print(f'instance: {task_name}; my cost: {best_cost: .1f}; opt cost: {true_value}')

                absolute_difference = abs(true_value - best_cost)
                gaps.append(absolute_difference / true_value)
        print(np.mean(gaps))
        return np.mean(gaps)


    # 使用例子
    result_file_path = '.tsp_lib_res.txt'
    true_values_file_path = 'opt_cost.txt'
    compare_results(result_file_path, true_values_file_path)

