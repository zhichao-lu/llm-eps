import time
from typing import Callable

import numpy as np
from numba import jit

import sys
sys.path.append('../')
from tsp_eval_helper import gls_operators
from tsp_eval_helper import utils
import random

np.random.seed(2024)
random.seed(2024)


# @jit(nopython=True)
def nearest_neighbor(dis_matrix, depot):
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)
    tour.append(depot)
    return tour


# @jit(nopython=True)
def nearest_neighbor_2End(dis_matrix, depot):
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(depot)
    route2End = np.zeros((n, 2))
    route2End[0, 0] = tour[-2]
    route2End[0, 1] = tour[1]
    for i in range(1, n):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    return route2End


@jit(nopython=True)
def local_search(init_tour, init_cost, D, N, first_improvement=False):
    cur_route, cur_cost = init_tour, init_cost
    improved = True
    while improved:
        improved = False
        delta, new_tour = gls_operators.two_opt_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour
        delta, new_tour = gls_operators.relocate_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour
    return cur_route, cur_cost


@jit(nopython=True)
def route2tour(route):
    s = 0
    tour = []
    for i in range(len(route)):
        tour.append(route[s, 1])
        s = route[s, 1]
    return tour


@jit(nopython=True)
def tour2route(tour):
    n = len(tour)
    route2End = np.zeros((n, 2))
    route2End[tour[0], 0] = tour[-1]
    route2End[tour[0], 1] = tour[1]
    for i in range(1, n - 1):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    route2End[tour[n - 1], 0] = tour[n - 2]
    route2End[tour[n - 1], 1] = tour[0]
    return route2End


# @jit(nopython=True)
def guided_local_search(coords, edge_weight, nearest_indices, init_tour, init_cost, t_lim, ite_max, perturbation_moves,
                        first_improvement=False, guide_algorithm: Callable = None):

    cur_route, cur_cost = local_search(init_tour, init_cost, edge_weight, nearest_indices, first_improvement)
    best_route, best_cost = cur_route, cur_cost
    length = len(edge_weight[0])

    iter_i = 0
    edge_penalty = np.zeros((length, length))
    while iter_i < ite_max and time.time() < t_lim:

        for move in range(perturbation_moves):
            cur_tour, best_tour = route2tour(cur_route), route2tour(best_route)

            # TODO RZ: Some modifications here, the 'guide_algorithm' refers to the 'update_edge_distance()' function.
            # edge_weight_guided = guide_algorithm.update_edge_distance(edge_weight, np.array(cur_tour), edge_penalty)
            edge_weight_guided = guide_algorithm(edge_weight, np.array(cur_tour), edge_penalty)
            edge_weight_guided = np.asmatrix(edge_weight_guided)
            edge_weight_gap = edge_weight_guided - edge_weight

            for topid in range(5):
                max_indices = np.argmin(-edge_weight_gap, axis=None)
                rows, columns = np.unravel_index(max_indices, edge_weight_gap.shape)
                edge_penalty[rows, columns] += 1
                edge_penalty[columns, rows] += 1
                edge_weight_gap[rows, columns] = 0
                edge_weight_gap[columns, rows] = 0

                for id in [rows, columns]:
                    delta, new_route = gls_operators.two_opt_o2a_all(cur_route, edge_weight_guided, nearest_indices, id)
                    if delta < 0:
                        cur_cost = utils.tour_cost_2End(edge_weight, new_route)
                        cur_route = new_route
                    delta, new_route = gls_operators.relocate_o2a_all(cur_route, edge_weight_guided, nearest_indices,
                                                                      id)
                    if delta < 0:
                        cur_cost = utils.tour_cost_2End(edge_weight, new_route)
                        cur_route = new_route

        cur_route, cur_cost = local_search(cur_route, cur_cost, edge_weight, nearest_indices, first_improvement)
        cur_cost = utils.tour_cost_2End(edge_weight, cur_route)
        if cur_cost < best_cost:
            best_route, best_cost = cur_route, cur_cost
        iter_i += 1
        if iter_i % 50 == 0:
            cur_route, cur_cost = best_route, best_cost

    return best_route, best_cost, iter_i
