import multiprocessing
import sys
from typing import Any, Callable

sys.path.append('../')
from my_task import _evaluator_accelerate
from my_task.tsp_eval_helper import ael_evaluation


def evaluate(function: Callable):
    evaluator = ael_evaluation.Evaluation()
    return evaluator.evaluate(function)


class Sandbox:
    def __init__(self, verbose=False, numba_accelerate=True, timeout=30):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate
        self._timeout = timeout

    def run(self,
            function_to_evolve: str,  # RZ: accelerate the code by decorating @numba.jit() on function_to_evolve.
            ) -> tuple[Any, bool]:
        try:
            func_name = _evaluator_accelerate._extract_function_name(function_to_evolve)
            function_to_evolve = _evaluator_accelerate.add_import_package_statement(function_to_evolve, 'numpy', 'np')
            if self._numba_accelerate:
                function_to_evolve = _evaluator_accelerate.add_numba_decorator(
                    program=function_to_evolve,
                    function_name=[func_name]
                )
            # print(function_to_evolve)
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(function_to_evolve, all_globals_namespace)
            # get the pointer of 'function_to_run'
            func_pointer = all_globals_namespace[func_name]
            # return the execution results
            results = evaluate(func_pointer)
            if results is not None:
                if not isinstance(results, (int, float)):
                    results = (None, False)
                else:
                    # negation because our optimization objective is bigger, the better
                    results = (results, True)  # convert to FunSearch result format
            else:
                results = None, False
        except Exception as e:
            results = None, False
        return results


test_code = '''
def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray) -> np.ndarray:
    """
    Args:
        edge_distance (np.ndarray): Original edge distance matrix.
        local_opt_tour (np.ndarray): Local optimal solution path.
        edge_n_used (np.ndarray): Matrix representing the number of times each edge is used.
    Return:
        updated_edge_distance: updated score of each edge distance matrix.
    """ 
    # Assuming edge_distance, local_opt_tour, and edge_n_used have compatible shapes
    num_nodes = edge_distance.shape[0]

    # Initialize an array to store the updated edge distances
    updated_edge_distance = np.copy(edge_distance)

    # Iterate over the edges in the local optimal tour
    for i in range(num_nodes - 1):
        current_node = local_opt_tour[i]
        next_node = local_opt_tour[i + 1]

        # Update the edge distance based on the number of times it has been used
        updated_edge_distance[current_node, next_node] *= (1 + edge_n_used[current_node, next_node])

    # Update the last edge in the tour
    updated_edge_distance[local_opt_tour[-1], local_opt_tour[0]] *= (1 + edge_n_used[local_opt_tour[-1], local_opt_tour[0]])

    return updated_edge_distance
'''

if __name__ == '__main__':
    sandbox = Sandbox()
    res = sandbox.run(test_code)
    print(res)
