import json
import multiprocessing
import pickle
from argparse import ArgumentParser
from typing import Collection, Any, Tuple

import requests
import sys

sys.path.append('../')

# The evaluation of gls_tsp is helped by AEL module as it is too complex for me.
# Thanks to Fei LIU for his help!
from gls_tsp.eval_helper import ael_evaluation

from gls_tsp import utils
from rand_search_impl import rand_search
from rand_search_impl import config
from rand_search_impl import sampler
from rand_search_impl import evaluator_accelerate
from rand_search_impl import evaluator
from rand_search_impl import code_manipulation

parser = ArgumentParser()
parser.add_argument('--run', type=int)
# parser.add_argument('--port', type=int, default=11045)
parser.add_argument('--config', type=str, default='run1_runtime_llm_config.json')
args = parser.parse_args()


def _load_runtime_config():
    config_path = args.config
    assert config_path is not None
    with open(config_path, 'r') as file:
        data_dict = json.load(file)
    return data_dict


class LocalLLM(sampler.LLM):
    """Language model that predicts continuation of provided source code.
    """

    def _update_config(self):
        try:
            config_dict = _load_runtime_config()
            if config_dict['url'] is None:
                self._url = self._default_url
            else:
                self._url = config_dict['url']

            if config_dict['batch_inference'] is None:
                self._batch_inference = self._default_batch_inference
            else:
                self._batch_inference = config_dict['batch_inference']

            if config_dict['samples_per_prompt'] is None:
                self._samples_per_prompt = self._default_samples_per_prompt
            else:
                self._samples_per_prompt = config_dict['samples_per_prompt']
        except:
            self._url = self._default_url
            self._batch_inference = self._default_batch_inference
            self._samples_per_prompt = self._default_samples_per_prompt

    def __init__(
            self,
            samples_per_prompt: int,
            batch_inference: bool = False
    ) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample functions. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)
        self._default_url = 'http://127.0.0.1:11045/completions'
        self._default_samples_per_prompt = samples_per_prompt
        self._default_batch_inference = batch_inference
        #
        self._url = self._default_url
        self._batch_inference = self._default_batch_inference

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`.
        """
        while True:
            try:
                self._update_config()
                all_samples = []
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)
                return all_samples
            except:
                continue

    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference (inorder to decease the sample delay)
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        if response.status_code == 200:
            response = response.json()["content"]
            return response if self._batch_inference else response[0]


class Sandbox(evaluator.Sandbox):
    """Sandbox for executing generated code. Implemented by RZ.

    RZ: Sandbox returns the 'score' of the program and:
    1) avoids the generated code to be harmful (accessing the internet, take up too much RAM).
    2) stops the execution of the code in time (avoid endless loop).
    """

    def __init__(
            self, verbose=False,
            numba_accelerate=True
    ):
        """
        Args:
            verbose         : Print evaluate information.
            numba_accelerate: Use numba to accelerate the evaluation. It should be noted that not all numpy functions
                              support numba acceleration, such as np.piecewise().
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
            self,
            program: str,
            function_to_run: str,  # RZ: refers to the name of the function to run (e.g., 'evaluate')
            function_to_evolve: str,  # RZ: accelerate the code by decorating @numba.jit() on function_to_evolve.
            inputs: Any,  # refers to the dataset
            test_input: str,  # refers to the current instance
            timeout_seconds: int,
            # **kwargs  # RZ: add this
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded.
        RZ: If the generated code (generated by LLM) is executed successfully,
        the output of this function is the score of a given program.
        The evaluate time limitation and exception handling modules are implemented within AEL's evaluation module.
        """
        try:
            if self._numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_name=[function_to_evolve]
                )
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(program, all_globals_namespace)
            # get the pointer of 'function_to_evolve', which will be sent to AEL's evaluation module later
            function_to_evolve_pointer = all_globals_namespace[function_to_evolve]
            evaluator = ael_evaluation.Evaluation()
            # do evaluate
            results = evaluator.evaluate(heuristic_func=function_to_evolve_pointer)
            # make sure the score is int or float
            if results is not None:
                if not isinstance(results, (int, float)):
                    results = (None, False)
                else:
                    # negation because our optimization objective is bigger, the better
                    results = (-results, True)  # convert to rand_search result format
            else:
                results = None, False
        except:
            results = None, False

        return results


# It should be noted that the if __name__ == '__main__' is required.
# Because the inner code uses multiprocess evaluation.
if __name__ == '__main__':
    class_config = config.ClassConfig(llm_class=LocalLLM, sandbox_class=Sandbox)
    config = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=50)
    global_max_sample_num = 10_000  # if it is set to None, rand_search will execute an endless loop

    rand_search.main(
        specification=utils.specification,
        inputs=[None],
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir=f'logs/run{args.run}',
    )
