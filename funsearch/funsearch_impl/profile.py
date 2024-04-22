# implemented by RZ
# profile the experiment using tensorboard

from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
from funsearch_impl import code_manipulation
from torch.utils.tensorboard import SummaryWriter


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
            resume_run: bool = False
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._samples_json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._samples_json_dir, exist_ok=True)
        self._island_programs_json_dir = os.path.join(log_dir, 'island_programs')
        os.makedirs(self._island_programs_json_dir, exist_ok=True)
        # self._pkl_dir = pkl_dir
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        # self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

        if resume_run:
            self._resume_run()

    def _resume_run(self):
        # read total sample nums from file
        log_path = self._samples_json_dir
        max_samples = 0
        for dir in os.listdir(log_path):
            order = int(dir.split('.')[0].split('_')[1])
            if order > max_samples:
                max_samples = order
        self._num_samples = max_samples

        # read max score from file
        max_score = float('-inf')
        for dir in os.listdir(log_path):
            json_file = os.path.join(log_path, dir)
            with open(json_file, 'r') as f:
                json_dict = json.load(f)
            if json_dict['score'] is not None and json_dict['score'] > max_score:
                max_score = json_dict['score']
        assert max_score != float('-inf')
        self._cur_best_program_score = max_score

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )

    def _write_json(self, programs: code_manipulation.Function):
        sample_order = self._num_samples
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(programs)
        score = programs.score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score
        }
        path = os.path.join(self._samples_json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return
        # sample_orders: int = programs.global_sample_nums
        self._num_samples += 1
        self._record_and_verbose(programs)
        self._write_tensorboard()
        self._write_json(programs)

    def record_program_nums_in_island(self, program_nums_in_islands: List[int]):
        try:
            content = {
                'sample_order': self._num_samples,
                'program_nums_in_islands': program_nums_in_islands
            }
            path = os.path.join(self._island_programs_json_dir, f'samples_{self._num_samples}.json')
            with open(path, 'w') as json_file:
                json.dump(content, json_file)
        except:
            pass

    def _record_and_verbose(self, function):
        # function = self._all_sampled_functions[sample_orders]
        # function_name = function.name
        # function_body = function.body.strip('\n')
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(self._num_samples)}')
        print(f'======================================================\n\n')

        # update best function
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = self._num_samples

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

        # update ...
        # self._each_sample_best_program_score.append(self._cur_best_program_score)
        # self._each_sample_evaluate_success_program_num.append(self._evaluate_success_program_num)
        # self._each_sample_evaluate_failed_program_num.append(self._evaluate_failed_program_num)
        # self._each_sample_tot_sample_time.append(self._tot_sample_time)
        # self._each_sample_tot_evaluate_time.append(self._tot_evaluate_time)
