import json
import logging
import subprocess
import sys

import numpy as np
import os
import hydra
import logging
import os
from pathlib import Path
import subprocess
from utils.utils import *
from my_task import admissible_set_eval


class ReEvo:
    # def _local_llm_get_prompt

    def __init__(self, cfg, root_dir, evaluator=None, log_path=None, local_llm=False, runtime_config_path=None) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.evaluator = evaluator
        self.log_path = log_path
        #
        self.local_llm = local_llm
        self.runtime_config_path = runtime_config_path
        _run = 'None' if 'run' not in self.cfg else self.cfg.run
        _cur_file_ = os.path.dirname(__file__)
        if local_llm:
            _llm = 'codellama'
        else:
            _llm = 'gpt35'
        self._my_log_path = os.path.join(_cur_file_, 'all_logs', f'{self.cfg.problem.problem_name}_{_llm}_run{_run}')
        os.makedirs(self._my_log_path, exist_ok=True)
        #
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.best_obj_overall = float("inf")
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None

        self.init_prompt()
        self.init_population()

    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type

        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)

        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/{self.cfg.suffix.lower()}.py"

        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        if os.path.exists(f'{problem_prompt_path}/external_knowledge.txt'):
            self.external_knowledge = file_to_string(f'{problem_prompt_path}/external_knowledge.txt')
            self.long_term_reflection_str = self.external_knowledge
        else:
            self.external_knowledge = ""

        # Common prompts
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        self.user_reflector_st_prompt = file_to_string(
            f'{self.prompt_dir}/common/user_reflector_st.txt') if self.problem_type != "black_box" else file_to_string(
            f'{self.prompt_dir}/common/user_reflector_st_black_box.txt')  # shrot-term reflection
        self.user_reflector_lt_prompt = file_to_string(
            f'{self.prompt_dir}/common/user_reflector_lt.txt')  # long-term reflection
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')
        self.mutataion_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name,
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )
        self.seed_prompt = file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flag to print prompts
        self.print_crossover_prompt = True  # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True  # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = True  # Print short-term reflection prompt for the first iteration
        self.print_long_term_reflection_prompt = True  # Print long-term reflection prompt for the first iteration

    def ___init_population(self):
        pass

    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logging.info("Seed function code: \n" + code)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        self.update_iter()

        # Generate responses
        system = self.system_generator_prompt
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
        # TODO 这里是发送请求的部分额？
        responses = chat_completion(self.cfg.pop_size, messages, self.cfg.model, self.cfg.temperature, self.local_llm,
                                    self.runtime_config_path)

        # Responses to population
        population = self.responses_to_population(responses)

        # Run code and evaluate population
        population = self.evaluate_population(population)

        # Update iteration
        self.population = population
        self.update_iter()

    def response_to_individual(self, response, response_id, file_name=None) -> dict:
        """
        Convert response to individual
        """
        # content = response.message.content  
        content = response  # TODO 如果用自己的API，那么response其实就是content

        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(content + '\n')

        code = extract_code_from_generator(content)

        # Extract code and description from response
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"

        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
        }
        return individual

    def responses_to_population(self, responses) -> list[dict]:
        """
        Convert responses to population. Applied to the initial population.
        """
        population = []
        for response_id, response in enumerate(responses):
            individual = self.response_to_individual(response, response_id)
            population.append(individual)
        return population

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    # def __evaluate_population(self, population: list[dict]) -> list[dict]:
    #     """
    #     Evaluate population by running code in parallel and computing objective values.
    #     """
    #     inner_runs = []
    #     # Run code to evaluate
    #     for response_id in range(len(population)):  # TODO 这里所谓的response id其实就是一个编号
    #         self.function_evals += 1
    #         # Skip if response is invalid
    #         if population[response_id]["code"] is None:
    #             population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
    #             inner_runs.append(None)
    #             continue
    #
    #         logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
    #
    #         try:
    #             process = self._run_code(population[response_id], response_id)
    #             inner_runs.append(process)
    #         except Exception as e:  # If code execution fails
    #             logging.info(f"Error for response_id {response_id}: {e}")
    #             population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
    #             inner_runs.append(None)
    #
    #     # Update population with objective values
    #     for response_id, inner_run in enumerate(inner_runs):
    #         if inner_run is None:  # If code execution fails, skip
    #             continue
    #         try:
    #             inner_run.communicate(timeout=self.cfg.timeout)  # Wait for code execution to finish
    #         except subprocess.TimeoutExpired as e:
    #             logging.info(f"Error for response_id {response_id}: {e}")
    #             population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
    #             inner_run.kill()
    #             continue
    #
    #         individual = population[response_id]
    #         stdout_filepath = individual["stdout_filepath"]
    #         with open(stdout_filepath, 'r') as f:  # read the stdout file
    #             stdout_str = f.read()
    #         traceback_msg = filter_traceback(stdout_str)
    #
    #         individual = population[response_id]
    #         # Store objective value for each individual
    #         if traceback_msg == '':  # If execution has no error
    #             try:  # TODO 这里得到fitness evaluation的结果
    #                 # TODO individual['obj'] = score
    #                 individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(
    #                     stdout_str.split('\n')[-2])
    #                 # TODO individual['exec_success'] = 是否执行成功
    #                 individual["exec_success"] = True
    #             except:
    #                 population[response_id] = self.mark_invalid_individual(population[response_id],
    #                                                                        "Invalid std out / objective value!")
    #         else:  # Otherwise, also provide execution traceback error feedback
    #             population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)
    #
    #         logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
    #     return population

    def evaluate_population(self, population: list[dict]) -> list[dict]:
        """Evaluate population by running code in parallel and computing objective values.
        """
        from my_task import admissible_set_eval
        sand_box = admissible_set_eval.Sandbox()

        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):  # TODO 这里所谓的response id其实就是一个编号
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue

            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")

            result, run_ok = sand_box.run(population[response_id]['code'])

            # 将sample写到日志中
            with open(os.path.join(self._my_log_path, f'samples_{self.function_evals}.json'), 'w') as f:
                _score = result if run_ok else None
                content = {
                    'function': population[response_id]['code'],
                    'score': _score,
                    'iter': self.iteration
                }
                json.dump(content, f)
                f.close()

            individual = population[response_id]
            if run_ok:
                individual["obj"] = result
                individual["exec_success"] = run_ok
            else:
                population[response_id] = self.mark_invalid_individual(population[response_id], 'RZ: no message.')

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population

    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")

        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')

        # Execute the python file with flags
        with open(individual["stdout_filepath"], 'w') as f:
            process = subprocess.Popen(
                ['python', '-u', f'{self.root_dir}/problems/{self.problem}/eval.py', f'{self.problem_size}',
                 self.root_dir, "train"],
                stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration,
                            response_id=response_id)
        return process

    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual["obj"] for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))

        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]

        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")

        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1

    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [individual for individual in population if
                          individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        if ind1["obj"] == ind2["obj"]:
            raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["obj"] < ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        elif ind1["obj"] > ind2["obj"]:
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])

        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            worse_code=worse_code,
            better_code=better_code
        )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
            logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code

    def short_term_reflection(self, population: list[dict]) -> tuple[list[list[str]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i + 1]

            # Short-term reflection
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)

        # Multi-processed chat completion
        responses_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature, self.local_llm, self.runtime_config_path)
        return responses_lst, worse_code_lst, better_code_lst

    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc=self.problem_desc,
            prior_reflection=self.long_term_reflection_str,
            new_reflection="\n".join(short_term_reflections),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_reflection_prompt = False

        response = chat_completion(1, messages, self.cfg.model, self.cfg.temperature, self.local_llm, self.runtime_config_path)
        self.long_term_reflection_str = response[0]

        # Write reflections to file
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(short_term_reflections) + '\n')

        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_reflection_str + '\n')

    def crossover(self, short_term_reflection_tuple: tuple[list[list[str]], list[str], list[str]]) -> list[dict]:
        reflection_responses_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        crossed_population = []
        messages_lst = []
        for response, worse_code, better_code in zip(reflection_responses_lst, worse_code_lst, better_code_lst):
            # reflection = response[0].message.content
            reflection = response[0]

            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                func_name=self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)

            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False

        # Multi-processed chat completion
        responses_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature, self.local_llm, self.runtime_config_path)
        response_id = 0
        for i in range(len(responses_lst)):
            individual = self.response_to_individual(responses_lst[i][0], response_id)
            crossed_population.append(individual)
            response_id += 1

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population

    def mutate(self) -> list[dict]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1)
        user = self.mutataion_prompt.format(
            user_generator=self.user_generator_prompt,
            reflection=self.long_term_reflection_str + self.external_knowledge,
            func_signature1=func_signature1,
            elitist_code=filter_code(self.elitist["code"]),
            func_name=self.func_name,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False

        responses = chat_completion(int(self.cfg.pop_size * self.mutation_rate), messages, self.cfg.model,
                                    self.cfg.temperature, self.local_llm, self.runtime_config_path)
        population = self.responses_to_population(responses)
        return population

    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            # If all individuals are invalid, stop
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")

            # Select  # add elitist to population for selection
            population_to_select = self.population if self.elitist is None else [self.elitist] + self.population
            selected_population = self.random_select(population_to_select)
            if selected_population is not None:
                # Short-term reflection
                short_term_reflection_tuple = self.short_term_reflection(
                    selected_population)  # (responses_lst, worse_code_lst, better_code_lst)
                # Crossover
                crossed_population = self.crossover(short_term_reflection_tuple)
                # Evaluate
                self.population = self.evaluate_population(crossed_population)
                # Update
                self.update_iter()
                # Long-term reflection
                # self.long_term_reflection([response[0].message.content for response in short_term_reflection_tuple[0]])
                self.long_term_reflection([response[0] for response in short_term_reflection_tuple[0]])
            # Mutate
            mutated_population = self.mutate()
            # Evaluate
            self.population.extend(self.evaluate_population(mutated_population))
            # Update
            self.update_iter()

        return self.best_code_overall, self.best_code_path_overall


ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    if 'local_llm' in cfg:
        local_llm = True
        runtime_config_path = str(cfg.runtime_config_path)
    else:
        local_llm = False
        runtime_config_path = ''

    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.model}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    # Main algorithm
    reevo = ReEvo(cfg, ROOT_DIR, local_llm=local_llm, runtime_config_path=runtime_config_path)
    best_code_overall, best_code_path_overall = reevo.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")

    # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
    with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/{cfg.suffix.lower()}.py", 'w') as file:
        file.writelines(best_code_overall + '\n')
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script...: {test_script}")
    with open(test_script_stdout, 'w') as stdout:
        subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
    logging.info(f"Validation script finished. Results are saved in {test_script_stdout}.")

    # Print the results
    with open(test_script_stdout, 'r') as file:
        for line in file.readlines():
            logging.info(line.strip())


if __name__ == "__main__":
    """
python reevo_xxx.py problem=admissible_set_15_10 +local_llm=True +runtime_config_path=admissible_set_run1.json +run=1
    """
    main()
