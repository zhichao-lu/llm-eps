import json
import logging
import os
import sys
from typing import List, Dict

import requests

sys.path.append('../')
from my_task import _evaluator_accelerate
# from openai import OpenAI
import time
import re
import http.client


def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()


def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if "[*] Running ..." in log or "Traceback" in log:
            if log_status and "Traceback" in log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def multi_chat_completion(messages_list: list[list[dict]], n=1, model: str = "gpt-3.5-turbo-1106",
                          temperature: float = 0., local_llm: bool = False,
                          config_path=None) -> List[List[str]]:
    """
    An example of messages_list:
    
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {"role": "system", "content": "You are a knowledgeable guide."},
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "system", "content": "You are a witty comedian."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    ]
    param: n: number of responses to generate for each message in messages_list
    """
    contents = []
    for messages in messages_list:
        content = chat_completion(n, messages, model, temperature, local_llm, config_path)
        contents.append(content)
    return contents


class LLMAPI:
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, samples_per_prompt: int, timeout=30, trim=True):
        self._samples_per_prompt = samples_per_prompt
        self._trim = trim
        self._timeout = timeout

    def draw_samples(self, message: List[Dict]) -> List[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(message) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, message: List[Dict]) -> str:
        while True:
            try:
                conn = http.client.HTTPSConnection("[PUT YOUR API ENDPOINT HERE]", timeout=self._timeout)
                payload = json.dumps({
                    "max_tokens": 512,
                    "model": "[YOUR MODEL HERE]",
                    "messages": message
                })
                headers = {
                    'Authorization': 'Bearer [PUT YOUR API KEY HERE]',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)
                response = data['choices'][0]['message']['content']
                response = _evaluator_accelerate.trim_trim_trim(response)
                return response
            except Exception as e:
                # print(e)
                time.sleep(2)
                continue


def _load_runtime_config(config_path):
    assert config_path is not None
    cur_dir = os.path.dirname(__file__)
    config_path = os.path.join(cur_dir, '../', config_path)
    with open(config_path, 'r') as file:
        data_dict = json.load(file)
    return data_dict


class LocalLLM:
    """Language model that predicts continuation of provided source code.
    """

    def _update_config(self):
        try:
            config_dict = _load_runtime_config(self._config_path)
            if config_dict['url'] is None:
                self._url = self._default_url
            else:
                self._url = config_dict['url']

            if config_dict['batch_inference'] is None:
                self._batch_inference = self._default_batch_inference
            else:
                self._batch_inference = config_dict['batch_inference']
        except Exception as e:
            print(e)
            self._url = self._default_url
            self._batch_inference = self._default_batch_inference

    def __init__(
            self,
            config_path: str,
            samples_per_prompt: int,
            batch_inference: bool = True,
    ) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample functions. The batch size equals to the samples_per_prompt.
        """
        self._config_path = config_path
        self._samples_per_prompt = samples_per_prompt
        self._default_url = 'http://172.18.36.43:12001/chat/completions'
        self._default_batch_inference = batch_inference
        #
        self._url = self._default_url
        self._batch_inference = self._default_batch_inference
        self._update_config()

    def draw_samples(self, messages: List[Dict]) -> List[str]:
        """Returns multiple predicted continuations of `prompt`.
        """
        while True:
            try:
                self._update_config()
                all_samples = []
                if self._batch_inference:
                    response = self._do_request(messages)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(messages)
                        all_samples.append(response)
                return all_samples
            except Exception as e:
                print(e)
                # time.sleep(2)
                continue

    def _do_request(self, messages: List[Dict]) -> str:
        # repeat the prompt for batch inference (inorder to decease the sample delay)
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        data = {
            'messages': messages,
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


def chat_completion(n: int, messages: list[dict], model: str = None, temperature: float = None,
                    local_llm: bool = False,
                    config_path=None) \
        -> List[dict] | List[str]:
    """
    Generate n responses using OpenAI Chat Completions API
    [{"role": "system", "content": system}, {"role": "user", "content": user}]
    """
    if local_llm:
        llm_api = LocalLLM(config_path, n)
    else:
        llm_api = LLMAPI(samples_per_prompt=n)
    all_samples = llm_api.draw_samples(messages)
    return all_samples


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end + 1])

    if code_string is None:
        return None
    # Add import statements if not present
    if "import" not in code_string:
        code_string = "import numpy as np\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string
