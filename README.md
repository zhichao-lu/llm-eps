# Code implementations for *FunSearch*, *RandomSampling*, *(1+1)-EPS*, EoH, ReEvo

This repository includes code for the following paper:

> *Towards Understanding the Effectiveness of Automatic Heuristic Design with Large Language Models*

------

## Installation and requirements

Please note that **the Python version must be larger or equal to Python 3.9**, or the '*ast*' package used in the implementations will fail to work. 

You can run  locally if enough GPU devices are available. Or you can try to use LLM interfaces to request responses online. 

Please install the packages listed in `requirements.txt`.

## LLMs

Not all LLMs can be used for an LLM-based EPS method.

| LLMs                   | FunSearch | (1+1)-EPS (HillClimbing) | RandomSampling | EoH  | ReEvo |
| :--------------------- | :-------: | :----------------------: | :------------: | :--: | :---: |
| UniXcoder              |     x     |            x             |       v        |  x   |   x   |
| CodeLlama-7B-instruct  |     v     |            v             |       v        |  v   |   v   |
| CodeLlama-34B-Instruct |     v     |            v             |       v        |  v   |   v   |
| StarCoder              |     v     |            v             |       v        |  x   |   x   |
| DeepSeek-Coder-6.7B    |     v     |            v             |       v        |  v   |   v   |
| DeepSeek-Coder-33B     |     v     |            v             |       v        |  v   |   v   |
| GPT-3.5 (API)          |     v     |            v             |       v        |  v   |   v   |
| GPT-4 (API)            |     v     |            v             |       v        |  v   |   v   |
| Claude-3-Opus (API)    |     v     |            v             |       v        |  v   |   v   |

## Project structure

**`funsearch`** folder includes FunSearch implementations.

&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name` (admissible_set, bin_packing_or, bin_packing_weibull, gls_tsp)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name_completion.py` for LLMs that support code completion generation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name_instruct.py` for LLMs that support instruct generation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name_api.py` using LLM api.

&nbsp;&nbsp;&nbsp;&nbsp;|----`funsearch_impl`

**`rand_search`** folder includes RandomSampling implementations.

&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name` (admissible_set, bin_packing_or, bin_packing_weibull, gls_tsp)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name_completion.py` for LLMs that support code completion generation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name_instruct.py` for LLMs that support instruct generation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name_api.py` using LLM api.

&nbsp;&nbsp;&nbsp;&nbsp;|----`rand_search_impl`

**`hill_climb`** folder includes (1+1)-EPS implementations.

&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name` (admissible_set, bin_packing_or, bin_packing_weibull, gls_tsp)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name_completion.py` for LLMs that support code completion generation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name_instruct.py` for LLMs that support instruct generation.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`task_name_api.py` using LLM api.

&nbsp;&nbsp;&nbsp;&nbsp;|----`hill_climb_impl`

## Run LLM-based EPS

### Python script

Run differnet python script for different LLM.

| LLMs                   | Run Python file         |
| :--------------------- | :---------------------- |
| UniXcoder              | task_name_completion.py |
| CodeLlama-7B-instruct  | task_name_completion.py |
| CodeLlama-34B-Instruct | task_name_completion.py |
| StarCoder              | task_name_completion.py |
| DeepSeek-Coder-6.7B    | task_name_instruct.py   |
| DeepSeek-Coder-33B     | task_name_instruct.py   |
| GPT-3.5 (API)          | task_name_api.py        |
| GPT-4 (API)            | task_name_api.py        |
| Claude-3-Opus (API)    | task_name_api.py        |

### Run with local LLM

##### Download LLM

Please refer to tutorials at Huggingface (https://huggingface.co) or Huggingface mirror  (https://hf-mirror.com).

##### Deploy an LLM locally

Please note that it is suggested to deploy the LLM server in a 'screen' environment. Which can be created as below:

```shee
screen -R local_llm
```

The server scripts are in `llm_server` folder. Please find the right python file.

```shell
cd llm_server

python xxx.py \
	--path [your path to LLM] \
  --d [GPU ids, e.g., 0 1 2 3 4] \
  --host [the ip of the server, default is 127.0.0.1] \
  --port [the port of the server] \
  --quantization  # if you want to do quantization, please add this argument
```

##### Prepare a config file

If there is an issue with the LLM server, or you need to change the url of the LLM during **runtime**, you can simply modify the config file instead of re-running.

An template for a config file is shown below:

```json
{
  "url": "http://127.0.0.1:11101/completions",  # the URL for the LLM server
  "samples_per_prompt": 4,                      # samples per prompt
  "batch_inference": true                       # whether to use batch inference to accelerate
}
```

##### Start LLM-based EPS 

We suggest run using 'nohup'. The instruction is shown below.

```shell
nohup python xxx.py --run 1 --config run1_runtime_llm_config.json > run1.out 2>&1 &
```

### Run with LLM API

##### Choose the correct LLM model and set the API key

Please modify the code of `_draw_sample` function in `task_name_api.py`. 

```python
def _draw_sample(self, content: str) -> str:
    prompt = '\n'.join([content, self._additional_prompt])
    while True:
        try:
            conn = http.client.HTTPSConnection(
              "api.chatanywhere.com.cn",  # 1. Set API URL.
              timeout=self._timeout
            )  
            payload = json.dumps({
                "max_tokens": 512,
                "model": "gpt-3.5-turbo",  # 2. Set LLM model name. Please refer to your API provider.
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            headers = {
                'Authorization': 'Bearer sk-your api key here',  # 3. Set your API key.
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            data = json.loads(data)
            response = data['choices'][0]['message']['content']
            # trim function
            if self._trim:
                response = _trim_preface_of_body(response)
            return response
        except Exception as e:
            # print(e)
            time.sleep(2)
            continue
```

##### Start LLM-based EPS

We suggest run using 'nohup'. The instruction is shown below.

```shell
nohup python xxx.py --run 1 > run1.out 2>&1 &
```

