# ReEvo

## Run using API
1. Please fill your API endpoint and API key in `class LLMAPI` in `utils/utils.py`.
```python
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
```
2. Run script.
```shell
python reevo_[task_name].py problem=[task_name] +local_llm=False +llm_name=GPT4 +run=1

# examples:
python reevo_admissible_set.py problem=admissible_set_15_10 +local_llm=False +llm_name=GPT4 +run=1
python reevo_bin_packing_or.py problem=bin_packing_or +local_llm=False +llm_name=GPT4 +run=1
python reevo_bin_packing_weibull.py problem=bin_packing_weibull +local_llm=False +llm_name=GPT4 +run=1
python reevo_gls_tsp.py problem=gls_tsp +local_llm=False +llm_name=GPT4 +run=1
```

## Run using deployed LLM
1. Please prepare a json file similar to admissible_set_run1.json. The json file should define the url for your local deployed LLM.
2. Run script
```shell
python reevo_[task_name].py problem=[task_name] +local_llm=True +llm_name=CodeLlama7b +runtime_config_path=[Your json file].json +run=1
```
