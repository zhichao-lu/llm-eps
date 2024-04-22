import gc
import os
from argparse import ArgumentParser
from typing import List, Dict

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)

# model_path_home = '/raid/zhichao/zhangrui/LLMs'
default_model_path_path = '/raid/zhichao/zhangrui/LLMs/deepseek-coder-6.7b-instruct'

# arguments
parser = ArgumentParser()
parser.add_argument('--d', nargs='+', default=['0', '1'])
parser.add_argument('--quantization', default=False, action='store_true')
parser.add_argument('--path', type=str, default=default_model_path_path)
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=22001)
parser.add_argument('--threaded', type=bool, default=False)
parser.add_argument('--max_input_length', type=int, default=5000)
args = parser.parse_args()

# cuda visible devices
cuda_visible_devices = ','.join(args.d)
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

# set quantization (do not quantization by default)
if args.quantization:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.float16,
        # llm_int8_enable_fp32_cpu_offload=True
    )
else:
    quantization_config = None

# CodeLlama-Python model
pretrained_model_path = args.path
config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_path,
    # model_id,
    # cache_dir=None,
)
# config.pretraining_tp = 1
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_path,
    # model_id=None,
    # config=config,
    quantization_config=quantization_config,
    device_map='auto',
    # cache_dir=None,
    # use_safetensors=False,
    torch_dtype=torch.float16
)

# if not args.quantization:
#     model = model.half()
#     gc.collect()
#     if torch.cuda.device_count() > 0:
#         torch.cuda.empty_cache()

# tokenizer for the LLM
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_path,
)

# Flask API
app = Flask(__name__)
CORS(app)


@app.route(f'/completions', methods=['POST'])
def completions():
    content = request.json
    prompt = content['prompt'].strip()
    prompt = [
        {'role': 'user', 'content': prompt}
    ]
    repeat_prompt = content.get('repeat_prompt', 1)

    # due to the limitations of the GPU devices in the server, the maximum repeat prompt have to be restricted
    max_repeat_prompt = 20
    repeat_prompt = min(max_repeat_prompt, repeat_prompt)

    print(f'========================================== Prompt ==========================================')
    print(f'{prompt}\n')
    print(f'============================================================================================')
    print(f'\n\n')

    max_new_tokens = 512
    temperature = None
    do_sample = True
    top_k = None
    top_p = None
    num_return_sequences = 1
    eos_token_id = 32021
    pad_token_id = 32021

    if 'params' in content:
        params: dict = content.get('params')
        max_new_tokens = params.get('max_new_tokens', max_new_tokens)
        temperature = params.get('temperature', temperature)
        do_sample = params.get('do_sample', do_sample)
        top_k = params.get('top_k', top_k)
        top_p = params.get('top_p', top_p)
        num_return_sequences = params.get('num_return_sequences', num_return_sequences)
        eos_token_id = params.get('eos_token_id', eos_token_id)
        pad_token_id = params.get('pad_token_id', pad_token_id)

    while True:
        inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        )
        inputs = torch.vstack([inputs] * repeat_prompt).to(model.device)

        try:
            # LLM inference
            output = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )
        except torch.cuda.OutOfMemoryError as e:
            # clear cache
            gc.collect()
            if torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
            # decrease repeat_prompt num
            repeat_prompt = max(repeat_prompt // 2, 1)
            continue

        content = []
        for i, out_ in enumerate(output):
            content.append(tokenizer.decode(output[i, len(inputs[i]):], skip_special_tokens=True))

        print(f'======================================== Response Content ========================================')
        print(f'{content}\n')
        print(f'==================================================================================================')
        print(f'\n\n')

        # clear cache
        gc.collect()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

        # Send back the response.
        return jsonify(
            {'content': content}
        )


def trim_messages(messages: List[Dict], max_len):
    total_len = 0
    for i, m in enumerate(messages):
        total_len += len(m['content'])
        if total_len > max_len:
            rest = max(len(m['content']) - (total_len - max_len), 0)
            m['content'] = m['content'][:rest]
            return messages[:i + 1]
    return messages


@app.route(f'/chat/completions', methods=['POST'])
def chat_completions():
    content = request.json
    messages: List[Dict] = content['messages']
    messages = trim_messages(messages, args.max_input_length)

    # due to the limitations of the GPU devices in the server
    # the maximum repeat prompt have to be restricted
    repeat_prompt = content.get('repeat_prompt', 1)
    max_repeat_prompt = 20
    repeat_prompt = min(max_repeat_prompt, repeat_prompt)

    print(f'========================================== Prompt ==========================================')
    print(f'{messages}\n')
    print(f'============================================================================================')
    print(f'\n\n')

    max_new_tokens = 512
    temperature = None
    do_sample = True
    top_k = None
    top_p = None
    num_return_sequences = 1
    eos_token_id = 32021
    pad_token_id = 32021

    if 'params' in content:
        params: dict = content.get('params')
        max_new_tokens = params.get('max_new_tokens', max_new_tokens)
        temperature = params.get('temperature', temperature)
        do_sample = params.get('do_sample', do_sample)
        top_k = params.get('top_k', top_k)
        top_p = params.get('top_p', top_p)
        num_return_sequences = params.get('num_return_sequences', num_return_sequences)
        eos_token_id = params.get('eos_token_id', eos_token_id)
        pad_token_id = params.get('pad_token_id', pad_token_id)

    while True:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
        inputs = torch.vstack([inputs] * repeat_prompt).to(model.device)

        try:
            # LLM inference
            output = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )
        except torch.cuda.OutOfMemoryError as e:
            # clear cache
            gc.collect()
            if torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
            # decrease repeat_prompt num
            repeat_prompt = max(repeat_prompt // 2, 1)
            continue

        content = []
        for i, out_ in enumerate(output):
            content.append(tokenizer.decode(output[i, len(inputs[i]):], skip_special_tokens=True))

        print(f'======================================== Response Content ========================================')
        print(f'{content}\n')
        print(f'==================================================================================================')
        print(f'\n\n')

        # clear cache
        gc.collect()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

        # Send back the response.
        return jsonify(
            {'content': content}
        )


if __name__ == '__main__':
    app.run(host=args.host, port=args.port, threaded=args.threaded)
