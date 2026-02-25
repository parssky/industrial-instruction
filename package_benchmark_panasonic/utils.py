import subprocess
import os
import requests
import time

        

def start_vllm_service(model_path: str, port: int= 8182):
    os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    vllm_args = ["vllm", "serve", model_path, "--port", str(port), "--gpu_memory_utilization", str(0.95), "--served-model-name", "test", "--max_model_len", "8196"]
    proc = subprocess.Popen(vllm_args, preexec_fn=os.setsid)
    base_url = f'http://127.0.0.1:{port}/v1/'
    while True:
        try:
            response = requests.get(f'{base_url}models')
            if response.status_code == 200:
                break
        except:
            pass
        print("Waiting for vllm to launch. Retrying in 10 seconds")
        time.sleep(10)
    
    print(f"Model {model_path} is on {base_url}")
    return (proc, base_url)

def start_vllm_servicev3(model_path: str, lora_path: str, port: int= 8182):
    os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    vllm_args = ["vllm", "serve", model_path, "--port", str(port), "--gpu_memory_utilization", str(0.95), "--served-model-name", "test", "--max_model_len", "8196", "--enable-lora","--max-lora-rank", "32", "--lora-modules", f"my_lora={lora_path}/adapters"]
    proc = subprocess.Popen(vllm_args, preexec_fn=os.setsid)
    base_url = f'http://127.0.0.1:{port}/v1/'
    while True:
        try:
            response = requests.get(f'{base_url}models')
            if response.status_code == 200:
                break
        except:
            pass
        print("Waiting for vllm to launch. Retrying in 10 seconds")
        time.sleep(10)
    
    print(f"Model {model_path} is on {base_url}")
    return (proc, base_url)

def start_vllm_serviceV2(model_path: str, port: int= 8182):
    os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    vllm_args = ["vllm", "serve", model_path, "--port", str(port), "--gpu_memory_utilization", str(0.95), "--served-model-name", "test", "--max_model_len", "4096", "--quantization", "bitsandbytes"]
    proc = subprocess.Popen(vllm_args, preexec_fn=os.setsid)
    base_url = f'http://127.0.0.1:{port}/v1/'
    while True:
        try:
            response = requests.get(f'{base_url}models')
            if response.status_code == 200:
                break
        except:
            pass
        print("Waiting for vllm to launch. Retrying in 10 seconds")
        time.sleep(10)
    
    print(f"Model {model_path} is on {base_url}")
    return (proc, base_url)