# Change the json.load[-1] to json.load[0]
# Load dataset
# add more normalize things 
from datasets import load_from_disk
from utils import start_vllm_service
from openai import OpenAI
import json
from simple_transitionV2 import evaluate_transition
from tqdm import tqdm
import os
import signal
from datetime import datetime
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

MAX_CONCURRENCY = 32

# Constant
model_path = "../gallery/Qwen3-4B-Instruct-2507-ft-panasonic_qa_v1_train-2x32-sft"
ds_path = './FailureSensorIQ-v2.0'
ds = load_from_disk(ds_path)

# For RAG
RAG = False
faiss_index_path = "./faiss_panasonic-only-md-no-img-v0.1/index"
faiss_mapping_path = "./faiss_panasonic-only-md-no-img-v0.1/id_map.json"
embed_model_path = "../models/embeddinggemma-300m"

# Generation
MAX_TOKEN = 4096
if RAG:
    from vector_store_module import Retriever
    retriever = Retriever(faiss_index_path, faiss_mapping_path, embed_model_path)

# Load Model
(proc, base_url) = start_vllm_service(model_path, port=8182)

try:
    # Inference
    client = OpenAI(
        base_url=base_url,
        api_key="none"
    )

    
    # Model sees as a chat completion
    def chat_completion(prompt, max_tokens=MAX_TOKEN, temperature=0.0):
        global RAG
        if RAG:
            retrieved_docs = retriever.search(prompt, 3)
            retrieved_str = "\n".join(retrieved_docs)
            prompt_with_external_knowledge = "Based on relevat document answer this question.\n"
            prompt_with_external_knowledge += f"relevant document: {retrieved_str}\n"
            prompt_with_external_knowledge += f"question: {prompt}"
        else:
            prompt_with_external_knowledge = prompt


        msg = [
            {"role": "system", "content": 'You are a helpful assistant. You must output your answer strictly as valid JSON in the format {"answer": ["choice"]}.'},
            {"role": "user", "content": prompt_with_external_knowledge},
        ]

        response = client.chat.completions.create(
            model="test",
            messages=msg,
            max_tokens=max_tokens,
            temperature=temperature,
            # stop=["\n\n"]  # Optional: stop sequences
        )
        return response.choices[0].message.content


    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
    async def async_generate_completion(prompt, max_tokens=MAX_TOKEN, temperature=0.0):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            chat_completion,
            prompt, max_tokens, temperature
        )

    async def process_sample(sample):
        sample = dict(sample)

        llm_res = await async_generate_completion(sample["prompt"])
        sample["model_original_output"] = llm_res
        
        json_blocks = re.findall(r'\{.*?\}', llm_res, flags=re.DOTALL)
        if not json_blocks:
            sample["model_output"] = llm_res
            return sample

        try:
            # Add Normalizer : "" and '' and also save first block
            fixed = json_blocks[0].replace("'", '"')
            sample["model_output"] = json.loads(fixed)["answer"]
        except:
            sample["model_output"] = llm_res

        return sample

    async def process_split(split, ds):
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        tasks = []

        async def worker(sample):
            async with semaphore:
                return await process_sample(sample)

        for sample in ds[split]:
            tasks.append(asyncio.create_task(worker(sample)))

        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{split}"):
            results.append(await future)

        return results

    async def runner():
        ds_split = ['org', 'pert']
        results = {}

        for split in ds_split:
            results[split] = await process_split(split, ds)

        return results


    results = asyncio.run(runner())


    # Date Time
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Only for name
    if RAG:
        rag = "RAG"
    else:
        rag = "NORAG"

    # Save results based on samples
    model_name = f"{(model_path.split("/"))[-1]}-p"
    with open(f"./results_with_table/{model_name}-{timestamp}-samples-{rag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)



    final_results = evaluate_transition(original_records=results['org'], perturbed_records=results['pert'])
    with open(f"./results_with_table/{model_name}-{timestamp}-evaluation-{rag}.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"Saved Evaluation results !")
    print(f"[EVALUATION_RESULT]\n {final_results}")

    # kill model
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    print(f"Killed vllm process")
except KeyboardInterrupt:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    print(f"Killed vllm process")