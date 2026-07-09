# %%
from datasets import load_from_disk
import time


t0 = time.time()
ds_split = ['org', 'pert']
ds_path = './FailureSensorIQ-v2.0'

ds = load_from_disk(ds_path)

# %%
ds.shuffle(seed=42)

# %%
# Should only use one documents
# [1] {<Document 1>}
useless_doc = """
<Documents>
{doc}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. q* should be related to the <Documents>, but the <Documents> can not provide any useful information for answering q*.
2. a* should be able to answer q*, ensuring that the response a* is accurate, detailed, and comprehensive.
3. q* must be a standalone question. Strictly avoid using phrases that reference the source material, such as "Based on the provided context," "According to the documents," "In the text," or similar meta-references.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
IMPORTANT: If the <Simulated Instruction> is a multiple-choice question with options, then your generated q* MUST ALSO include exactly 5 options labeled A–E, and a* MUST include the correct option(s) in the same JSON list format used in the simulated instruction. Always generate 5 options, even if the original document does not contain any option-like content.
Please directly generate the question-answer-options (q*, a*, options*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ..., \"options*\": ...}}, so for a* select A-E correct option.
Ensure the quality of the generated (q*, a*, options*).
"""

# %%
# Should only use one documents
# [1] {<Document 1>}
single_doc_support = """
<Documents>
{doc}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. <Documents> can support q* by providing useful information or hints, but they do not contain explicit answers.
2. a* should use useful information from <Documents> to aid in answering q*, ensuring that the response is accurate,
detailed, and comprehensive.
3. q* must be a standalone question. Strictly avoid using phrases that reference the source material, such as "Based on the provided context," "According to the documents," "In the text," or similar meta-references.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
IMPORTANT: If the <Simulated Instruction> is a multiple-choice question with options, then your generated q* MUST ALSO include exactly 5 options labeled A–E, and a* MUST include the correct option(s) in the same JSON list format used in the simulated instruction. Always generate 5 options, even if the original document does not contain any option-like content.
Please directly generate the question-answer-options (q*, a*, options*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ..., \"options*\": ...}}, so for a* select A-E correct option.
Ensure the quality of the generated (q*, a*, options*).
"""


# %%
# Should retrieved multi docs
# [1] {<Document 1>}
# [2] {<Document 2>}
# [3] ...
multi_doc_support = """
<Documents>
{docs}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. Multiple documents within <Documents> can support q* by providing useful information or hints, but they do not contain
explicit answers.
2. a* should use useful information from <Documents> to aid in answering q*, ensuring that the response is accurate,
detailed, and comprehensive.
3. q* must be a standalone question. Strictly avoid using phrases that reference the source material, such as "Based on the provided context," "According to the documents," "In the text," or similar meta-references.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
IMPORTANT: If the <Simulated Instruction> is a multiple-choice question with options, then your generated q* MUST ALSO include exactly 5 options labeled A–E, and a* MUST include the correct option(s) in the same JSON list format used in the simulated instruction. Always generate 5 options, even if the original document does not contain any option-like content.
Please directly generate the question-answer-options (q*, a*, options*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ..., \"options*\": ...}}, so for a* select A-E correct option.
Ensure the quality of the generated (q*, a*, options*).
"""


# %%
# Should only use one documents
# [1] {<Document 1>}
single_doc_answer= """
<Documents>
{doc}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. Ensure that q* can be answered directly using the content of <Documents>, meaning its answer can be fully derived from
<Documents>.
2. a* should use the information from <Documents> to answer q* accurately, ensuring that the response is accurate, detailed,
and comprehensive.
3. q* must be a standalone question. Strictly avoid using phrases that reference the source material, such as "Based on the provided context," "According to the documents," "In the text," or similar meta-references.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
IMPORTANT: If the <Simulated Instruction> is a multiple-choice question with options, then your generated q* MUST ALSO include exactly 5 options labeled A–E, and a* MUST include the correct option(s) in the same JSON list format used in the simulated instruction. Always generate 5 options, even if the original document does not contain any option-like content.
Please directly generate the question-answer-options (q*, a*, options*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ..., \"options*\": ...}}, so for a* select A-E correct option.
Ensure the quality of the generated (q*, a*, options*).
"""

# %%
# Should retrieved multi docs
# [1] {<Document 1>}
# [2] {<Document 2>}
# [3] ...
multi_doc_answer = """
<Documents>
{docs}
</Documents>
Your task is to generate an English question q* and a corresponding response a* based on the provided <Documents>. Please
note that the question q* can take various forms, not limited to questions with a question mark, but also including statements,
instructions, and other formats. You need to follow the requirements below to generate the q* and a* (RAG Paradigms):
1. The answer to q* can be derived from multiple documents within <Documents>, involving multi-hop reasoning or the
integration of information from several documents.
2. a* should leverage the information in <Documents> to provide an accurate answer to q*, ensuring that the response is
accurate, detailed, and comprehensive.
3. q* must be a standalone question. Strictly avoid using phrases that reference the source material, such as "Based on the provided context," "According to the documents," "In the text," or similar meta-references.
Additionally, to ensure diversity, richness, and high quality in the question q* you generate, we will randomly provide a
question for you to emulate. In other words, while satisfying the requirements above, make q* similar in task requirement
and expression to the <Simulated Instruction> below:
<Simulated Instruction>
{simulated_instruction}
</Simulated Instruction>
IMPORTANT: If the <Simulated Instruction> is a multiple-choice question with options, then your generated q* MUST ALSO include exactly 5 options labeled A–E, and a* MUST include the correct option(s) in the same JSON list format used in the simulated instruction. Always generate 5 options, even if the original document does not contain any option-like content.
Please directly generate the question-answer-options (q*, a*, options*) following all the rules above in the format of {{\"q*\": ..., \"a*\": ..., \"options*\": ...}}, so for a* select A-E correct option.
Ensure the quality of the generated (q*, a*, options*).
"""

# %%
from vector_store_module import Retriever

retriever = Retriever(index_path="./faiss_panasonic-only-md-no-img-v0.1/index",
 mapping_path="./faiss_panasonic-only-md-no-img-v0.1/id_map.json",
 embed_model_path="../models/embeddinggemma-300m")

# %%
ds['org'][0]


# %%
import json
import anthropic
import time


client = anthropic.Anthropic(
    api_key="",
    base_url="",
)
def chat_completion(prompt, temperature=0.1, max_tokens=8192, max_retries=3):
    msg = [
        {"role": "system", "content": "You are a helpful assistant. Follow exactly the Instructions."},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-opus-4-6",
                messages=msg,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.content[0].text
            
        except anthropic.RateLimitError as e:
            # Wait longer on each retry: 5s, then 10s, then 15s
            wait_time = (attempt + 1) * 5
            print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
            
        except anthropic.APIError as e:
            print(f"API Error: {e}. Retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(5)
            
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    # If it fails after all retries, return a string that will trigger your JSON except block
    print("Max retries reached. Failing this request.")
    return "API_FAILURE"

def format_llm_response(qa):
    json.loads(qa)


# %%
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

splits = ["org", "pert"]
r0_samples = []
r1_samples = []
r2_samples = []
r3_samples = []
r4_samples = []
faild_sample = []

def process_sample(sample):
    """Process a single sample for all retrieval methods"""
    prompt = sample['prompt']
    retireved_docs = retriever.search(query=prompt, k=3)
    results = {}
    
    # r0
    retireved_r0 = retireved_docs[0]
    r0_prompt = useless_doc.format(doc=retireved_r0, simulated_instruction=prompt)
    r0_results = chat_completion(r0_prompt)
    try:
        r0_dic = json.loads(r0_results)
        r0_dic["documents"] = [retireved_r0]
        results['r0'] = r0_dic
    except Exception as e:
        print(f"r0 is not a json model output: {r0_results}")
        return 'failed', sample
    
    # r1
    retireved_r1 = retireved_docs[0]
    r1_prompt = single_doc_support.format(doc=retireved_r1, simulated_instruction=prompt)
    r1_results = chat_completion(r1_prompt)
    try:
        r1_dic = json.loads(r1_results)
        r1_dic["documents"] = [retireved_r1]
        results['r1'] = r1_dic
    except Exception as e:
        print(f"r1 is not a json model output: {r1_results}")
        return 'failed', sample
    
    # r2
    retireved_r2 = retireved_docs
    r2_prompt = multi_doc_support.format(docs=retireved_r2, simulated_instruction=prompt)
    r2_results = chat_completion(r2_prompt)
    try:
        r2_dic = json.loads(r2_results)
        r2_dic["documents"] = retireved_r2
        results['r2'] = r2_dic
    except Exception as e:
        print(f"r2 is not a json model output: {r2_results}")
        return 'failed', sample
    
    # r3
    retireved_r3 = retireved_docs[0]
    r3_prompt = single_doc_answer.format(doc=retireved_r3, simulated_instruction=prompt)
    r3_results = chat_completion(r3_prompt)
    try:
        r3_dic = json.loads(r3_results)
        r3_dic["documents"] = [retireved_r3]
        results['r3'] = r3_dic
    except Exception as e:
        print(f"r3 is not a json model output: {r3_results}")
        return 'failed', sample
    
    # r4
    retireved_r4 = retireved_docs
    r4_prompt = multi_doc_answer.format(docs=retireved_r4, simulated_instruction=prompt)
    r4_results = chat_completion(r4_prompt)
    try:
        r4_dic = json.loads(r4_results)
        r4_dic["documents"] = retireved_r4
        results['r4'] = r4_dic
    except Exception as e:
        print(f"r4 is not a json model output: {r4_results}")
        return 'failed', sample
    
    return 'success', results

# Collect all samples to process
all_samples = []
for split in splits:
    for sample in ds[split]:
        all_samples.append(sample)

# Process samples in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    # Submit all tasks
    futures = {executor.submit(process_sample, sample): sample for sample in all_samples}
    
    # Process results as they complete
    for future in tqdm(as_completed(futures), total=len(all_samples), desc="Processing"):
        status, data = future.result()
        if status == 'failed':
            faild_sample.append(data)
        else:
            r0_samples.append(data['r0'])
            r1_samples.append(data['r1'])
            r2_samples.append(data['r2'])
            r3_samples.append(data['r3'])
            r4_samples.append(data['r4'])

# %%
with open("r0_samples.json", "w", encoding="utf-8") as f:
    json.dump(r0_samples, f, ensure_ascii=False, indent=2)

with open("r1_samples.json", "w", encoding="utf-8") as f:
    json.dump(r1_samples, f, ensure_ascii=False, indent=2)

with open("r2_samples.json", "w", encoding="utf-8") as f:
    json.dump(r2_samples, f, ensure_ascii=False, indent=2)

with open("r3_samples.json", "w", encoding="utf-8") as f:
    json.dump(r3_samples, f, ensure_ascii=False, indent=2)

with open("r4_samples.json", "w", encoding="utf-8") as f:
    json.dump(r4_samples, f, ensure_ascii=False, indent=2)

with open("faild_samples.json", "w", encoding="utf-8") as f:
    json.dump(faild_sample, f, ensure_ascii=False, indent=2)

print(f"Process Completed, Processing time with num_worker=42 is {time.time() - t0} seconds.")




