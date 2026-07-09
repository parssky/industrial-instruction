# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# %%
rag_instruct_dataset_path = "./hf-dataset/RAG-Instruct"
pana_dataset_path = "./hf-dataset/panasonic_qa_claude_v1_train"
model_path = "./models/Qwen3-4B-Instruct-2507"
tokenizer_path = "./models/Qwen3-4B-Instruct-2507"
gallery_path = "./gallery"
device = "cuda"
batch_train = 2
batch_eval = 2
grad_accumulation = 32
max_seq_len = 4096

# %%
from datasets import load_from_disk, concatenate_datasets, DatasetDict, load_dataset


ds = load_from_disk(pana_dataset_path)

split_ds = ds.train_test_split(test_size=0.05, seed=42)

dataset = DatasetDict({
    'train': split_ds['train'],
    'test': split_ds['test']
})
dataset

# %%
from unsloth import FastLanguageModel
import torch
dtype = None
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_len,
    dtype = dtype,
    device_map = "balanced",
    load_in_4bit = load_in_4bit,
    full_finetuning = True,
    # float32_mixed_precision = True
)

# %%
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen3-instruct"
)

def normalize_documents(documents):
    flat_docs = []
    for doc in documents:
        if isinstance(doc, list):
            flat_docs.append(" ".join(map(str, doc)))
        elif isinstance(doc, dict):
            flat_docs.append(" ".join(f"{k}: {v}" for k, v in doc.items()))
        else:
            flat_docs.append(str(doc))
    return flat_docs

def formatting_prompts_func(example):
    question = example["question"]
    documents = normalize_documents(example["documents"])
    answer = str(example["answer"])
    prompt = """
        Based on relevat document answer this question.
        relevant document: {}
        question: {}
    """
    input = prompt.format("\n".join(documents), question)

    texts = tokenizer.apply_chat_template(
        [
            {"role":"user", "content": input},
            {"role":"assistant", "content": answer}
        ],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : texts, }

# %%
# Map the dataset on function
# Here may face the error ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute
# This error is because this model is not instruct so in tokenizer_config.json there is no "chat_template". So we can find another chat template from instruct version and add it to this model.
dataset['train'] = dataset['train'].map(formatting_prompts_func)
dataset['test'] = dataset['test'].map(formatting_prompts_func)

# %%
dataset['train'][100]['text']

# %%
# # TODO: Complete the compute metrics based on evaluation
# # Check : https://huggingface.co/docs/evaluate/index
# import numpy as np
# import evaluate
# import random

# exact_match_metric = evaluate.load("exact_match")

# def compute_metrics(eval_preds):
#     logits, labels = eval_preds

#     # Move tensors to CPU and get argmax safely
#     preds = tokenizer.batch_decode(
#         np.where(labels != tokenizer.pad_token_id, logits.argmax(-1), tokenizer.pad_token_id),
#         skip_special_tokens=True
#     )
#     refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Strip whitespace
#     pred_str = [p.strip() for p in preds]
#     label_str = [l.strip() for l in refs]

#     # Compute exact match
#     exact_match = exact_match_metric.compute(
#         predictions=pred_str,
#         references=label_str
#     )

#     # Pick 2 random samples
#     indices = random.sample(range(len(pred_str)), k=min(2, len(pred_str)))
#     examples_shown = []
#     for i in indices:
#         print(f"pred: {pred_str[i]}")
#         print(f"ref : {label_str[i]}")
#         print("-"*20)
#         examples_shown.append({"pred": pred_str[i], "ref": label_str[i]})

#     # Return metrics
#     return {
#         "exact_match": exact_match["exact_match"],
#         "examples_shown": examples_shown
#     }


# %%
from trl import SFTConfig
num_of_reports = 32

model_name = model_path.split('/')[-1]
dataset_name = pana_dataset_path.split('/')[-1]


SAVE_EVAL_LOG_STEPS = 1 / num_of_reports

ft_model_id = f'{gallery_path}/{model_name}-ft-{dataset_name}-{batch_train}x{grad_accumulation}-sft'

args = SFTConfig(
    ft_model_id,
    run_name = ft_model_id,
    dataset_text_field = "text",

    per_device_train_batch_size=batch_train,
    per_device_eval_batch_size=batch_eval,
    gradient_accumulation_steps=grad_accumulation,

    num_train_epochs=3,
    learning_rate=5e-5,
    save_total_limit = 2,

    eval_strategy='steps',
    save_strategy='steps',
    logging_strategy='steps',

    save_steps=SAVE_EVAL_LOG_STEPS,
    eval_steps=SAVE_EVAL_LOG_STEPS,
    logging_steps=SAVE_EVAL_LOG_STEPS,

    
    # weight_decay = 0.001,
    lr_scheduler_type = "linear",
    report_to='tensorboard',
    seed=42,
    eval_on_start = True,
    torch_compile=False,
    torch_compile_backend=None,
)

# %%
from trl import SFTTrainer


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    # compute_metrics = compute_metrics,
    args = args,
)

# %%
stats = trainer.train()

# %%
trainer.model.save_pretrained(f"{ft_model_id}")
trainer.tokenizer.save_pretrained(f"{ft_model_id}")


