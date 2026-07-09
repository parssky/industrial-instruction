# save as reshard_model.py
import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer

src = "/home/parsa/panasonic/Industrial-Instruction/models/Qwen3-4B-Instruct-2507-ft-panasonic_qa_v1_train-2x32-sft"
dst = "/home/parsa/panasonic/Industrial-Instruction/models/Qwen3-4B-Instruct-2507-ft-panasonic_qa_v1_train-2x32-sft-900mb"

os.makedirs(dst, exist_ok=True)

tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    src,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)

model.save_pretrained(dst, safe_serialization=True, max_shard_size="900MB")
tok.save_pretrained(dst)

for f in ["chat_template.jinja", "README.md", "generation_config.json"]:
    p = os.path.join(src, f)
    if os.path.exists(p):
        shutil.copy2(p, os.path.join(dst, f))

print("Done:", dst)
