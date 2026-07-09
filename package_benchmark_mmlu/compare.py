import json
from pathlib import Path

# Map a label to each results.json path
models = {
    "original":     "./result/__home__parsa__panasonic__models__Qwen3-4B-Instruct-2507/results_2026-07-05T13-51-12.477288.json",
    "finetuned_v1_Qwen": "./result/__home__parsa__panasonic__gallery__Qwen3-4B-Instruct-2507-ft-panasonic_qa_v1_train-2x32-sft/results_2026-07-05T13-42-28.914966.json",
    "finetuned_v2_Claude": "./result/__home__parsa__panasonic__gallery__Qwen3-4B-Instruct-2507-ft-panasonic_qa_claude_v1_train-2x32-sft/results_2026-07-08T13-27-07.500037.json",
}

def load_scores(path):
    with open(path) as f:
        data = json.load(f)["results"]
    scores = {}
    for task, metrics in data.items():
        acc = metrics.get("acc,none", metrics.get("acc_norm,none"))
        stderr = metrics.get("acc_stderr,none", metrics.get("acc_norm_stderr,none"))
        if acc is not None:
            scores[task] = (acc, stderr)
    return scores

all_scores = {name: load_scores(path) for name, path in models.items()}

# Union of all task/group names across the 3 runs
all_tasks = sorted(set().union(*[s.keys() for s in all_scores.values()]))

# Print a comparison table
header = f"{'Task':45s} | " + " | ".join(f"{name:14s}" for name in models)
print(header)
print("-" * len(header))

for task in all_tasks:
    row = f"{task:45s} | "
    cells = []
    for name in models:
        val = all_scores[name].get(task)
        cells.append(f"{val[0]*100:6.2f}%      " if val else f"{'--':14s}")
    print(row + " | ".join(cells))
