# Evaluation


activate eval end on conda: 
```bash
conda activate eval
```

then install package:

```bash
pip install lm-eval
```

then use commands :
```bash
lm_eval   --model hf   --model_args pretrained=/home/parsa/panasonic/gallery/Qwen3-4B-Instruct-2507-ft-panasonic_qa_claude_v1_train-2x32-sft   --tasks mmlu   --num_fewshot 5   --batch_size auto   --output_path ./result   --log_samples   --device cuda:0   --apply_chat_template   --fewshot_as_multiturn
```


