# Industrial-Instruction

Industrial-Instruction is an end-to-end framework for constructing industrial instruction data and evaluation benchmarks from real technical PDF reports. The repository provides reproducible pipelines for layout-aware extraction, retrieval index construction, QA synthesis under realistic retrieval conditions, model fine-tuning, and benchmark evaluation.

Using Panasonic technical documentation as a case study, Industrial-Instruction supports research on robust retrieval-augmented generation (RAG) in high-noise, multi-document industrial settings.

## Repository Scope

This repository contains:
- Dataset generation notebooks and scripts
- Vector store construction notebooks and utilities
- Training notebooks/scripts for fine-tuning
- Benchmark/evaluation packages and result-processing code

This repository intentionally excludes heavy artifacts (datasets, model weights, and FAISS index binaries), which are published on Hugging Face.

## Planned Hugging Face Repositories

- Dataset: `industrial-instruction-dataset`
- Model: `industrial-instruction-qwen4b`
- Vector index: `industrial-instruction-faiss`

## Reproducibility Outline

1. Build/prepare source documents and extracted text-table content.
2. Build vector index and retrieval pipeline.
3. Generate and filter QA data under retrieval relationship settings.
4. Fine-tune small open LLMs on Industrial-Instruction training data.
5. Evaluate with and without RAG on held-out benchmark split.

## Citation

If you use this repository, please cite the Industrial-Instruction paper (citation block to be added after publication).
