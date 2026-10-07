"""Benchmark a served model on IBM's FailureSensorIQ, the paper's test splits,
this project's own test split, or a custom multiple-choice set.

    vllm serve my-org/my-model --served-model-name my-model --port 8000
    ii bench --suite ibm --suite paper-claude --context none --context gold \\
             --base-url http://localhost:8000/v1
"""

from industrial_instruction.benchmark.parsing import normalize_label, parse_answer
from industrial_instruction.benchmark.runner import (
    ContextBuilder,
    Endpoint,
    format_table,
    run_benchmarks,
)
from industrial_instruction.benchmark.suites import (
    BenchItem,
    available_suites,
    load_suite,
    register_suite,
)

__all__ = [
    "BenchItem",
    "ContextBuilder",
    "Endpoint",
    "available_suites",
    "format_table",
    "load_suite",
    "normalize_label",
    "parse_answer",
    "register_suite",
    "run_benchmarks",
]
