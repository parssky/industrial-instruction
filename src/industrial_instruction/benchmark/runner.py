"""Run benchmark suites against a served model.

The model is whatever the user serves (``vllm serve ...``); this module only
talks to its OpenAI-compatible endpoint. Every reply is kept next to its
parsed answer, so a surprising score can be traced to individual items.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from industrial_instruction.benchmark.metrics import ibm_metrics, set_metrics
from industrial_instruction.benchmark.parsing import parse_answer
from industrial_instruction.benchmark.suites import BenchItem, load_suite
from industrial_instruction.config import Config, EndpointConfig
from industrial_instruction.utils.io import ensure_dir, write_json
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


class Endpoint:
    """Chat client for the model under test."""

    def __init__(self, config: EndpointConfig, client=None) -> None:
        self.config = config
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=config.base_url,
                api_key=config.resolve_api_key(),
                timeout=config.timeout,
                max_retries=0,
            )
        self.client = client
        self.model = config.model or self._served_model()

    def _served_model(self) -> str:
        try:
            models = self.client.models.list().data
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"cannot reach the model server at {self.config.base_url}: {exc}. "
                "Start it first, e.g. vllm serve <model> --port 8000"
            ) from exc
        if not models:
            raise RuntimeError(f"{self.config.base_url} serves no models")
        return models[0].id

    def ask(self, user: str) -> str:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": user})
        last: Optional[Exception] = None
        for attempt in range(1, max(self.config.max_retries, 1) + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < self.config.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"request failed after {self.config.max_retries} attempts: {last}")


class ContextBuilder:
    """Puts documents (none / gold / retrieved) in front of a question."""

    def __init__(self, config: Config, store=None) -> None:
        self.config = config
        self.template = config.benchmark.context_template
        self._store = store

    @property
    def store(self):
        if self._store is None:
            from industrial_instruction.store.runner import load_store

            self._store = load_store(self.config)
        return self._store

    def documents(self, item: BenchItem, mode: str) -> Optional[List[str]]:
        if mode == "none":
            return None
        if mode == "gold":
            return item.documents
        hits = self.store.search(item.query, k=self.config.benchmark.retrieval_k)
        return [h.chunk.as_context() for h in hits]

    def prompt(self, item: BenchItem, mode: str) -> str:
        docs = self.documents(item, mode)
        if docs is None:
            return item.prompt
        # replace(), not format(): documents contain braces (JSON, tables).
        return self.template.replace("{documents}", "\n".join(docs)).replace(
            "{question}", item.prompt
        )


def score_items(
    items: Sequence[BenchItem], endpoint: Endpoint, context: ContextBuilder, mode: str,
    workers: int = 8, desc: str = "",
) -> List[Dict[str, Any]]:
    """Ask every item; returns one row per item with reply and parsed answer."""

    def one(item: BenchItem) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "id": item.id,
            "group": item.group,
            "pair": list(item.pair),
            "gold": item.gold,
            "labels": item.labels,
        }
        try:
            prompt = context.prompt(item, mode)
            reply = endpoint.ask(prompt)
            row["reply"] = reply
            row["pred"] = parse_answer(reply, valid=item.labels)
        except Exception as exc:  # noqa: BLE001 - one item never stops the run
            row["reply"], row["pred"], row["error"] = "", None, repr(exc)[:500]
        row["correct"] = row["pred"] is not None and set(row["pred"]) == set(item.gold)
        if item.extra.get("raw_option_ids"):
            row["legacy_correct"] = _legacy_ibm_correct(row, item)
        return row

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        return list(tqdm(pool.map(one, items), total=len(items), desc=desc, unit="q"))


def _legacy_ibm_correct(row: dict, item: BenchItem) -> bool:
    """The original IBM script's rule, for comparison with earlier results.

    It tested ``option_id in model_output`` with raw ids (``"P)"``) and, if
    the reply didn't parse, against the raw text - so perturbed items were
    never correct and stray letters in prose could match.
    """
    blocks = re.findall(r"\{.*?\}", row.get("reply") or "", flags=re.S)
    output: Any = row.get("reply") or ""
    if blocks:
        try:
            output = json.loads(blocks[0].replace("'", '"'))["answer"]
        except Exception:  # noqa: BLE001
            pass
    hits = [oid in output for oid in item.extra["raw_option_ids"]]
    return hits == list(item.extra["correct"])


def summarize(suite: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = sum(1 for r in rows if r.get("error"))
    if suite == "ibm":
        summary = ibm_metrics(
            [dict(r, pair=tuple(r["pair"])) for r in rows]
        )
        legacy = [r for r in rows if "legacy_correct" in r]
        if legacy:
            org = [r["legacy_correct"] for r in legacy if r["group"] == "org"]
            pert = [r["legacy_correct"] for r in legacy if r["group"] == "pert"]
            summary["legacy"] = {
                "acc_original": sum(org) / len(org) if org else None,
                "acc_perturb": sum(pert) / len(pert) if pert else None,
                "note": "original IBM script scoring; perturbed items cannot match",
            }
    else:
        summary = set_metrics(rows)
    summary["errors"] = errors
    return summary


def run_benchmarks(
    config: Config,
    suites: Optional[Sequence[str]] = None,
    contexts: Optional[Sequence[str]] = None,
    endpoint: Optional[Endpoint] = None,
    store=None,
) -> Dict[str, Any]:
    """Run every (suite, context) pair; write samples and a summary."""
    cfg = config.benchmark
    suites = list(suites or cfg.suites)
    contexts = list(contexts or cfg.contexts)
    endpoint = endpoint or Endpoint(cfg.endpoint)
    builder = ContextBuilder(config, store=store)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", endpoint.model).strip("_")[-80:]
    out_dir = Path(cfg.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(config.paths.root) / out_dir
    out_dir = ensure_dir(out_dir / slug / stamp)

    results: Dict[str, Any] = {}
    for suite in suites:
        items = load_suite(suite, config)
        logger.info("bench: %s - %d items", suite, len(items))
        for mode in contexts:
            if mode == "gold" and not any(i.documents for i in items):
                logger.warning("bench: %s has no gold documents; skipping context=gold", suite)
                continue
            rows = score_items(
                items, endpoint, builder, mode,
                workers=cfg.endpoint.max_workers, desc=f"{suite}[{mode}]",
            )
            name = f"{suite}-{mode}"
            with open(out_dir / f"{name}.samples.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            results[name] = summarize(suite, rows)
            if results[name]["errors"]:
                logger.warning(
                    "bench: %s had %d request errors (scored as wrong); see %s",
                    name, results[name]["errors"], out_dir / f"{name}.samples.jsonl",
                )

    summary = {
        "model": endpoint.model,
        "base_url": cfg.endpoint.base_url,
        "created_at": stamp,
        "config": cfg.model_dump(mode="json", exclude={"endpoint": {"api_key_env"}}),
        "results": results,
    }
    write_json(out_dir / "summary.json", summary)
    summary["output_dir"] = str(out_dir)
    return summary


def format_table(summary: Dict[str, Any]) -> str:
    """Compact text table of a run summary."""
    lines = [f"model: {summary['model']}"]
    for name, res in summary["results"].items():
        if "acc_original" in res:
            lines.append(
                f"  {name:<24} acc_org {res['acc_original']:.1%}  acc_pert {res['acc_perturb']:.1%}"
                f"  consistency {res['consistency']:.1%}  (n={res['n_pairs']} pairs)"
            )
        elif res.get("n"):
            lines.append(
                f"  {name:<24} set-match {res['set_match']:.1%}  F1 {res['f1']:.1%}"
                f"  jaccard {res['jaccard']:.1%}  unparsed {res['parse_failures']:.1%}  (n={res['n']})"
            )
        else:
            lines.append(f"  {name:<24} no items")
    return "\n".join(lines)
