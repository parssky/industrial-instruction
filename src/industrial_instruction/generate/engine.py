"""Generation engine.

One loop over ``(seed x relation)`` pairs replaces the five copy-pasted
blocks in the original ``create_dataset_part_one.py``. Two behavioural fixes
matter most:

1. **Per-relation isolation.** Previously one unparseable reply discarded all
   five samples for that seed. Now each relation succeeds or fails on its own
   and failures are written to ``rejects.jsonl`` with the raw output, so they
   can be inspected and re-run.
2. **Retrieval happens once per seed** and the candidate pool is sliced per
   relation, instead of re-searching the index five times.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from industrial_instruction.config import Config
from industrial_instruction.generate.client import LLMClient, LLMError
from industrial_instruction.generate.prompt_loader import (
    OUTPUT_RULE_MCQ,
    OUTPUT_RULE_QA,
    PromptLibrary,
    format_documents,
    options_rule,
)
from industrial_instruction.generate.seeds import load_seeds
from industrial_instruction.schemas import (
    DocMode,
    QASample,
    RelationSpec,
    SampleStatus,
    Seed,
    StageReport,
)
from industrial_instruction.store.faiss_store import FaissStore
from industrial_instruction.utils.io import append_jsonl, ensure_dir, stable_id
from industrial_instruction.utils.logging import get_logger

logger = get_logger(__name__)


class GenerationEngine:
    """Turns seeds + retrieved context into :class:`QASample` rows."""

    def __init__(
        self,
        config: Config,
        store: Optional[FaissStore] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.store = store
        self.client = client or LLMClient(config.generate)
        self.library = PromptLibrary(config.generate.prompt_dir)

    # ------------------------------------------------------------------

    def _ensure_store(self) -> FaissStore:
        if self.store is None:
            from industrial_instruction.store.runner import load_store

            self.store = load_store(self.config)
        return self.store

    def _pool_size(self, relations: Sequence[RelationSpec]) -> int:
        needed = max((r.k_docs for r in relations), default=1)
        return max(self.config.generate.retrieval_k, needed)

    def build_prompt(
        self, relation: RelationSpec, contexts: Sequence[str], seed_text: str
    ) -> str:
        wants_options = relation.requires_options or self.config.generate.require_options
        return self.library.render(
            relation.prompt,
            docs=format_documents(contexts),
            doc=format_documents(contexts, numbered=False),
            simulated_instruction=seed_text,
            options_rule=(
                options_rule(self.config.generate.n_options) if wants_options else ""
            ),
            output_rule=OUTPUT_RULE_MCQ if wants_options else OUTPUT_RULE_QA,
        )

    def generate_one(
        self, relation: RelationSpec, seed: Seed, hits: Sequence
    ) -> QASample:
        """Generate a single sample. Never raises; failures become rows."""
        k = relation.k_docs if relation.doc_mode == DocMode.MULTI else 1
        selected = list(hits[:k])
        contexts = [h.chunk.as_context() for h in selected]
        base = {
            "relation": relation.id,
            "seed": seed.text,
            "seed_id": seed.id,
            "generator": self.config.generate.model,
            "documents": contexts,
            "doc_ids": [h.chunk.id for h in selected],
            "meta": {
                "source_doc_ids": [h.chunk.doc_id for h in selected],
                "retrieval_scores": [round(h.score, 4) for h in selected],
                "prompt_template": relation.prompt,
                "prompt_sha": self.library.fingerprint(relation.prompt),
            },
        }
        sample_id = stable_id(relation.id, seed.id, *[h.chunk.id for h in selected])

        if not selected:
            return QASample(
                id=sample_id,
                question="",
                status=SampleStatus.ERROR,
                reject_reason="no documents retrieved",
                **base,
            )

        prompt = self.build_prompt(relation, contexts, seed.text)
        try:
            payload, raw = self.client.complete_json(prompt)
        except (LLMError, ValueError) as exc:
            return QASample(
                id=sample_id,
                question="",
                status=SampleStatus.ERROR,
                reject_reason=str(exc)[:500],
                raw_output="",
                **base,
            )

        sample = QASample.from_llm_dict(payload, id=sample_id, **base)
        if not sample.question:
            sample.status = SampleStatus.INVALID
            sample.reject_reason = "missing question (q*)"
            sample.raw_output = raw[:2000]
        return sample

    def generate_for_seed(
        self, seed: Seed, relations: Sequence[RelationSpec]
    ) -> List[QASample]:
        """All relations for one seed, sharing a single retrieval call."""
        store = self._ensure_store()
        hits = store.search(seed.text, k=self._pool_size(relations))
        return [self.generate_one(relation, seed, hits) for relation in relations]


# ----------------------------------------------------------------------


def generate_samples(
    config: Config,
    seeds: Optional[Sequence[Seed]] = None,
    relations: Optional[Sequence[RelationSpec]] = None,
    output_dir: Optional[str] = None,
) -> StageReport:
    """Run the generation stage and write ``r*.jsonl`` plus ``rejects.jsonl``."""
    t0 = time.time()
    relations = list(relations or config.generate.enabled_relations())
    if not relations:
        raise ValueError("No enabled relations in generate.relations")

    seed_list = list(seeds) if seeds is not None else load_seeds(config)
    if not seed_list:
        raise RuntimeError(
            "No seeds loaded. Check seeds.source / seeds.path, or use "
            "seeds.source='self' to bootstrap from your own documents."
        )
    if config.generate.limit:
        seed_list = seed_list[: int(config.generate.limit)]

    out = Path(output_dir).resolve() if output_dir else config.paths.resolve("generated")
    ensure_dir(out)
    for relation in relations:
        target = out / f"{relation.id}.jsonl"
        if target.exists():
            target.unlink()
    rejects_path = out / "rejects.jsonl"
    if rejects_path.exists():
        rejects_path.unlink()

    engine = GenerationEngine(config)
    engine._ensure_store()  # fail fast on a missing/mismatched index

    counts: Dict[str, int] = {r.id: 0 for r in relations}
    n_rejects = 0
    n_errors = 0
    workers = max(config.generate.max_workers, 1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(engine.generate_for_seed, seed, relations): seed
            for seed in seed_list
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"generate[{len(relations)} relations]",
            unit="seed",
        ):
            seed = futures[future]
            try:
                samples = future.result()
            except Exception as exc:  # noqa: BLE001 - keep the run alive
                logger.warning("seed %s failed entirely: %s", seed.id, exc)
                n_errors += 1
                append_jsonl(
                    rejects_path,
                    [{"seed_id": seed.id, "seed": seed.text, "error": repr(exc)}],
                )
                continue
            for sample in samples:
                if sample.status == SampleStatus.OK:
                    append_jsonl(out / f"{sample.relation}.jsonl", [sample])
                    counts[sample.relation] += 1
                else:
                    append_jsonl(rejects_path, [sample])
                    n_rejects += 1
                    if sample.status == SampleStatus.ERROR:
                        n_errors += 1

    total = sum(counts.values())
    report = StageReport(
        stage="generate",
        inputs=len(seed_list),
        outputs=total,
        rejected=n_rejects,
        errors=n_errors,
        seconds=round(time.time() - t0, 2),
        details={
            "output_dir": str(out),
            "per_relation": counts,
            "model": config.generate.model,
            "base_url": config.generate.base_url or "default",
            "workers": workers,
        },
    )
    logger.info(
        "generate: %d samples from %d seeds across %d relations (%d rejected) in %.1fs",
        total,
        len(seed_list),
        len(relations),
        n_rejects,
        report.seconds,
    )
    return report
