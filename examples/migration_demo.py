"""End-to-end migration, with no dependencies and no network.

    python examples/migration_demo.py

Walks a corpus from one embedding space to another the way you would in production:
backfill while the old space keeps serving, measure how differently the new space
retrieves, cut over atomically, then roll back to prove it is reversible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veccore.embedders import HashEmbedder
from veccore.engine import VecCore
from veccore.models import Document
from veccore.similarity import cosine

CORPUS = [
    ("oom", "A container killed for exceeding its memory limit exits with code 137. The "
            "kernel OOM killer terminates it immediately, with no graceful shutdown, so "
            "in-flight requests are dropped."),
    ("crash", "CrashLoopBackOff means the container starts and exits repeatedly. Check the "
              "exit code first: 127 means the entrypoint binary was not found, 0 means the "
              "process finished successfully and was restarted anyway."),
    ("probe", "A readiness probe pointing at the wrong port keeps a healthy pod out of the "
              "Service endpoints. The process is fine; the kubelet is knocking on a door "
              "nothing is listening behind."),
    ("sched", "FailedScheduling with Insufficient cpu means no node has enough allocatable "
              "CPU for the pod's requests. The scheduler compares requests, not usage, so a "
              "cluster that looks idle can still be unschedulable."),
    ("tflock", "Terraform state locking prevents two applies running at once. DynamoDB "
               "provides the lock for the S3 backend, and a stale lock can be cleared with "
               "force-unlock once you are certain no apply is running."),
    ("irsa", "IRSA lets a pod assume an AWS IAM role through a projected service account "
             "token, so no long-lived AWS credential is ever stored in the cluster."),
    ("pasta", "Fresh pasta needs eggs, flour, and a long rest before rolling. Rushing the "
              "rest makes the dough tear when it goes through the machine."),
]

QUERIES = [
    "why was my container killed",
    "pod is running but gets no traffic",
    "terraform apply is stuck on a lock",
    "give a pod aws permissions without secrets",
    "not enough capacity to place the pod",
]

RULE = "-" * 78


def h(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> int:
    h("0.  The problem, in one number")
    a = HashEmbedder("demo-a", 512, "a", word_ngram=1)
    b = HashEmbedder("demo-b", 512, "a", word_ngram=1, char_ngram=3)
    text = "container killed for exceeding its memory limit"
    va, vb = a.embed([text])[0], b.embed([text])[0]
    print(f"  Same text, two models, same dimension ({a.space.dimension}).")
    print(f"    space A fingerprint : {a.space.fingerprint}")
    print(f"    space B fingerprint : {b.space.fingerprint}")
    print(f"    compatible          : {a.space.compatible_with(b.space)}")
    print(f"\n  cosine(A(text), B(text)) = {cosine(va, vb):+.3f}")
    print("  Nothing raises. That number looks like a score and means nothing.")
    print("  VecCore makes this an error at the boundary instead of quiet drift.")

    h("1.  Serve traffic from words-v1")
    core = VecCore()
    core.add_space(HashEmbedder("words-v1", 512, "a", word_ngram=1), activate=True)
    for doc_id, body in CORPUS:
        core.ingest(Document(id=doc_id, text=body))
    print(f"  {core.store.document_count()} documents, {core.store.chunk_count()} chunks, "
          f"active space = {core.registry.active.space.id}")
    for hit in core.query(QUERIES[0], k=2):
        print(f"    {hit.score:.3f}  {hit.chunk_id:<12} {hit.text[:52]}...")

    h("2.  Backfill subword-v2 -- in batches, while v1 keeps serving")
    core.add_space(HashEmbedder("subword-v2", 512, "a", word_ngram=1, char_ngram=3))
    m = core.migrate_to("subword-v2")

    partial = m.backfill(batch_size=3, max_batches=1)
    print(f"  batch 1: embedded {partial.embedded}, {partial.remaining} remaining "
          f"-> state = {core.registry.get('subword-v2').state.value}")
    print("  reads still served by words-v1:", [x.chunk_id for x in core.query(QUERIES[2], k=2)])

    try:
        core.query("anything", space_id="subword-v2")
    except Exception as exc:
        print(f"  querying the incomplete space is refused:\n    {str(exc)[:88]}...")

    done = m.backfill(batch_size=3)
    print(f"  resumed: embedded {done.embedded} more, complete = {done.complete} "
          f"-> state = {core.registry.get('subword-v2').state.value}")

    h("3.  Shadow -- measure how differently the candidate retrieves")
    report = m.compare(QUERIES, k=3)
    for row in report.per_query:
        print(f"  overlap {row['overlap']:.2f}  {row['query']}")
        print(f"      v1: {row['old_top']}")
        print(f"      v2: {row['new_top']}")
    print(f"\n  mean overlap    {report.mean_overlap:.3f}")
    print(f"  top-1 agreement {report.top1_agreement:.3f}")
    print(f"  mean rank shift {report.mean_rank_shift:.3f}")
    print(f"  verdict: {report.verdict}")

    h("4.  Cutover -- gated on the measurement, not on confidence")
    try:
        m.cutover(min_overlap=0.99, report=report)
    except Exception as exc:
        print(f"  strict gate refuses:\n    {str(exc)[:92]}...")
        print(f"  active is unchanged: {core.registry.active.space.id}")

    new, old = m.cutover(min_overlap=0.4, report=report)
    print(f"\n  cut over: {old.space.id} -> {new.space.id}")
    print(f"  {old.space.id} is now {old.state.value} -- retired, not deleted "
          f"({core.store.vector_count(old.space.id)} vectors kept)")

    h("5.  Rollback -- a state change, not a reindex")
    restored, demoted = core.rollback_to("words-v1")
    print(f"  active is {restored.space.id} again; {demoted.space.id} is {demoted.state.value}")
    print("  no vectors were recomputed:")
    for s in core.spaces():
        print(f"    {s.space.id:<12} {s.state.value:<9} {s.vector_count} vectors")

    print(f"\n{RULE}\nTotal downtime: none. Both spaces stayed queryable throughout.\n{RULE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
