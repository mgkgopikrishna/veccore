"""Interactive migration console for Hugging Face Spaces.

You drive the state machine yourself: back-fill the candidate space, watch it refuse to
serve while incomplete, measure how differently it retrieves, then cut over -- or watch a
strict gate refuse the cutover. Runs on CPU with no API key and no model download.
"""

from __future__ import annotations

import gradio as gr

from veccore.embedders import HashEmbedder
from veccore.engine import VecCore
from veccore.models import Document
from veccore.registry import RegistryError
from veccore.similarity import cosine

CORPUS = [
    ("oom", "A container killed for exceeding its memory limit exits with code 137. The kernel "
            "OOM killer terminates it immediately, with no graceful shutdown, so in-flight "
            "requests are dropped."),
    ("crash", "CrashLoopBackOff means the container starts and exits repeatedly. Check the exit "
              "code first: 127 means the entrypoint binary was not found, 0 means the process "
              "finished successfully and was restarted anyway."),
    ("probe", "A readiness probe pointing at the wrong port keeps a healthy pod out of the "
              "Service endpoints. The process is fine; the kubelet is knocking on a door "
              "nothing is listening behind."),
    ("sched", "FailedScheduling with Insufficient cpu means no node has enough allocatable CPU "
              "for the pod's requests. The scheduler compares requests, not usage, so a cluster "
              "that looks idle can still be unschedulable."),
    ("tflock", "Terraform state locking prevents two applies running at once. DynamoDB provides "
               "the lock for the S3 backend, and a stale lock can be cleared with force-unlock."),
    ("irsa", "IRSA lets a pod assume an AWS IAM role through a projected service account token, "
             "so no long-lived AWS credential is ever stored in the cluster."),
    ("pasta", "Fresh pasta needs eggs, flour, and a long rest before rolling. Rushing the rest "
              "makes the dough tear when it goes through the machine."),
]

DEFAULT_QUERIES = "\n".join([
    "why was my container killed",
    "pod is running but gets no traffic",
    "terraform apply is stuck on a lock",
    "give a pod aws permissions without secrets",
    "not enough capacity to place the pod",
])

V1, V2 = "words-v1", "subword-v2"


def build() -> VecCore:
    core = VecCore()
    core.add_space(HashEmbedder(V1, 512, "a", word_ngram=1), activate=True)
    for doc_id, text in CORPUS:
        core.ingest(Document(id=doc_id, text=text))
    core.add_space(HashEmbedder(V2, 512, "a", word_ngram=1, char_ngram=3))
    return core


def state_table(core: VecCore) -> str:
    """Markdown rather than gr.Dataframe -- no pandas, no jinja2, nothing to go wrong."""
    head = ("| space | state | fingerprint | vectors | missing |\n"
            "|---|---|---|---|---|\n")
    rows = []
    for s in core.spaces():
        missing = len(core.store.chunk_ids_missing_in(s.space.id))
        rows.append(
            f"| `{s.space.id}` | **{s.state.value.upper()}** | `{s.space.fingerprint}` "
            f"| {s.vector_count} | {missing} |"
        )
    return head + "\n".join(rows)


def reset():
    core = build()
    return (
        core,
        state_table(core),
        "Fresh start. `words-v1` is ACTIVE and serving. `subword-v2` is BUILDING — it holds "
        "no vectors yet and is not allowed to answer a query.",
        "", "",
    )


def do_backfill(core, batch, max_batches):
    try:
        m = core.migrate_to(V2)
        p = m.backfill(batch_size=int(batch), max_batches=int(max_batches) or None)
    except RegistryError as exc:
        return core, state_table(core), f"REFUSED — {exc}"
    state = core.registry.get(V2).state.value.upper()
    note = (
        f"Embedded {p.embedded} chunk(s) in {p.batches} batch(es); {p.remaining} remaining.\n"
        f"`{V2}` is now {state}."
    )
    if not p.complete:
        note += ("\n\nStill incomplete, so it stays BUILDING and cannot serve or be compared. "
                 "Press Backfill again — it resumes from what is missing rather than starting over.")
    else:
        note += "\n\nFully backfilled, so it was promoted to SHADOW automatically. Now compare it."
    return core, state_table(core), note


def try_query_candidate(core, q):
    try:
        hits = core.query(q, k=3, space_id=V2)
        return "\n".join(f"{h.score:.3f}  {h.chunk_id}" for h in hits)
    except RegistryError as exc:
        return f"REFUSED — {exc}"


def do_compare(core, raw_queries, k):
    queries = [q.strip() for q in raw_queries.splitlines() if q.strip()]
    if not queries:
        return core, "", "Enter at least one query."
    try:
        report = core.migrate_to(V2).compare(queries, k=int(k))
    except RegistryError as exc:
        return core, "", f"REFUSED — {exc}"
    rows = ("| query | overlap | words-v1 top | subword-v2 top |\n|---|---|---|---|\n") + "\n".join(
        f"| {r['query']} | {r['overlap']:.2f} | `{', '.join(r['old_top'])}` | `{', '.join(r['new_top'])}` |"
        for r in report.per_query
    )
    summary = (
        f"mean overlap **{report.mean_overlap:.3f}**  ·  "
        f"top-1 agreement **{report.top1_agreement:.3f}**  ·  "
        f"mean rank shift **{report.mean_rank_shift:.2f}**\n\n"
        f"**{report.verdict}**\n\n"
        "Overlap measures *change*, not quality — it tells you the new space ranks different "
        "documents, and makes the switch reversible. Deciding which is better needs labelled "
        "queries you care about."
    )
    return core, rows, summary


def do_cutover(core, raw_queries, k, gate):
    queries = [q.strip() for q in raw_queries.splitlines() if q.strip()]
    try:
        m = core.migrate_to(V2)
        report = m.compare(queries, k=int(k)) if queries else None
        new, old = m.cutover(min_overlap=float(gate), report=report)
    except RegistryError as exc:
        return core, state_table(core), f"REFUSED — {exc}\n\nNothing changed. The registry validates before it mutates."
    return core, state_table(core), (
        f"Cut over: **{old.space.id} → {new.space.id}**.\n\n"
        f"`{old.space.id}` is RETIRED, not deleted — its "
        f"{core.store.vector_count(old.space.id)} vectors are untouched, which is what makes "
        "rollback a state change rather than a reindex."
    )


def do_rollback(core):
    try:
        restored, demoted = core.rollback_to(V1)
    except RegistryError as exc:
        return core, state_table(core), f"REFUSED — {exc}"
    return core, state_table(core), (
        f"Rolled back. `{restored.space.id}` is ACTIVE again and `{demoted.space.id}` is SHADOW.\n\n"
        "No vectors were recomputed — compare the counts in the table."
    )


def do_search(core, q):
    if not q.strip():
        return "", ""
    active = core.registry.active.space.id
    live = "\n".join(f"{h.score:.3f}  {h.chunk_id:<10} {h.text[:58]}..." for h in core.query(q, k=3))
    other = V2 if active == V1 else V1
    try:
        cand = "\n".join(
            f"{h.score:.3f}  {h.chunk_id:<10} {h.text[:58]}..."
            for h in core.query(q, k=3, space_id=other)
        )
    except RegistryError as exc:
        cand = f"REFUSED — {str(exc)[:150]}"
    return f"ACTIVE — {active}\n{live}", f"{other}\n{cand}"


CSS = ".mono textarea {font-family: ui-monospace, monospace !important; font-size: 12.5px}"

with gr.Blocks(title="VecCore", theme=gr.themes.Soft(), css=CSS) as demo:
    core_state = gr.State(build())

    a = HashEmbedder(V1, 512, "a", word_ngram=1)
    b = HashEmbedder(V2, 512, "a", word_ngram=1, char_ngram=3)
    t = "container killed for exceeding its memory limit"
    nonsense = cosine(a.embed([t])[0], b.embed([t])[0])

    gr.Markdown(f"""
# VecCore — versioned embedding spaces

Two different embedding models. Same dimension. Same sentence:

```
cosine( modelA("{t}"),
        modelB("{t}") ) = {nonsense:+.3f}
```

That number is **meaningless** — and nothing raises. It looks like a relevance score,
flows through your pipeline, and your retrieval quietly degrades with no alert and no
failed request. VecCore makes it an error at the boundary, and turns changing embedding
model into a migration you can measure, gate, and roll back.

**Drive it yourself below.** No API key, no model download, nothing to install.
""")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("#### Spaces")
            table = gr.Markdown(state_table(build()))
        with gr.Column(scale=2):
            note = gr.Markdown(
                "`words-v1` is ACTIVE and serving. `subword-v2` is BUILDING — no vectors yet, "
                "and not allowed to answer a query."
            )

    gr.Markdown("### 1 · Backfill — the live space keeps serving throughout")
    with gr.Row():
        batch = gr.Slider(1, 8, value=3, step=1, label="Batch size")
        max_b = gr.Slider(0, 5, value=1, step=1, label="Max batches (0 = run to completion)")
        backfill_btn = gr.Button("Backfill", variant="primary")
    gr.Markdown(
        "_Leave max batches at 1 and press twice — the second run resumes from what is "
        "missing instead of re-embedding everything. Try querying while it is incomplete._"
    )
    probe_q = gr.Textbox(value="why was my container killed", label="Query the incomplete candidate space")
    probe_btn = gr.Button("Query subword-v2")
    probe_out = gr.Textbox(label="Result", lines=3, elem_classes="mono")

    gr.Markdown("### 2 · Shadow — measure the disagreement before trusting it")
    with gr.Row():
        queries = gr.Textbox(value=DEFAULT_QUERIES, lines=5, label="Queries", elem_classes="mono")
        with gr.Column():
            k = gr.Slider(1, 5, value=3, step=1, label="k")
            compare_btn = gr.Button("Compare spaces", variant="primary")
    cmp_table = gr.Markdown()
    cmp_summary = gr.Markdown()

    gr.Markdown("### 3 · Cutover — gated on the measurement, not on confidence")
    with gr.Row():
        gate = gr.Slider(0.0, 1.0, value=0.99, step=0.01,
                         label="Required mean overlap (0.99 will refuse — lower it to ~0.5 to proceed)")
        cutover_btn = gr.Button("Cut over", variant="primary")
        rollback_btn = gr.Button("Roll back")

    gr.Markdown("### 4 · Search either space, any time")
    search_q = gr.Textbox(value="terraform apply is stuck on a lock", label="Query")
    search_btn = gr.Button("Search both")
    with gr.Row():
        live_out = gr.Textbox(label="Active space", lines=4, elem_classes="mono")
        cand_out = gr.Textbox(label="Other space", lines=4, elem_classes="mono")

    reset_btn = gr.Button("Reset everything")

    gr.Markdown(
        "---\nSource, design notes and honest limits: "
        "[github.com/mgkgopikrishna/veccore](https://github.com/mgkgopikrishna/veccore)"
    )

    backfill_btn.click(do_backfill, [core_state, batch, max_b], [core_state, table, note])
    probe_btn.click(try_query_candidate, [core_state, probe_q], probe_out)
    compare_btn.click(do_compare, [core_state, queries, k], [core_state, cmp_table, cmp_summary])
    cutover_btn.click(do_cutover, [core_state, queries, k, gate], [core_state, table, note])
    rollback_btn.click(do_rollback, [core_state], [core_state, table, note])
    search_btn.click(do_search, [core_state, search_q], [live_out, cand_out])
    reset_btn.click(reset, None, [core_state, table, note, cmp_table, cmp_summary])

if __name__ == "__main__":
    import os

    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
