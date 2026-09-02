# VecCore

A RAG server where **embedding spaces are versioned** and changing the embedding model
is a migration you can measure, gate, and roll back — instead of a reindex outage.

```
cosine( modelA("container killed for exceeding its memory limit"),
        modelB("container killed for exceeding its memory limit") ) = +0.748
```

Two different models. Same dimension. Same text. That number is meaningless — and
nothing anywhere raises an error. It looks like a relevance score, it flows through your
pipeline, and your retrieval quality quietly degrades with no alert and no failed
request. VecCore makes that an exception at the boundary instead of a slow leak in
production.

## The problem this exists for

You want to change embedding model. Perhaps a better one shipped, perhaps you are
cutting cost, perhaps you changed your chunk size — which is the same problem wearing a
disguise.

Vectors from the old model and the new model are not comparable. So the moment you
switch, every vector you have stored is worthless. The usual options are all bad:

- **Reindex in place** — retrieval is broken until the backfill finishes.
- **Stand up a parallel system** — now you have two sources of truth and a sync problem.
- **Do it at 3am and hope** — no measurement, no rollback, and you find out from users.

The thing nobody does is the thing that matters most: **measure how differently the new
model retrieves on your own corpus, before committing.** Benchmark scores are someone
else's documents.

## How VecCore handles it

Four phases. Each has a property worth stating.

```
 1. BACKFILL   embed the corpus into the new space      old space keeps serving
                                                        resumable — restart costs one batch

 2. DUAL-WRITE new documents go into both spaces        candidate never falls behind

 3. SHADOW     run real queries through both, compare   the phase everyone skips

 4. CUTOVER    one atomic state transition              old space retired, not deleted
                                                        rollback is a state change
```

Run it yourself — no dependencies, no network, about a second:

```bash
python examples/migration_demo.py
```

Abridged output:

```
2.  Backfill subword-v2 -- in batches, while v1 keeps serving
  batch 1: embedded 3, 4 remaining -> state = building
  reads still served by words-v1: ['probe::0', 'tflock::0']
  querying the incomplete space is refused:
    space 'subword-v2' is still building and would return partial results...
  resumed: embedded 4 more, complete = True -> state = shadow

3.  Shadow -- measure how differently the candidate retrieves
  overlap 0.67  why was my container killed
      v1: ['crash::0', 'oom::0', 'pasta::0']
      v2: ['oom::0',   'crash::0', 'irsa::0']
  ...
  mean overlap    0.800
  top-1 agreement 0.400
  verdict: moderate drift -- review a sample before cutting over

4.  Cutover -- gated on the measurement, not on confidence
  strict gate refuses:
    refusing to cut over: measured overlap 0.800 is below the required 0.990...
  active is unchanged: words-v1
```

Look at the first query in phase 3. The old space ranked a **pasta recipe** third for
*"why was my container killed"*; the new one put the OOM document first. That is the
kind of thing a benchmark number never tells you and a shadow comparison does.

## The design decisions worth arguing about

**An embedding space is a first-class object with a content-addressed fingerprint.**
The fingerprint hashes the model id, dimension, normalisation, pooling, and a
preprocessor version. Vectors carry their space id; queries carry a space id; the store
refuses to compare across spaces. Renaming a space is free. Changing the model, the
dimension, the normalisation, *or the chunking* is not.

**Chunking is part of the space.** The same model over differently chunked text does not
produce comparable vectors. `preprocessor_version` makes that explicit, so changing
chunk size is a migration rather than a config tweak. This is the second most common way
a RAG index silently rots, and it is almost never modelled.

**Documents are stored once; vectors are stored per space.** One corpus can live in
several embedding spaces simultaneously. That single structural choice is what makes
zero-downtime migration possible at all — and what makes rollback free, since the old
vectors were never deleted.

**Exactly one space is ACTIVE, and only an ACTIVE space serves reads.** A half-backfilled
space is `BUILDING` and cannot be queried, so it can never return partial results that
look like bad relevance. Every transition validates before it mutates, so a rejected
cutover leaves the registry untouched.

**Cutover can be gated on measured behaviour.** `min_overlap` refuses the switch when the
divergence report falls below your bar. The machine checks the number instead of trusting
your confidence.

**The core package has zero dependencies.** The engine, registry, migration logic and
similarity are pure Python. FastAPI is only needed for the HTTP layer,
sentence-transformers only for real embeddings. CI has a dedicated job that installs
*nothing* but pytest and runs the whole suite, so a stray import cannot creep in.

## What it does not do

Honest limits, because a migration tool that overclaims is worse than none:

- **It does not tell you the new model is better.** It tells you how differently it
  behaves and makes the switch reversible. Deciding "better" needs labelled queries you
  care about. Overlap is a change signal, not a quality metric.
- **Search is exact, not approximate.** Brute-force cosine over every vector, with a
  numpy fast path. Correct and fully deterministic, and fine to a few hundred thousand
  vectors. Beyond that you want HNSW or IVF — the `VectorStore` protocol is where that
  plugs in.
- **The reference store is in-memory.** Restart and the corpus is gone. The interface is
  deliberately small so a persistent backend only has to reproduce `MemoryStore`'s
  behaviour.
- **No authentication on the API.** Put it behind something.
- **Migration state lives in the process.** Multi-replica coordination would need the
  registry in shared storage.

## Install

```bash
git clone https://github.com/mgkgopikrishna/veccore
cd veccore
pip install -e ".[dev,server]"
pytest -q
```

## Use it as a library

```python
from veccore.embedders import HashEmbedder
from veccore.engine import VecCore
from veccore.models import Document

core = VecCore()
core.add_space(HashEmbedder("words-v1", 512, word_ngram=1), activate=True)
core.ingest(Document(id="runbook", text="..."))

core.query("why was my pod killed", k=5)

# later: a better model appears
core.add_space(HashEmbedder("subword-v2", 512, word_ngram=1, char_ngram=3))
m = core.migrate_to("subword-v2")

m.backfill(batch_size=256)                       # v1 keeps serving
report = m.compare(["your", "real", "queries"])  # measure the drift
print(report.mean_overlap, report.verdict)

m.cutover(min_overlap=0.7, report=report)        # refused if the bar is not met
core.rollback_to("words-v1")                     # instant, nothing recomputed
```

Real embeddings instead of the deterministic one:

```python
from veccore.embedders import SentenceTransformerEmbedder
core.add_space(SentenceTransformerEmbedder("minilm-v1"), activate=True)
```

## Command line

```bash
veccore demo                                        # the full walkthrough
veccore query "terraform lock stuck" --load examples/corpus
veccore serve --load examples/corpus                # needs veccore[server]
```

## HTTP API

```
GET  /health                     active space, document and chunk counts
GET  /spaces                     every space with state and vector count
POST /documents                  ingest (dual-writes to all live spaces)
GET  /query?q=...&k=5&space=...  defaults to ACTIVE; BUILDING returns 409

POST /migrations/backfill        resumable, batched
POST /migrations/compare         divergence report
POST /migrations/cutover         optional min_overlap gate
POST /migrations/rollback        restore a retired space
```

Registry violations return **409 Conflict**, not 500 — refusing an unsafe transition is
correct behaviour, not a server error.

```bash
docker compose up --build        # http://localhost:8000/docs
```

## Layout

| Module | Role |
|---|---|
| `models.py` | `EmbeddingSpace` and its fingerprint — the core idea |
| `registry.py` | Space lifecycle and the one-active-space invariant |
| `migration.py` | Backfill, shadow comparison, cutover, rollback |
| `engine.py` | Ingest, query, dual-write |
| `chunking.py` | Sentence-aware splitting with overlap |
| `stores/` | `VectorStore` protocol and the in-memory reference |
| `embedders/` | Deterministic hashing embedder and sentence-transformers |
| `api.py` | FastAPI surface |

## Tests

67 tests, no network, under two seconds.

```bash
pytest -q
pytest -q --ignore=tests/test_api.py   # core only, needs nothing but pytest
```

They assert the properties, not the implementation: that same-dimension different-model
spaces are incompatible, that an incomplete backfill cannot be promoted or queried, that
a refused cutover changes nothing, that backfill resumes without re-embedding, that
retired spaces stop receiving writes, and that rollback recomputes no vectors.

## Licence

MIT
