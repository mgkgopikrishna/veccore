"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .embedders import HashEmbedder
from .engine import VecCore
from .models import Document
from .registry import RegistryError


def _build_embedder(spec: str):
    """Parse `id:words` / `id:subword` / `id:st:<model>` into an embedder."""
    parts = spec.split(":")
    sid = parts[0]
    kind = parts[1] if len(parts) > 1 else "words"
    if kind == "words":
        return HashEmbedder(sid, 512, "a", word_ngram=1)
    if kind == "subword":
        return HashEmbedder(sid, 512, "a", word_ngram=1, char_ngram=3)
    if kind == "st":
        from .embedders import SentenceTransformerEmbedder

        model = parts[2] if len(parts) > 2 else "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(sid, model_id=model)
    raise SystemExit(f"unknown embedder kind {kind!r}; use words, subword or st:<model>")


def _load_corpus(core: VecCore, paths: list[str]) -> int:
    n = 0
    for raw in paths:
        p = Path(raw)
        files = sorted(p.rglob("*.txt")) + sorted(p.rglob("*.md")) if p.is_dir() else [p]
        for f in files:
            core.ingest(Document(id=f.stem, text=f.read_text(encoding="utf-8", errors="replace"),
                                 metadata={"path": str(f)}))
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="veccore",
        description="A RAG server where embedding spaces are versioned and migrations are safe.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="run the end-to-end migration walkthrough")
    demo.add_argument("--k", type=int, default=3)

    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--space", default="words-v1:words", help="id:kind of the initial space")
    serve.add_argument("--load", nargs="*", default=[], help="files or directories to ingest")

    q = sub.add_parser("query", help="one-shot query over a directory of documents")
    q.add_argument("text")
    q.add_argument("--load", nargs="+", required=True)
    q.add_argument("--space", default="words-v1:words")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--json", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "demo":
        from pathlib import Path as _P

        demo_path = _P(__file__).resolve().parents[1] / "examples" / "migration_demo.py"
        code = compile(demo_path.read_text(encoding="utf-8"), str(demo_path), "exec")
        ns: dict = {"__name__": "__main__", "__file__": str(demo_path)}
        try:
            exec(code, ns)  # noqa: S102 - running our own bundled example
        except SystemExit as e:
            return int(e.code or 0)
        return 0

    core = VecCore()
    core.add_space(_build_embedder(args.space), activate=True)

    if args.cmd == "query":
        loaded = _load_corpus(core, args.load)
        try:
            hits = core.query(args.text, k=args.k)
        except RegistryError as exc:
            print(exc, file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps([h.to_dict() for h in hits], indent=2))
        else:
            print(f"{loaded} document(s) indexed in space {core.registry.active.space.id}\n")
            for h in hits:
                print(f"  {h.score:.3f}  {h.chunk_id}")
                print(f"         {h.text[:100]}...")
        return 0

    if args.cmd == "serve":
        if args.load:
            print(f"ingested {_load_corpus(core, args.load)} document(s)")
        try:
            import uvicorn

            from .api import create_app
        except ImportError:
            print("The server needs the extra: pip install 'veccore[server]'", file=sys.stderr)
            return 2
        uvicorn.run(create_app(core), host=args.host, port=args.port)
        return 0

    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
