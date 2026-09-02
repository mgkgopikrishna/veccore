"""Shared corpus and space builders for the test suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veccore.embedders import HashEmbedder  # noqa: E402

CORPUS = [
    ("oom", "A container killed for exceeding its memory limit exits with code 137. "
            "The kernel OOM killer terminates it without a graceful shutdown."),
    ("crash", "CrashLoopBackOff means the container starts and exits repeatedly. "
              "Exit code 127 means the entrypoint binary was not found."),
    ("probe", "A readiness probe failing on the wrong port keeps the pod out of the "
              "Service endpoints even though the process is healthy."),
    ("tflock", "Terraform state locking prevents two applies running at once. "
               "DynamoDB provides the lock for the S3 backend."),
    ("iam", "An IRSA role lets a pod assume an AWS IAM role without a stored secret."),
    ("pasta", "Fresh pasta needs eggs, flour and a long rest before rolling."),
]

QUERIES = [
    "why was my container killed",
    "pod not receiving traffic",
    "terraform lock stuck",
    "aws permissions for a pod",
]


def space_v1(sid="words-v1"):
    return HashEmbedder(sid, dimension=512, seed="a", word_ngram=1)


def space_v2(sid="subword-v2"):
    return HashEmbedder(sid, dimension=512, seed="a", word_ngram=1, char_ngram=3)
