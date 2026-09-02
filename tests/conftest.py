import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.helpers import CORPUS, space_v1, space_v2  # noqa: E402
from veccore.engine import VecCore  # noqa: E402
from veccore.models import Document  # noqa: E402


@pytest.fixture
def core():
    """v1 active, corpus loaded. v2 is NOT registered yet."""
    vc = VecCore()
    vc.add_space(space_v1(), activate=True)
    for doc_id, text in CORPUS:
        vc.ingest(Document(id=doc_id, text=text))
    return vc


@pytest.fixture
def core_with_candidate(core):
    """v2 registered after the corpus exists -- how a real migration begins."""
    core.add_space(space_v2())
    return core
