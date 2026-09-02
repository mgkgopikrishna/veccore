"""HTTP surface. Skipped when the server extra is not installed."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tests.helpers import CORPUS, QUERIES, space_v1, space_v2  # noqa: E402
from veccore.api import create_app  # noqa: E402
from veccore.engine import VecCore  # noqa: E402


@pytest.fixture
def client():
    vc = VecCore()
    vc.add_space(space_v1(), activate=True)
    app = TestClient(create_app(vc))
    for doc_id, text in CORPUS:
        app.post("/documents", json={"id": doc_id, "text": text})
    vc.add_space(space_v2())          # candidate registered after the corpus exists
    return app


def test_health_reports_the_active_space(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["active_space"] == "words-v1"
    assert body["documents"] == 6


def test_spaces_lists_states_and_counts(client):
    spaces = {s["space"]["id"]: s for s in client.get("/spaces").json()["spaces"]}
    assert spaces["words-v1"]["state"] == "active"
    assert spaces["subword-v2"]["state"] == "building"
    assert spaces["words-v1"]["vector_count"] == 6
    assert spaces["subword-v2"]["vector_count"] == 0


def test_query_serves_the_active_space(client):
    body = client.get("/query", params={"q": "terraform lock stuck", "k": 3}).json()
    assert body["space"] == "words-v1"
    assert body["hits"]
    assert body["hits"][0]["score"] >= body["hits"][-1]["score"]


def test_querying_a_building_space_returns_409_not_partial_results(client):
    r = client.get("/query", params={"q": "anything", "space": "subword-v2"})
    assert r.status_code == 409
    assert "still building" in r.json()["detail"]


def test_full_migration_over_http(client):
    backfill = client.post(
        "/migrations/backfill", json={"target_space": "subword-v2", "batch_size": 2}
    ).json()
    assert backfill["embedded"] == 6
    assert backfill["complete"] is True
    assert backfill["batches"] == 3

    compare = client.post(
        "/migrations/compare",
        json={"target_space": "subword-v2", "queries": QUERIES, "k": 3},
    ).json()
    assert compare["queries"] == len(QUERIES)
    assert "verdict" in compare

    cut = client.post(
        "/migrations/cutover", json={"target_space": "subword-v2", "min_overlap": 0.0,
                                     "queries": QUERIES, "k": 3}
    ).json()
    assert cut["active_space"] == "subword-v2"
    assert cut["retired_space"] == "words-v1"

    rolled = client.post("/migrations/rollback", json={"target_space": "words-v1"}).json()
    assert rolled["active_space"] == "words-v1"
    assert client.get("/health").json()["active_space"] == "words-v1"


def test_cutover_is_refused_when_overlap_is_below_the_gate(client):
    client.post("/migrations/backfill", json={"target_space": "subword-v2"})
    r = client.post(
        "/migrations/cutover",
        json={"target_space": "subword-v2", "min_overlap": 1.01, "queries": QUERIES, "k": 3},
    )
    assert r.status_code == 409
    assert "below the required" in r.json()["detail"]
    assert client.get("/health").json()["active_space"] == "words-v1"


def test_empty_document_text_is_rejected_by_validation(client):
    assert client.post("/documents", json={"id": "x", "text": ""}).status_code == 422
