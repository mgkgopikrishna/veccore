"""Config files are code too.

Added after shipping a CI file with a YAML syntax error: `run:` had an unquoted
`": "` inside it, which ends a plain scalar, so GitHub rejected the whole workflow and
no job ran at all. Nothing in the test suite would have caught it. Now something does.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


def _yaml_files():
    return sorted(
        [*ROOT.glob(".github/workflows/*.yml"), *ROOT.glob("deploy/k8s/*.yaml"), ROOT / "docker-compose.yml"]
    )


@pytest.mark.parametrize("path", _yaml_files(), ids=lambda p: p.name)
def test_every_yaml_file_parses(path):
    with open(path, encoding="utf-8") as fh:
        docs = list(yaml.safe_load_all(fh))
    assert docs, f"{path.name} parsed to nothing"


def test_ci_workflow_declares_the_jobs_we_rely_on():
    wf = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert {"test", "core-has-no-dependencies", "docker"} <= set(wf["jobs"])


def test_the_zero_dependency_job_installs_nothing_but_pytest():
    """This job is the guard against a stray import in the core package."""
    wf = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    installs = [
        s["run"] for s in wf["jobs"]["core-has-no-dependencies"]["steps"] if "run" in s
    ]
    assert any(r.strip() == "pip install pytest" for r in installs), installs


def test_pyproject_core_has_no_runtime_dependencies():
    # tomllib is stdlib from 3.11; the CI matrix still covers 3.10.
    tomllib = pytest.importorskip("tomllib", reason="stdlib tomllib needs Python 3.11+")

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["dependencies"] == [], (
        "the core package must stay dependency-free; put it in an optional extra"
    )
