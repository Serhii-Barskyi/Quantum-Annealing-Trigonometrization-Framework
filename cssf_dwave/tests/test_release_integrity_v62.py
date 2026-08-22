from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "releases" / "scientific_manifests"
HARD_ROOT = "/content/drive/MyDrive/cssf_dwave"
NOTEBOOKS = [
    "CSSF_dwave_case300.ipynb",
    "CSSF_QA_DWave_Evidence_Simulator_v56.ipynb",
]
PROFILE_PATH = ROOT / "releases" / "RELEASE_PROFILE_v62.json"


def release_profile() -> str:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profile"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pythonize_notebook_cell(source: str) -> str:
    # Colab shell/magic lines are valid notebook syntax but not Python AST.
    return "\n".join(
        "pass" if line.lstrip().startswith(("!", "%")) else line
        for line in source.splitlines()
    )


def test_exactly_two_canonical_notebooks():
    notebooks = sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks == sorted(NOTEBOOKS)


def test_both_notebooks_are_output_free_and_code_cells_parse():
    expected_code_cells = {
        "CSSF_dwave_case300.ipynb": 26,
        "CSSF_QA_DWave_Evidence_Simulator_v56.ipynb": 56,
    }
    for name in NOTEBOOKS:
        nb = _parse_notebook(ROOT / "notebooks" / name)
        code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        assert len(code) == expected_code_cells[name]
        assert all(not c.get("outputs") for c in code)
        assert all(c.get("execution_count") is None for c in code)
        for i, cell in enumerate(code):
            src = _pythonize_notebook_cell("".join(cell.get("source", [])))
            ast.parse(src, filename=f"{name}:code-cell-{i}")


def test_hard_colab_root_is_preserved_and_legacy_root_absent():
    legacy = re.compile(r"/content/drive/MyDrive/cssf(?!_dwave)")
    targets = [ROOT / "README.md"]
    targets += [ROOT / "notebooks" / name for name in NOTEBOOKS]
    targets += list((ROOT / "docs").glob("*.md"))
    targets += [ROOT / "requirements-colab.txt", ROOT / "requirements-case300.txt"]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert not legacy.search(text), path
    required = [
        ROOT / "README.md",
        ROOT / "notebooks" / "CSSF_dwave_case300.ipynb",
        ROOT / "notebooks" / "CSSF_QA_DWave_Evidence_Simulator_v56.ipynb",
        ROOT / "docs" / "NOTEBOOK_CASE300.md",
        ROOT / "docs" / "NOTEBOOK_SIMULATOR.md",
        ROOT / "docs" / "PUBLIC_METHOD_AND_CLAIM_BOUNDARIES.md",
        ROOT / "requirements-colab.txt",
    ]
    for path in required:
        assert HARD_ROOT in path.read_text(encoding="utf-8"), path


def test_paid_monograph_is_not_bundled_and_framework_readme_is_technical_only():
    assert not (ROOT / ".github" / "README.md").exists()
    assert (ROOT / "README.md").is_file()
    paid_pdf_name = "Quantum_Annealing_" + "Trigonometrization.pdf"
    monograph = ROOT / "docs" / paid_pdf_name
    assert not monograph.exists()
    assert release_profile() in {"PUBLIC_GITHUB_RELEASE", "FULL_INTERNAL_TEST_RELEASE"}
    technical_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "CSSF_dwave_case300.ipynb" in technical_readme
    assert "CSSF_QA_DWave_Evidence_Simulator_v56.ipynb" in technical_readme
    assert paid_pdf_name not in technical_readme
    assert "[Quantum Annealing Trigonometrization](https://www.linkedin.com/in/serhii-barskyi/)" in technical_readme
    assert "Quantum-Annealing-Trigonometrization-Framework" in technical_readme


def test_case300_notebook_has_no_embedded_secret_and_explicit_solver_gate():
    text = (ROOT / "notebooks" / "CSSF_dwave_case300.ipynb").read_text(encoding="utf-8")
    assert "DWAVE_API_TOKEN" in text
    assert "CSSF_DWAVE_SOLVER_ID" in text
    assert "Advantage_system4" in text and "Advantage_system6" in text
    assert "case300_compat" in text
    # Common D-Wave token prefix and generic assignments with a literal secret.
    assert not re.search(r"CFox-[A-Za-z0-9_-]{12,}", text)
    assert not re.search(r"DWAVE_API_TOKEN['\"]?\s*[:=]\s*['\"][^'\"]{8,}", text)


def test_case300_dc_baseline_is_fail_closed():
    text = (ROOT / "notebooks" / "CSSF_dwave_case300.ipynb").read_text(encoding="utf-8")
    assert "rho_dc_declared = ds.params.get('rho_dc_vs_ac')" in text
    assert "DC baseline: not available" in text
    assert "rho_dc_vs_ac=ds.params.get('rho_dc_vs_ac', 0.)" not in text


def test_case300_compatibility_api_and_fail_closed_solver_policy():
    import case300_compat
    from case300_compat import compute_lsf_h_bias, compute_lsf_offsets, validate_solver_id
    import numpy as np
    assert np.allclose(compute_lsf_offsets([0.1, 0.5, 0.9], delta_max=0.25), [0.0, -0.125, -0.25])
    assert np.allclose(compute_lsf_h_bias([0.1, 0.5, 0.9], gamma=0.05), [-0.005, -0.025, -0.045])
    assert validate_solver_id("Advantage_system4") == "Advantage_system4"
    assert validate_solver_id("Advantage_system6") == "Advantage_system6"
    try:
        validate_solver_id("Advantage2_system1")
    except ValueError:
        pass
    else:
        raise AssertionError("Zephyr/Advantage2 must be rejected by the frozen Pegasus notebook")
    assert hasattr(case300_compat, "solve_ising_dwave")


def test_all_python_sources_compile():
    bad = []
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:  # pragma: no cover - test report path
            bad.append((str(path.relative_to(ROOT)), repr(exc)))
    assert bad == []


def test_frozen_source_and_input_manifests_are_exact():
    for name, expected_count in [("FROZEN_SOURCE_MANIFEST_v51.json", 67), ("FROZEN_INPUT_MANIFEST_v51.json", 5)]:
        data = json.loads((MANIFEST_DIR / name).read_text(encoding="utf-8"))
        files = data["files"]
        assert len(files) == expected_count
        for rel, digest in files.items():
            p = ROOT / rel
            assert p.is_file(), rel
            assert sha256(p) == digest, rel


def test_locked_runtime_manifest_matches_every_declared_file():
    p = MANIFEST_DIR / "LOCKED_RUNTIME_MANIFEST_v62.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema"] == "CSSF-LOCKED-RUNTIME-MANIFEST-v62"
    assert data["canonical_colab_root"] == HARD_ROOT
    assert data["notebooks"] == NOTEBOOKS
    for row in data["files"]:
        q = ROOT / row["path"]
        assert q.is_file(), row["path"]
        assert sha256(q) == row["sha256"], row["path"]


def test_active_inventory_is_exact_static_tree():
    invp = MANIFEST_DIR / "ACTIVE_RELEASE_INVENTORY_v62.json"
    data = json.loads(invp.read_text(encoding="utf-8"))
    declared = {row["path"] for row in data["files"]}
    actual = set()
    ignored = {"__pycache__", ".pytest_cache"}
    runtime_roots = {"results", "checkpoints", "outputs", "logs", "models", "cache", "artifacts", "embeddings", "samplesets", "solver_metadata"}
    for q in ROOT.rglob("*"):
        if not q.is_file():
            continue
        rel = q.relative_to(ROOT)
        if rel.parts and rel.parts[0] in runtime_roots:
            continue
        if any(part in ignored for part in rel.parts):
            continue
        if q.suffix in {".pyc", ".pyo"} or q.name.endswith(".tmp"):
            continue
        actual.add(rel.as_posix())
    assert declared == actual


def test_release_privacy_and_internal_profile_boundaries():
    profile = release_profile()
    bad = []
    forbidden_public_names = (
        "RESEARCH_CONTRACT", "INTERNAL_VALIDATION", "PACKAGING_VALIDATION",
        "TECHNICAL_COMMERCIAL_PROPOSAL",
    )
    for q in ROOT.rglob("*"):
        if not q.is_file():
            continue
        rel = q.relative_to(ROOT).as_posix()
        if q.name.startswith("old_") or "__pycache__" in q.parts or ".pytest_cache" in q.parts:
            bad.append(rel)
        if profile == "PUBLIC_GITHUB_RELEASE" and any(token in q.name.upper() for token in forbidden_public_names):
            bad.append(rel)
    assert bad == []

    if profile == "PUBLIC_GITHUB_RELEASE":
        forbidden_monograph_files = [
            q.relative_to(ROOT).as_posix()
            for q in ROOT.rglob("*")
            if q.is_file() and "quantum_annealing_trigonometrization" in q.name.lower()
            and q.suffix.lower() in {".pdf", ".epub", ".mobi", ".doc", ".docx", ".txt"}
        ]
        assert forbidden_monograph_files == []
        text_ext = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".cff"}
        privacy_bad = []
        for q in ROOT.rglob("*"):
            if not q.is_file() or q.suffix.lower() not in text_ext:
                continue
            if q.resolve() == Path(__file__).resolve():
                continue  # avoid matching the privacy-test patterns themselves
            text = q.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"Research Contract v\d+|CSSF_QA_RESEARCH_CONTRACT", text, re.I):
                privacy_bad.append(q.relative_to(ROOT).as_posix())
            if re.search(r"CFox-[A-Za-z0-9_-]{12,}", text):
                privacy_bad.append(q.relative_to(ROOT).as_posix())
        assert privacy_bad == []


def test_calibration_files_match_approved_hashes():
    expected = {
        "09-1263A-C_Advantage_system4_annealing_schedule.xlsx": "03350bb86bab2f752697e1a8c37f3e4c2100c6596d0f6c6bf8f6d2e3e97de4f1",
        "09-1273A-F_Advantage_system6_annealing_schedule.xlsx": "d266ee71c8a0611cc392781da4df65e20969aed658b5df60453ac099202fdc06",
    }
    for name, digest in expected.items():
        assert sha256(ROOT / "calibration" / name) == digest


def test_public_method_document_preserves_claim_boundaries():
    text = (ROOT / "docs" / "PUBLIC_METHOD_AND_CLAIM_BOUNDARIES.md").read_text(encoding="utf-8")
    assert HARD_ROOT in text
    assert "QZero comparator remains unavailable" in text
    assert "GPU/SQA evidence and real-QPU evidence are distinct" in text
    assert "hardware case300 != GPU/SQA case300" in text
    assert "artificial zero" in text
    assert "case300_compat/" in text
    assert "0.99053" in text and "0.9914438669" in text



def test_public_repository_is_english_only_and_paths_have_no_cyrillic():
    """All public text, notebook content, source comments/docstrings, and paths must be English-only."""
    cyrillic = re.compile(r"[\u0400-\u04FF]")
    bad = []
    text_suffixes = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".cff", ".ipynb"}
    for q in ROOT.rglob("*"):
        if not q.is_file():
            continue
        rel = q.relative_to(ROOT).as_posix()
        if cyrillic.search(rel):
            bad.append((rel, "path"))
        if q.suffix.lower() in text_suffixes or q.name in {".gitignore", "LICENSE"}:
            text = q.read_text(encoding="utf-8", errors="ignore")
            if cyrillic.search(text):
                bad.append((rel, "content"))
    assert bad == []
