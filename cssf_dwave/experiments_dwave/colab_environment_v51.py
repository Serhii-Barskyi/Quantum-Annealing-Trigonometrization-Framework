from __future__ import annotations

"""Fail-closed Google Colab environment policy for the CSSF(QA) simulator.

This module contains only environment-resolution and validation logic.  It does
not alter any scientific model, frozen core, experiment definition, or result.
"""

from importlib import metadata
from typing import Iterable, Mapping, Sequence

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

NUMPY_EXACT = "2.0.2"
PANDAS_TESTED_SPEC = ">=2.2.2,<2.3"
PACKAGING_TESTED_SPEC = ">=25,<27"

FULL_EXACT_VERSIONS: dict[str, str] = {
    "numpy": NUMPY_EXACT,
    "qiskit": "1.2.4",
    "qiskit-algorithms": "0.3.1",
    "qiskit-aer-gpu": "0.15.1",
    "dwave-ocean-sdk": "9.4.0",
    "pandapower": "3.2.2",
    "highspy": "1.15.1",
}


def _matching_requirement(
    requirement_lines: Iterable[str], package_name: str
) -> Requirement | None:
    target = canonicalize_name(package_name)
    for raw in requirement_lines:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if canonicalize_name(req.name) == target:
            return req
    return None


def google_colab_requirements() -> tuple[str, ...]:
    """Return installed google-colab requirement metadata, fail-closed if absent."""

    try:
        reqs = metadata.requires("google-colab")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "google-colab package metadata is unavailable; this notebook is Colab-only."
        ) from exc
    return tuple(reqs or ())


def resolve_colab_pandas_install_requirement(
    requirement_lines: Sequence[str],
) -> str:
    """Resolve the exact pandas pin demanded by the current Colab host.

    The host exact pin is accepted only if it remains within the deliberately
    tested 2.2.x scientific compatibility window.  This prevents silently
    following a future Colab major/minor pandas change without a new release.
    """

    req = _matching_requirement(requirement_lines, "pandas")
    if req is None:
        raise RuntimeError("google-colab metadata does not declare a pandas requirement.")

    specs = list(req.specifier)
    exact = [s for s in specs if s.operator == "==" and "*" not in s.version]
    if len(specs) != 1 or len(exact) != 1:
        raise RuntimeError(
            "google-colab pandas requirement is not a single exact pin: " f"{req.specifier}"
        )

    version = Version(exact[0].version)
    tested = SpecifierSet(PANDAS_TESTED_SPEC)
    if version not in tested:
        raise RuntimeError(
            "The current Colab pandas pin is outside the tested CSSF(QA) window: "
            f"google-colab requires pandas=={version}, tested window is {PANDAS_TESTED_SPEC}. "
            "Stop and issue a new validated environment release instead of auto-upgrading."
        )
    return f"pandas=={version}"


def _version_satisfies(actual: str | None, spec: str) -> bool:
    if actual is None:
        return False
    try:
        return Version(actual) in SpecifierSet(spec)
    except InvalidVersion:
        return False


def validate_base_versions(
    installed: Mapping[str, str | None], requirement_lines: Sequence[str]
) -> list[str]:
    """Validate the post-restart base layer, including the Colab pandas pin."""

    errors: list[str] = []
    numpy_actual = installed.get("numpy")
    if numpy_actual != NUMPY_EXACT:
        errors.append(f"numpy: expected {NUMPY_EXACT}, found {numpy_actual}")

    pandas_actual = installed.get("pandas")
    if not _version_satisfies(pandas_actual, PANDAS_TESTED_SPEC):
        errors.append(
            f"pandas: expected {PANDAS_TESTED_SPEC}, found {pandas_actual}"
        )

    colab_pandas_req = _matching_requirement(requirement_lines, "pandas")
    if colab_pandas_req is None:
        errors.append("google-colab metadata has no pandas requirement")
    elif pandas_actual is None or Version(pandas_actual) not in colab_pandas_req.specifier:
        errors.append(
            "pandas does not satisfy the installed google-colab requirement: "
            f"required {colab_pandas_req.specifier}, found {pandas_actual}"
        )

    packaging_actual = installed.get("packaging")
    if not _version_satisfies(packaging_actual, PACKAGING_TESTED_SPEC):
        errors.append(
            f"packaging: expected {PACKAGING_TESTED_SPEC}, found {packaging_actual}"
        )
    return errors


def validate_full_versions(
    installed: Mapping[str, str | None], requirement_lines: Sequence[str]
) -> list[str]:
    """Validate the complete scientific environment after project installation."""

    errors = validate_base_versions(installed, requirement_lines)
    for package, expected in FULL_EXACT_VERSIONS.items():
        if package == "numpy":
            continue
        actual = installed.get(package)
        if actual != expected:
            errors.append(f"{package}: expected {expected}, found {actual}")
    return errors


def collect_versions(package_names: Iterable[str]) -> dict[str, str | None]:
    """Collect installed distribution versions without importing scientific packages."""

    out: dict[str, str | None] = {}
    for package in package_names:
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def pip_check_failure(
    returncode: int, stdout: str = "", stderr: str = ""
) -> str | None:
    """Return a detailed error message when ``pip check`` is not clean."""

    if int(returncode) == 0:
        return None
    detail = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
    if not detail:
        detail = "pip check returned a non-zero exit status without diagnostic text"
    return "pip check reported broken requirements:\n" + detail
