# -*- coding: utf-8 -*-
# Author: Serhii Barskyi | https://www.linkedin.com/in/serhii-barskyi/
# Data Science Course: https://preply.com/en/tutor/7756455
# Framework: Complex Spectral Surrogate Framework (CSSF)
#
# Licensed under the Apache License, Version 2.0.
# You may not use this file except in compliance with the License.
# Full license text: https://www.apache.org/licenses/LICENSE-2.0
#
# Attribution required: if you use this code, please cite:
# Serhii Barskyi, Complex Spectral Surrogate Framework (CSSF),
# https://www.linkedin.com/in/serhii-barskyi/

"""Strict YAML loader for CSSF-QA-D-Wave configuration files.

All configuration files must be located below the fixed Google Colab root:

    /content/drive/MyDrive/cssf_dwave

The loader deliberately rejects:

* paths outside the project tree;
* unsupported file extensions;
* duplicate YAML keys;
* non-mapping YAML documents;
* unknown configuration fields;
* invalid cross-level settings;
* silent fallback to default or alternative configurations.

Later files override earlier files recursively. Lists and scalar values are
replaced as complete values; they are never concatenated implicitly.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import yaml
from pydantic import ValidationError

from config.schema import COLAB_PROJECT_ROOT, CSSFConfig


SUPPORTED_YAML_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".yaml", ".yml"}
)


class ConfigurationError(RuntimeError):
    """Raised when a CSSF configuration cannot be loaded or validated."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """PyYAML safe loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """Construct a YAML mapping while rejecting duplicate keys."""

    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)

        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConfigurationError(
                f"Unhashable YAML mapping key at line "
                f"{key_node.start_mark.line + 1}."
            ) from exc

        if duplicate:
            raise ConfigurationError(
                f"Duplicate YAML key {key!r} at line "
                f"{key_node.start_mark.line + 1}."
            )

        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _resolve_config_path(path: str | Path) -> Path:
    """Resolve and validate a configuration path below the Colab root."""

    supplied = Path(path)

    if supplied.is_absolute():
        candidate = supplied.resolve(strict=False)
    else:
        candidate = (COLAB_PROJECT_ROOT / supplied).resolve(strict=False)

    root = COLAB_PROJECT_ROOT.resolve(strict=False)

    if candidate != root and root not in candidate.parents:
        raise ConfigurationError(
            "Configuration path must remain below "
            f"{COLAB_PROJECT_ROOT}; received {candidate}."
        )

    if candidate.suffix.lower() not in SUPPORTED_YAML_SUFFIXES:
        raise ConfigurationError(
            "Configuration files must use .yaml or .yml; "
            f"received {candidate.name!r}."
        )

    return candidate


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load one strict YAML mapping from the fixed project tree.

    Parameters
    ----------
    path:
        Absolute path below ``COLAB_PROJECT_ROOT`` or a relative path resolved
        from that root.

    Returns
    -------
    dict[str, Any]
        Parsed YAML mapping.

    Raises
    ------
    ConfigurationError
        If the file is missing, malformed, duplicated, empty, non-mapping, or
        outside the project tree.
    """

    resolved = _resolve_config_path(path)

    if not resolved.exists():
        raise ConfigurationError(
            f"Configuration file does not exist: {resolved}"
        )

    if not resolved.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {resolved}"
        )

    try:
        with resolved.open("r", encoding="utf-8") as stream:
            document = yaml.load(
                stream,
                Loader=_UniqueKeySafeLoader,
            )
    except UnicodeDecodeError as exc:
        raise ConfigurationError(
            f"Configuration is not valid UTF-8: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in {resolved}: {exc}"
        ) from exc

    if document is None:
        raise ConfigurationError(
            f"Configuration file is empty: {resolved}"
        )

    if not isinstance(document, dict):
        raise ConfigurationError(
            f"Top-level YAML document must be a mapping: {resolved}"
        )

    non_string_keys = [
        key for key in document if not isinstance(key, str)
    ]
    if non_string_keys:
        raise ConfigurationError(
            "Top-level configuration keys must be strings; "
            f"received {non_string_keys!r} in {resolved}."
        )

    return document


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input.

    Nested mappings are merged recursively. Scalars, tuples, and lists from
    ``override`` replace values from ``base`` as complete values.
    """

    result: dict[str, Any] = deepcopy(dict(base))

    for key, override_value in override.items():
        base_value = result.get(key)

        if isinstance(base_value, Mapping) and isinstance(
            override_value,
            Mapping,
        ):
            result[key] = deep_merge(base_value, override_value)
        else:
            result[key] = deepcopy(override_value)

    return result


def validate_config_mapping(
    mapping: Mapping[str, Any],
    *,
    source_name: str = "<mapping>",
) -> CSSFConfig:
    """Validate one already assembled configuration mapping."""

    if not isinstance(mapping, Mapping):
        raise ConfigurationError(
            f"Configuration source {source_name} must be a mapping."
        )

    try:
        return CSSFConfig.model_validate(dict(mapping))
    except ValidationError as exc:
        raise ConfigurationError(
            f"Configuration validation failed for {source_name}:\n{exc}"
        ) from exc


def load_config(
    paths: Sequence[str | Path],
) -> CSSFConfig:
    """Load, recursively merge, and validate YAML configuration files.

    Files are applied from left to right:

    ``merged = paths[0] <- paths[1] <- ... <- paths[-1]``

    At least one path is mandatory. This prevents accidental use of hidden
    defaults when a requested experiment configuration is missing.
    """

    if not paths:
        raise ConfigurationError(
            "At least one explicit configuration file is required."
        )

    merged: dict[str, Any] = {}
    resolved_sources: list[str] = []

    for path in paths:
        resolved = _resolve_config_path(path)
        loaded = load_yaml_mapping(resolved)
        merged = deep_merge(merged, loaded)
        resolved_sources.append(str(resolved))

    source_name = " <- ".join(resolved_sources)
    return validate_config_mapping(
        merged,
        source_name=source_name,
    )


def config_to_mapping(config: CSSFConfig) -> dict[str, Any]:
    """Serialize a validated configuration to a JSON-compatible mapping."""

    return config.model_dump(
        mode="json",
        exclude_none=False,
        exclude_unset=False,
        exclude_defaults=False,
    )


__all__ = [
    "SUPPORTED_YAML_SUFFIXES",
    "ConfigurationError",
    "load_yaml_mapping",
    "deep_merge",
    "validate_config_mapping",
    "load_config",
    "config_to_mapping",
]
