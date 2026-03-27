"""YAML config loader with single-level inheritance via ``base:`` key."""

import dataclasses
import os
import typing
from typing import Any, Dict

import yaml

from .schema import Config


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overrides* into *base* (neither dict is mutated)."""
    merged = base.copy()
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _load_raw(path: str) -> Dict[str, Any]:
    """Load a YAML file, resolving ``base:`` inheritance first."""
    with open(path) as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    base_path = raw.pop("base", None)
    if base_path is not None:
        if not os.path.isabs(base_path):
            # Try relative to current file first, then relative to cwd
            candidate = os.path.join(os.path.dirname(path), base_path)
            if os.path.exists(candidate):
                base_path = candidate
            # else: keep as-is (relative to cwd)
        base_raw = _load_raw(base_path)
        raw = _deep_merge(base_raw, raw)

    return raw


def _resolve_type(cls, field_name: str):
    """Resolve a (possibly stringified) type hint to the actual class."""
    hints = typing.get_type_hints(cls)
    return hints.get(field_name)


def _build_dataclass(cls, data: Dict[str, Any]):
    """Instantiate a dataclass from a (possibly nested) dict."""
    kwargs: Dict[str, Any] = {}
    for fld in dataclasses.fields(cls):
        if fld.name not in data:
            continue
        val = data[fld.name]
        if isinstance(val, dict):
            resolved = _resolve_type(cls, fld.name)
            if resolved is not None and dataclasses.is_dataclass(resolved):
                val = _build_dataclass(resolved, val)
        kwargs[fld.name] = val
    return cls(**kwargs)


def load_config(path: str) -> Config:
    """Load a YAML config file and return a :class:`Config` instance.

    The YAML may contain a ``base: path/to/base.yaml`` key.  The base
    file is loaded first, then the current file's values are deep-merged
    on top.
    """
    raw = _load_raw(path)
    return _build_dataclass(Config, raw)
