"""Configuration loading with YAML inheritance and command-line overrides.

A config file may declare ``_base_: other.yaml`` (relative to its own folder);
the base is loaded first and the child values are deep-merged on top.
Command-line overrides use dotted keys, e.g. ``--set model.tagcn.channels=[64,128]``,
and are parsed as YAML so numbers, lists, and booleans keep their types.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, Optional

import yaml


class Config(dict):
    """Dictionary with attribute access, nested recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Config":
        return Config({k: copy.deepcopy(v, memo) for k, v in self.items()})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        out = cls()
        for key, value in data.items():
            out[key] = cls.from_dict(value) if isinstance(value, dict) else value
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = Config.from_dict(value) if isinstance(value, dict) else value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` updated by ``override`` recursively (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_with_base(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    base_name = data.pop("_base_", None)
    if base_name is None:
        return data
    base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base_name)
    return deep_merge(_load_yaml_with_base(base_path), data)


def parse_overrides(items: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Parse ``key.path=value`` strings; values are interpreted as YAML."""
    parsed: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must have the form key=value")
        key, raw = item.split("=", 1)
        parsed[key.strip()] = yaml.safe_load(raw.strip())
    return parsed


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Config:
    """Load a YAML config, resolve ``_base_`` inheritance, and apply overrides."""
    cfg = Config.from_dict(_load_yaml_with_base(path))
    for key, value in parse_overrides(overrides).items():
        cfg.set_path(key, value)
    return cfg


def save_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)
