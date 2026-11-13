"""Model registry and versioning."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import config
from src.utils import load_json, save_json, utc_now_iso


def load_registry(path: Path | None = None) -> dict[str, Any]:
    path = Path(path or config.REGISTRY_FILE)
    return load_json(path, default={"models": {}, "version": None, "updated_at": None}) or {
        "models": {},
        "version": None,
        "updated_at": None,
    }


def model_key(task: str, mode: str) -> str:
    return f"{task}/{mode}"


def register_model(
    task: str,
    mode: str,
    *,
    algorithm: str,
    version: str,
    bundle_path: Path,
    metadata: dict[str, Any],
    registry_path: Path | None = None,
) -> dict[str, Any]:
    registry_path = Path(registry_path or config.REGISTRY_FILE)
    registry = load_registry(registry_path)
    models = registry.setdefault("models", {})
    key = model_key(task, mode)
    models[key] = {
        "task": task,
        "mode": mode,
        "algorithm": algorithm,
        "version": version,
        "bundle": str(Path(bundle_path)),
        "updated_at": utc_now_iso(),
        "metadata": metadata,
    }
    registry["version"] = version
    registry["updated_at"] = utc_now_iso()
    save_json(registry_path, registry)
    return models[key]


def get_model_entry(task: str, mode: str, registry_path: Path | None = None) -> dict[str, Any] | None:
    registry = load_registry(registry_path)
    return (registry.get("models", {}) or {}).get(model_key(task, mode))


def list_models(registry_path: Path | None = None) -> list[dict[str, Any]]:
    registry = load_registry(registry_path)
    return list((registry.get("models", {}) or {}).values())
