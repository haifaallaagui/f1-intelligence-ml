"""Common collector behaviour: caching, manifests and provenance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import config
from src.utils import HttpClient, get_logger, save_json, load_json, utc_now_iso


class BaseCollector:
    source_name = "base"

    def __init__(self, raw_dir: Path, http: HttpClient | None = None):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.http = http or HttpClient()
        self.manifest_path = self.raw_dir / "manifest.json"
        self.manifest = load_json(self.manifest_path, default={}) or {}
        self.manifest.setdefault("source", self.source_name)
        self.manifest.setdefault("files", {})
        self.manifest.setdefault("errors", [])
        self.log = get_logger(self.source_name)

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

    def record_file(self, path: Path, metadata: Mapping[str, Any]) -> None:
        entry = dict(metadata)
        entry["path"] = self._relative(path)
        entry["collected_at"] = entry.get("collected_at") or utc_now_iso()
        self.manifest["files"][self._relative(path)] = entry
        self.manifest["updated_at"] = utc_now_iso()

    def record_error(self, context: str, error: Exception | str) -> None:
        self.manifest["errors"].append(
            {"context": context, "error": str(error), "at": utc_now_iso()}
        )
        # Keep the manifest bounded.
        self.manifest["errors"] = self.manifest["errors"][-200:]

    def save_manifest(self) -> None:
        self.manifest["updated_at"] = utc_now_iso()
        save_json(self.manifest_path, self.manifest)

    def file_metadata(self, path: Path) -> dict[str, Any] | None:
        return self.manifest.get("files", {}).get(self._relative(path))

    def write_json_payload(
        self,
        path: Path,
        data: Any,
        *,
        endpoint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "source": self.source_name,
            "endpoint": endpoint,
            "collected_at": utc_now_iso(),
            "data": data,
        }
        if metadata:
            payload.update(dict(metadata))
        save_json(path, payload)
        self.record_file(
            path,
            {
                "source": self.source_name,
                "endpoint": endpoint,
                "collected_at": payload["collected_at"],
                **(dict(metadata) if metadata else {}),
            },
        )

    @staticmethod
    def read_json_payload(path: Path) -> Any:
        path = Path(path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

    def existing_metadata(self, path: Path) -> dict[str, Any]:
        return self.file_metadata(path) or {}
