"""Shared utilities: logging, HTTP, IO and small numeric helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests

import config

LOGGER_NAME = "f1_intelligence"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    base = setup_logging()
    return base.getChild(name) if name else base


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso_datetime(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, "", "NaT"):
        return pd.NaT
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        return ts
    except Exception:
        return pd.NaT


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


def load_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def unwrap(payload: Any) -> Any:
    """Return the ``data`` member of a stored collector payload, if present."""
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_parquet(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def dataframe_hash(df: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    if df is None or df.empty:
        return "empty"
    subset = df[list(columns)] if columns else df
    # Sorting makes the hash independent of row order for the same content.
    try:
        subset = subset.sort_values(list(subset.columns)).reset_index(drop=True)
    except Exception:
        subset = subset.reset_index(drop=True)
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(subset, index=False).values.tobytes())
    return h.hexdigest()[:24]


def safe_div(numerator: float, denominator: float, default: float = math.nan) -> float:
    try:
        if denominator is None or denominator == 0 or pd.isna(denominator):
            return default
        return float(numerator) / float(denominator)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return value
    return max(low, min(high, value))


def parse_race_id(race_id: str) -> tuple[int, int]:
    """Accept 2024_5, 2024-05, 2024/5 and similar forms."""
    match = re.search(r"(\d{4})\D+(\d{1,2})", str(race_id))
    if not match:
        raise ValueError(f"Invalid race id {race_id!r}; expected e.g. 2024_5 or 2024-05")
    return int(match.group(1)), int(match.group(2))


def make_race_id(season: int, round_number: int) -> str:
    return f"{int(season)}_{int(round_number):02d}"


def normalise_driver_name(forename: str | None, surname: str | None, code: str | None = None) -> str:
    if forename and surname:
        return f"{forename} {surname}".strip()
    return (code or "Unknown").strip()


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return value.strip("_") or "unknown"


class RateLimiter:
    def __init__(self, min_interval: float = config.REQUEST_RATE_LIMIT_SECONDS):
        self.min_interval = max(0.0, float(min_interval))
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


class HttpClient:
    """Small requests wrapper with retries, backoff and rate limiting.

    The wrapper is intentionally dependency-light so collectors and tests can
    share the same behaviour.  Tests inject a fake session.
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: int = config.REQUEST_TIMEOUT_SECONDS,
        max_retries: int = config.MAX_RETRIES,
        rate_limiter: RateLimiter | None = None,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT, "Accept": "application/json"})

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> requests.Response:
        url = self._url(path)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.rate_limiter.wait()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} for {url}", response=response
                    )
                if response.status_code >= 400:
                    # 404 is a normal "not covered by this source" answer for
                    # some historical endpoints; let callers decide.  Client
                    # errors other than 429 are not transient, so do not retry
                    # them (retrying a 404 only wastes the rate-limit budget).
                    response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    # Deterministic client error (404, 403, ...): retrying
                    # cannot help and would only burn the rate-limit budget.
                    break
                if attempt == self.max_retries - 1:
                    break
                sleep_for = min(60.0, (2**attempt) * 1.5)
                logging.getLogger(LOGGER_NAME).warning(
                    "Request failed (%s). Retry %s/%s in %.1fs: %s",
                    url,
                    attempt + 1,
                    self.max_retries,
                    sleep_for,
                    exc,
                )
                time.sleep(sleep_for)
        assert last_error is not None
        raise last_error

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        response = self.get(path, params=params)
        return response.json()


def flatten_dict(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Mapping):
            out.update(flatten_dict(value, name))
        else:
            out[name] = value
    return out
