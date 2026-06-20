"""Lucian weekend IBKR balance history helpers.

This module is intentionally small and side-effect-light. It supports only the
scheduled Lucian weekend review flow; normal Telegram commands and realtime
trading paths should not call it.

AI status: Created with AI.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

HISTORY_FILENAME = "IBKR_BalanceHistory.json"
REQUEST_FILENAME = "GATEWAY_LUCIAN_BALANCE_SNAPSHOT_IBKR.txt"
RESPONSE_FILENAME = "GATEWAY_LUCIAN_BALANCE_SNAPSHOT_RESPONSE.json"
DEFAULT_NOTE = "Lucian weekend review"
UNAVAILABLE_SECTION = (
    "IBKR Balance Progress\n"
    "Unavailable - IBKR balance could not be fetched for this review."
)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def parse_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric balance")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("balance is not finite")
    return parsed


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_date(value: str | date | None, fallback_timestamp: str | None = None) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if value:
        return str(value)[:10]
    if fallback_timestamp:
        return str(fallback_timestamp)[:10]
    return date.today().isoformat()


def load_balance_history(history_path: str | Path, logger=None) -> dict[str, Any]:
    path = Path(history_path)
    if not path.exists():
        return {"last_updated": None, "snapshots": []}

    try:
        payload = read_json_object(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if logger:
            logger.warning("Lucian balance history malformed; starting fresh | path=%s | error=%s", path, exc)
        return {"last_updated": None, "snapshots": []}

    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        if logger:
            logger.warning("Lucian balance history missing snapshots list; starting fresh | path=%s", path)
        snapshots = []

    normalized = []
    for raw in snapshots:
        if not isinstance(raw, dict):
            continue
        try:
            normalized.append({
                "date": normalize_date(raw.get("date"), raw.get("timestamp")),
                "timestamp": str(raw.get("timestamp") or raw.get("date") or now_iso()),
                "net_liquidation": round(parse_float(raw.get("net_liquidation")), 2),
                "currency": str(raw.get("currency") or "USD").upper(),
                "source": str(raw.get("source") or "IBKR"),
                "note": str(raw.get("note") or DEFAULT_NOTE),
            })
        except (TypeError, ValueError):
            if logger:
                logger.warning("Lucian balance history skipped malformed snapshot | path=%s | snapshot=%s", path, raw)

    normalized.sort(key=lambda item: (item["date"], item["timestamp"]))
    return {
        "last_updated": payload.get("last_updated"),
        "snapshots": normalized,
    }


def record_balance_snapshot(
    history_path: str | Path,
    raw_snapshot: dict[str, Any],
    snapshot_date: str | date | None = None,
    logger=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp = str(raw_snapshot.get("timestamp") or now_iso())
    current = {
        "date": normalize_date(snapshot_date, timestamp),
        "timestamp": timestamp,
        "net_liquidation": round(parse_float(raw_snapshot.get("net_liquidation")), 2),
        "currency": str(raw_snapshot.get("currency") or "USD").upper(),
        "source": str(raw_snapshot.get("source") or "IBKR"),
        "note": str(raw_snapshot.get("note") or DEFAULT_NOTE),
    }

    path = Path(history_path)
    history = load_balance_history(path, logger=logger)
    snapshots = [
        snapshot for snapshot in history.get("snapshots", [])
        if snapshot.get("date") != current["date"]
    ]
    snapshots.append(current)
    snapshots.sort(key=lambda item: (item["date"], item["timestamp"]))

    updated = {
        "last_updated": current["timestamp"],
        "snapshots": snapshots,
    }
    atomic_write_json(path, updated)
    return updated, current


def previous_snapshot(history: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
    current_date = current.get("date")
    candidates = [
        snapshot for snapshot in history.get("snapshots", [])
        if isinstance(snapshot, dict) and snapshot.get("date") and snapshot.get("date") < current_date
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item["date"], item.get("timestamp") or ""))[-1]


def first_snapshot(history: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
    snapshots = [
        snapshot for snapshot in history.get("snapshots", [])
        if isinstance(snapshot, dict) and snapshot.get("date") and snapshot.get("date") <= current.get("date")
    ]
    if not snapshots:
        return None
    return sorted(snapshots, key=lambda item: (item["date"], item.get("timestamp") or ""))[0]


def format_amount(value: Any, currency: str | None = "USD") -> str:
    parsed = parse_float(value)
    code = str(currency or "USD").upper()
    if code == "USD":
        return f"${parsed:,.2f}"
    return f"{code} {parsed:,.2f}"


def format_signed_amount(delta: float, currency: str | None = "USD") -> str:
    code = str(currency or "USD").upper()
    sign = "+" if delta > 0 else "-" if delta < 0 else ""
    amount = abs(delta)
    if code == "USD":
        return f"{sign}${amount:,.2f}" if sign else "$0.00"
    return f"{sign}{code} {amount:,.2f}" if sign else f"{code} 0.00"


def pct_change(current_value: float, previous_value: float) -> float | None:
    if previous_value == 0:
        return None
    return (current_value - previous_value) / previous_value * 100


def format_signed_pct(value: float | None) -> str:
    if value is None:
        return "N.A."
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def format_change_line(label: str, current: dict[str, Any], baseline: dict[str, Any] | None, first_run_text: str) -> str:
    if baseline is None:
        return f"{label}: N.A. - {first_run_text}"

    current_value = parse_float(current.get("net_liquidation"))
    previous_value = parse_float(baseline.get("net_liquidation"))
    currency = current.get("currency") or baseline.get("currency") or "USD"
    delta = current_value - previous_value
    return f"{label}: {format_signed_amount(delta, currency)} ({format_signed_pct(pct_change(current_value, previous_value))})"


def build_balance_progress_section(history: dict[str, Any], current: dict[str, Any]) -> str:
    previous = previous_snapshot(history, current)
    first = first_snapshot(history, current)
    currency = current.get("currency") or "USD"

    lines = [
        "IBKR Balance Progress",
        f"Current: {format_amount(current.get('net_liquidation'), currency)}",
        f"Previous weekend: {format_amount(previous.get('net_liquidation'), previous.get('currency') or currency) if previous else 'N.A.'}",
        format_change_line("Weekly change", current, previous, "first recorded balance snapshot."),
    ]

    if first and first.get("date") != current.get("date"):
        lines.append(format_change_line("Since tracking started", current, first, "first recorded balance snapshot."))

    return "\n".join(lines)


def request_ibkr_balance_snapshot(
    functions_dir: str | Path,
    timeout_seconds: float = 20,
    poll_seconds: float = 0.5,
    logger=None,
) -> dict[str, Any]:
    functions_path = Path(functions_dir)
    request_path = functions_path / REQUEST_FILENAME
    response_path = functions_path / RESPONSE_FILENAME
    functions_path.mkdir(parents=True, exist_ok=True)

    try:
        response_path.unlink()
    except FileNotFoundError:
        pass

    request_payload = {
        "requested_at": now_iso(),
        "source": "Lucian weekend review",
    }
    atomic_write_json(request_path, request_payload)
    if logger:
        logger.info("Lucian requested IBKR balance snapshot | request=%s", request_path)

    deadline = time.monotonic() + max(float(timeout_seconds), 0.1)
    while time.monotonic() < deadline:
        if response_path.exists():
            try:
                response = read_json_object(response_path)
            finally:
                try:
                    response_path.unlink()
                except OSError:
                    pass

            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or "IBKR balance snapshot request failed"))
            response["net_liquidation"] = round(parse_float(response.get("net_liquidation")), 2)
            response["currency"] = str(response.get("currency") or "USD").upper()
            response["source"] = "IBKR"
            response["note"] = DEFAULT_NOTE
            return response

        time.sleep(max(float(poll_seconds), 0.05))

    try:
        request_path.unlink()
    except FileNotFoundError:
        pass
    raise TimeoutError(f"IBKR balance snapshot response timed out after {timeout_seconds}s")


def capture_weekend_balance_progress(
    functions_dir: str | Path,
    review_date: str | date | None = None,
    timeout_seconds: float = 20,
    logger=None,
) -> dict[str, Any]:
    history_path = Path(functions_dir) / HISTORY_FILENAME
    try:
        raw_snapshot = request_ibkr_balance_snapshot(
            functions_dir,
            timeout_seconds=timeout_seconds,
            logger=logger,
        )
        history, current = record_balance_snapshot(
            history_path,
            raw_snapshot,
            snapshot_date=review_date,
            logger=logger,
        )
        section = build_balance_progress_section(history, current)
        if logger:
            logger.info(
                "Lucian balance snapshot recorded | date=%s | currency=%s | history=%s",
                current.get("date"),
                current.get("currency"),
                history_path,
            )
        return {
            "ok": True,
            "section": section,
            "history_path": str(history_path),
            "snapshot": current,
        }
    except Exception as exc:
        if logger:
            logger.warning("Lucian IBKR balance progress unavailable: %s - %s", type(exc).__name__, exc)
        return {
            "ok": False,
            "section": UNAVAILABLE_SECTION,
            "history_path": str(history_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
