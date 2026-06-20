#!/usr/bin/env python
"""Read-only Xander trade/source reviewer used by Lucian.

The module keeps the original weekly trade-review behavior, and also exposes
small helpers for launch_xander.py and the standalone launch_lucian.py test entrypoint:
- deterministic Lucian intent/date/source parsing
- trade-history filtering
- source-stat loading from SOCIALMARKET_Stats.json
- optional Ollama wording over already-filtered data

AI status: Created with AI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

SCRIPT_DIR = Path(__file__).resolve().parent
BOT_LAUNCH_ROOT = SCRIPT_DIR.parent
BOTHUB_ROOT = os.environ.get("BOTHUB_ROOT") or str(BOT_LAUNCH_ROOT.parent)

if load_dotenv:
    load_dotenv(BOT_LAUNCH_ROOT / ".env")
    load_dotenv(SCRIPT_DIR / ".env")


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

BOTHUB_ROOT = os.environ.get("BOTHUB_ROOT") or BOTHUB_ROOT
FUNCTIONS_DIR = BOT_LAUNCH_ROOT / "Functions"
DEFAULT_TRADE_HISTORY_PATH = FUNCTIONS_DIR / "GATEWAY_IBKR_TradeHistory.txt"
DEFAULT_SOURCE_STATS_PATH = FUNCTIONS_DIR / "SOCIALMARKET_Stats.json"
DEFAULT_IBKR_HISTORICAL_HELPER_PATH = BOT_LAUNCH_ROOT / "IBKR" / "IBKR_HD.py"
DEFAULT_OUTPUT_DIR = BOT_LAUNCH_ROOT / "tools" / "weekly_reviewer" / "output"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
TRADE_REVIEW_IBKR_UNAVAILABLE_MESSAGE = (
    "⚠️ I cannot proceed with the trade review right now because Xander is not connected to IBKR.\n\n"
    "Normal trade stats are still available, but the deeper post-entry/post-exit review needs IBKR historical data."
)
FALLBACK_UNKNOWN_RESPONSE = (
    "I am not fully sure whether you mean trade performance or post assessment stats.\n\n"
    "Try asking:\n"
    "- Lucian, trade performance this week\n"
    "- Lucian, posts assessed yesterday\n"
    "- Lucian, all stats this month"
)
SINGAPORE_TZ = ZoneInfo(os.environ.get("XANDER_TIMEZONE", "Asia/Singapore"))
SOURCE_STATS_FIELDS = ("assessed_total", "cat_1_bullish", "cat_1_bearish", "cat_1_na", "cat_2")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# DATA MODELS / CLI
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime
    label: str
    note: str
    completed: bool = True
    config_sources: dict[str, str] | None = None
    config_fallback_used: bool = False


@dataclass(frozen=True)
class LucianRequest:
    raw_text: str
    query: str
    intents: tuple[str, ...]
    date_range: DateRange
    source_filter: str | None = None
    date_range_requested: bool = False
    trade_date_range: DateRange | None = None
    source_date_range: DateRange | None = None
    parse_note: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Xander trade/source metrics and a compact summary."
    )
    parser.add_argument(
        "--start-date",
        help="Inclusive review start date in YYYY-MM-DD, interpreted as Singapore local time.",
    )
    parser.add_argument(
        "--end-date",
        help="Inclusive review end date in YYYY-MM-DD, interpreted as Singapore local time.",
    )
    parser.add_argument(
        "--query",
        help="Answer a Lucian-style review question once and print the response.",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Do not call local Ollama; always use the deterministic fallback summary.",
    )
    return parser.parse_args()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# PARSING / NORMALIZATION HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def parse_date_time(value: Any) -> datetime | None:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    candidates = [
        raw,
        raw.replace("Z", "+00:00"),
        raw.replace("/", "-"),
    ]
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
    ]

    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            return ensure_singapore_time(parsed)
        except ValueError:
            pass

        for fmt in formats:
            try:
                parsed = datetime.strptime(candidate, fmt)
                return ensure_singapore_time(parsed)
            except ValueError:
                continue

    return None


def ensure_singapore_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=SINGAPORE_TZ)
    return value.astimezone(SINGAPORE_TZ)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_env_time(name: str, default_value: str) -> time:
    raw = os.environ.get(name, default_value).strip()
    try:
        return datetime.strptime(raw, "%H:%M").time()
    except ValueError:
        return datetime.strptime(default_value, "%H:%M").time()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# DATE RANGE / SESSION WINDOW HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def parse_env_time_first(names: tuple[str, ...], default_value: str) -> time:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return datetime.strptime(raw, "%H:%M").time()
        except ValueError:
                continue
    return datetime.strptime(default_value, "%H:%M").time()


def configured_time_source(names: tuple[str, ...], default_value: str) -> tuple[time, str, bool]:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return datetime.strptime(raw, "%H:%M").time(), name, False
        except ValueError:
            continue
    return datetime.strptime(default_value, "%H:%M").time(), f"default {default_value}", True


def format_human_time(value: datetime | time) -> str:
    raw = value.strftime("%I:%M %p")
    return raw[1:] if raw.startswith("0") else raw


def format_human_date(value: datetime) -> str:
    return value.strftime("%a, %d %b")


def describe_date_range(date_range: DateRange) -> str:
    start = ensure_singapore_time(date_range.start)
    end = ensure_singapore_time(date_range.end)
    time_span = f"{format_human_time(start)} - {format_human_time(end)} SGT"

    if "Xander trading week" in date_range.label:
        if date_range.completed:
            return (
                f"{date_range.label}, "
                f"{format_human_date(start)} evening through {format_human_date(end)} morning, {time_span}"
            )
        return (
            f"{date_range.label}, "
            f"{format_human_date(start)} evening through {format_human_date(end)}, {time_span}"
        )

    if "Xander trading windows" in date_range.label:
        return f"{date_range.label}, {format_human_date(start)} to {format_human_date(end)}, {time_span}"

    if "Xander trading window" in date_range.label:
        if start.date() != end.date():
            return (
                f"{date_range.label}, "
                f"{start.strftime('%A')} night to {end.strftime('%A')} morning, {time_span}"
            )
        return f"{date_range.label} on {format_human_date(start)}, {time_span}"

    if start.date() == end.date():
        return f"{date_range.label}, {format_human_date(start)}, {time_span}"

    return f"{date_range.label}, {format_human_date(start)} to {format_human_date(end)}, {time_span}"


def default_review_window(now: datetime | None = None) -> tuple[datetime, datetime, str]:
    """Return the most recent completed Mon 00:00 to Sat 08:59:59 SGT window."""
    now_sgt = now.astimezone(SINGAPORE_TZ) if now else datetime.now(SINGAPORE_TZ)
    today = now_sgt.date()
    days_since_saturday = (today.weekday() - 5) % 7
    cutoff_date = today - timedelta(days=days_since_saturday)
    cutoff = datetime.combine(cutoff_date, time(8, 59, 59), tzinfo=SINGAPORE_TZ)

    if now_sgt < cutoff:
        cutoff -= timedelta(days=7)

    start_date = cutoff.date() - timedelta(days=5)
    start = datetime.combine(start_date, time.min, tzinfo=SINGAPORE_TZ)
    note = (
        "Most recent completed Monday 00:00 through Saturday 08:59:59 "
        "Asia/Singapore review period."
    )
    return start, cutoff, note


def override_review_window(
    start_date: str | None, end_date: str | None
) -> tuple[datetime, datetime, str] | None:
    if not start_date and not end_date:
        return None
    if not start_date or not end_date:
        raise ValueError("--start-date and --end-date must be provided together.")

    try:
        start_day = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_day = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Date overrides must use YYYY-MM-DD.") from exc

    if end_day < start_day:
        raise ValueError("--end-date must be on or after --start-date.")

    start = datetime.combine(start_day, time.min, tzinfo=SINGAPORE_TZ)
    end = datetime.combine(end_day, time.max.replace(microsecond=0), tzinfo=SINGAPORE_TZ)
    note = "Manual override window; both dates are inclusive in Asia/Singapore time."
    return start, end, note


def xander_trading_times() -> tuple[time, time]:
    open_time = parse_env_time_first(("XANDER_TRADING_OPEN_SGT", "XANDER_MARKET_OPEN_SGT"), "17:00")
    close_time = parse_env_time_first(("XANDER_TRADING_CLOSE_SGT", "XANDER_MARKET_CLOSE_SGT"), "04:00")
    return open_time, close_time


def xander_trading_time_config_status() -> dict[str, Any]:
    open_time, open_source, open_default = configured_time_source(
        ("XANDER_TRADING_OPEN_SGT", "XANDER_MARKET_OPEN_SGT"),
        "17:00",
    )
    close_time, close_source, close_default = configured_time_source(
        ("XANDER_TRADING_CLOSE_SGT", "XANDER_MARKET_CLOSE_SGT"),
        "04:00",
    )
    return {
        "open_time": open_time,
        "open_source": open_source,
        "open_default": open_default,
        "close_time": close_time,
        "close_source": close_source,
        "close_default": close_default,
        "used_default": open_default or close_default,
    }


def xander_config_sources_for_date_range() -> tuple[dict[str, str], bool]:
    config_status = xander_trading_time_config_status()
    return (
        {
            "open": config_status["open_source"],
            "close": config_status["close_source"],
        },
        bool(config_status["used_default"]),
    )


def make_xander_date_range(
    start: datetime,
    end: datetime,
    label: str,
    note: str,
    completed: bool = True,
) -> DateRange:
    config_sources, fallback_used = xander_config_sources_for_date_range()
    return DateRange(
        start=start,
        end=max(start, end),
        label=label,
        note=note,
        completed=completed,
        config_sources=config_sources,
        config_fallback_used=fallback_used,
    )


def xander_window_for_start_day(start_day, label: str, end_at: datetime | None = None) -> DateRange:
    open_time, close_time = xander_trading_times()
    crosses_midnight = close_time <= open_time
    close_day = start_day + timedelta(days=1) if crosses_midnight else start_day
    start = datetime.combine(start_day, open_time, tzinfo=SINGAPORE_TZ)
    planned_end = datetime.combine(close_day, close_time, tzinfo=SINGAPORE_TZ)
    end = planned_end
    completed = True
    if end_at is not None:
        end_at_sgt = ensure_singapore_time(end_at)
        completed = end_at_sgt >= planned_end
        end = min(end, end_at_sgt)
    return make_xander_date_range(
        start=start,
        end=end,
        label=label,
        note=(
            "Resolved locally from XANDER_TRADING_OPEN_SGT/XANDER_TRADING_CLOSE_SGT, "
            "falling back to market open/close config if needed."
        ),
        completed=completed,
    )


def xander_window_containing(value: datetime, label: str | None = None) -> DateRange:
    value_sgt = ensure_singapore_time(value)
    open_time, close_time = xander_trading_times()
    crosses_midnight = close_time <= open_time
    if crosses_midnight and value_sgt.time() < close_time:
        start_day = value_sgt.date() - timedelta(days=1)
    else:
        start_day = value_sgt.date()
    return xander_window_for_start_day(start_day, label or "selected Xander trading window")


def xander_multi_window_range(start_day, end_start_day, label: str, end_at: datetime | None = None) -> DateRange:
    open_time, close_time = xander_trading_times()
    crosses_midnight = close_time <= open_time
    close_day = end_start_day + timedelta(days=1) if crosses_midnight else end_start_day
    start = datetime.combine(start_day, open_time, tzinfo=SINGAPORE_TZ)
    planned_end = datetime.combine(close_day, close_time, tzinfo=SINGAPORE_TZ)
    end = planned_end
    completed = True
    if end_at is not None:
        end_at_sgt = ensure_singapore_time(end_at)
        completed = end_at_sgt >= planned_end
        end = min(end, end_at_sgt)
    return make_xander_date_range(
        start=start,
        end=end,
        label=label,
        note=(
            "Resolved as a range of Xander trading windows using "
            "XANDER_TRADING_OPEN_SGT/XANDER_TRADING_CLOSE_SGT."
        ),
        completed=completed,
    )


def xander_trading_week_for_monday(monday, label: str, now: datetime | None = None) -> DateRange:
    end_start_day = monday + timedelta(days=4)
    planned = xander_multi_window_range(monday, end_start_day, label)
    if now is None:
        return planned
    now_sgt = ensure_singapore_time(now)
    if now_sgt < planned.start:
        return xander_multi_window_range(monday, end_start_day, label, end_at=planned.start)
    if now_sgt < planned.end:
        return xander_multi_window_range(monday, end_start_day, f"{label} so far", end_at=now_sgt)
    return planned


def current_xander_trading_week(now: datetime | None = None) -> DateRange:
    now_sgt = ensure_singapore_time(now or datetime.now(SINGAPORE_TZ))
    monday = now_sgt.date() - timedelta(days=now_sgt.weekday())
    return xander_trading_week_for_monday(monday, "this week's Xander trading week", now_sgt)


def previous_xander_trading_week(now: datetime | None = None) -> DateRange:
    now_sgt = ensure_singapore_time(now or datetime.now(SINGAPORE_TZ))
    this_monday = now_sgt.date() - timedelta(days=now_sgt.weekday())
    previous_monday = this_monday - timedelta(days=7)
    return xander_trading_week_for_monday(previous_monday, "last week's Xander trading week")


def rolling_seven_day_window(now: datetime | None = None) -> DateRange:
    now_sgt = ensure_singapore_time(now or datetime.now(SINGAPORE_TZ))
    start = now_sgt - timedelta(days=7)
    return make_xander_date_range(
        start,
        now_sgt,
        "past 7 days",
        "Explicit rolling seven-day window in Asia/Singapore time.",
        completed=False,
    )


def current_or_latest_xander_window(now: datetime | None = None) -> DateRange:
    now_sgt = ensure_singapore_time(now or datetime.now(SINGAPORE_TZ))
    open_time, close_time = xander_trading_times()
    crosses_midnight = close_time <= open_time

    if crosses_midnight:
        start_day = now_sgt.date() if now_sgt.time() >= open_time else now_sgt.date() - timedelta(days=1)
    else:
        start_day = now_sgt.date()

    current_window = xander_window_for_start_day(start_day, "today's Xander trading window")
    if current_window.start <= now_sgt <= current_window.end:
        return xander_window_for_start_day(start_day, "today's Xander trading window so far", end_at=now_sgt)

    return latest_completed_us_session(now_sgt)


def latest_completed_us_session(now: datetime | None = None, label: str = "latest completed Xander trading window") -> DateRange:
    now_sgt = ensure_singapore_time(now or datetime.now(SINGAPORE_TZ))
    open_time, close_time = xander_trading_times()

    close_date = now_sgt.date()
    if now_sgt.time() <= close_time:
        close_date -= timedelta(days=1)

    crosses_midnight = close_time <= open_time
    for offset in range(14):
        candidate_close_date = close_date - timedelta(days=offset)
        candidate_open_date = (
            candidate_close_date - timedelta(days=1)
            if crosses_midnight
            else candidate_close_date
        )
        if candidate_open_date.weekday() >= 5:
            continue
        start = datetime.combine(candidate_open_date, open_time, tzinfo=SINGAPORE_TZ)
        end = datetime.combine(candidate_close_date, close_time, tzinfo=SINGAPORE_TZ)
        return make_xander_date_range(
            start=start,
            end=end,
            label=label,
            note=(
                "Resolved locally from XANDER_TRADING_OPEN_SGT/XANDER_TRADING_CLOSE_SGT, "
                "falling back to market open/close config if needed."
            ),
            completed=True,
        )

    start, end, note = default_review_window(now_sgt)
    return DateRange(start, end, "latest completed review window", note)


def resolve_calendar_date_range(query: str, now: datetime | None = None) -> DateRange:
    query_l = query.lower()
    now_sgt = ensure_singapore_time(now or datetime.now(SINGAPORE_TZ))

    if "last night" in query_l:
        return latest_completed_us_session(now_sgt)

    if "weekly summary" in query_l or "week summary" in query_l:
        start, end, note = default_review_window(now_sgt)
        return DateRange(start, end, "most recent completed weekly review", note)

    if "last week" in query_l:
        this_monday = now_sgt.date() - timedelta(days=now_sgt.weekday())
        start_day = this_monday - timedelta(days=7)
        end_day = this_monday - timedelta(days=1)
        return DateRange(
            datetime.combine(start_day, time.min, tzinfo=SINGAPORE_TZ),
            datetime.combine(end_day, time.max.replace(microsecond=0), tzinfo=SINGAPORE_TZ),
            "last calendar week",
            "Previous Monday through Sunday in Asia/Singapore time.",
        )

    if "this week" in query_l:
        start_day = now_sgt.date() - timedelta(days=now_sgt.weekday())
        return DateRange(
            datetime.combine(start_day, time.min, tzinfo=SINGAPORE_TZ),
            now_sgt,
            "this week to date",
            "Current Monday through now in Asia/Singapore time.",
        )

    if "last month" in query_l or "previous month" in query_l:
        first_this_month = now_sgt.date().replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return DateRange(
            datetime.combine(last_month_start, time.min, tzinfo=SINGAPORE_TZ),
            datetime.combine(last_month_end, time.max.replace(microsecond=0), tzinfo=SINGAPORE_TZ),
            "last month",
            "Previous calendar month in Asia/Singapore time.",
        )

    if "this month" in query_l or "current month" in query_l or "whole month" in query_l or re.search(r"\bmonth\b", query_l):
        start_day = now_sgt.date().replace(day=1)
        return DateRange(
            datetime.combine(start_day, time.min, tzinfo=SINGAPORE_TZ),
            now_sgt,
            "this month to date",
            "Current calendar month through now in Asia/Singapore time.",
        )

    if "past week" in query_l or "last 7 days" in query_l:
        return DateRange(
            now_sgt - timedelta(days=7),
            now_sgt,
            "past 7 days",
            "Rolling seven-day window in Asia/Singapore time.",
        )

    if "yesterday" in query_l:
        day = now_sgt.date() - timedelta(days=1)
        return DateRange(
            datetime.combine(day, time.min, tzinfo=SINGAPORE_TZ),
            datetime.combine(day, time.max.replace(microsecond=0), tzinfo=SINGAPORE_TZ),
            "yesterday",
            "Previous Singapore calendar day.",
        )

    if "today" in query_l:
        day = now_sgt.date()
        return DateRange(
            datetime.combine(day, time.min, tzinfo=SINGAPORE_TZ),
            now_sgt,
            "today",
            "Current Singapore calendar day to now.",
        )

    return latest_completed_us_session(now_sgt)


def resolve_lucian_window(query: str, now: datetime | None = None) -> DateRange:
    query_l = query.lower()
    now_sgt = ensure_singapore_time(now or datetime.now(SINGAPORE_TZ))

    if "calendar" in query_l or "midnight" in query_l:
        return resolve_calendar_date_range(query, now_sgt)

    if any(term in query_l for term in ("past 7 days", "last 7 days", "rolling 7 days")):
        return rolling_seven_day_window(now_sgt)

    if any(term in query_l for term in ("last night", "yesterday night", "last session", "previous session", "overnight session")):
        return latest_completed_us_session(now_sgt, "last night's Xander window")

    if "yesterday" in query_l:
        return latest_completed_us_session(now_sgt, "yesterday's Xander window")

    if "today" in query_l:
        return current_or_latest_xander_window(now_sgt)

    if "last week" in query_l:
        return previous_xander_trading_week(now_sgt)

    if any(term in query_l for term in ("this week", "weekly", "week summary", "weekly summary", "past week")):
        return current_xander_trading_week(now_sgt)

    if "last month" in query_l or "previous month" in query_l:
        first_this_month = now_sgt.date().replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return xander_multi_window_range(last_month_start, last_month_end, "last month's Xander trading windows")

    if "this month" in query_l or "current month" in query_l or "whole month" in query_l or re.search(r"\bmonth\b", query_l):
        start_day = now_sgt.date().replace(day=1)
        reference_window = current_or_latest_xander_window(now_sgt)
        end_start_day = max(start_day, reference_window.start.date())
        return xander_multi_window_range(start_day, end_start_day, "this month's Xander trading windows", end_at=reference_window.end)

    return latest_completed_us_session(now_sgt)


def resolve_date_range(query: str, now: datetime | None = None) -> DateRange:
    return resolve_lucian_window(query, now)


def resolve_trade_date_range(query: str, now: datetime | None = None) -> DateRange:
    return resolve_lucian_window(query, now)


def resolve_source_date_range(query: str, now: datetime | None = None) -> DateRange:
    return resolve_trade_date_range(query, now)


def has_explicit_date_range(query: str) -> bool:
    query_l = query.lower()
    return any(
        term in query_l
        for term in (
            "last night",
            "yesterday night",
            "last session",
            "previous session",
            "overnight session",
            "yesterday",
            "today",
            "this week",
            "last week",
            "past week",
            "weekly",
            "weekly summary",
            "week summary",
            "last 7 days",
            "past 7 days",
            "this month",
            "last month",
            "previous month",
            "current month",
            "whole month",
        )
    )


def extract_lucian_query(text: str) -> str | None:
    match = re.match(r"^\s*(?:@lucian\w*|/lucian|lucian)\b\s*[:,\-]?\s*(.*)$", text, re.I)
    if not match:
        return None
    query = match.group(1).strip()
    return query or None


def normalize_source_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower().replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in {"walter", "walter bloomberg", "deitaone", "walterbloomberg"}:
        return "Walter"
    if cleaned in {"trump", "truth social", "donald trump", "donald j trump", "realdonaldtrump"}:
        return "Trump"
    if cleaned in {"stock titan", "stocktitan", "stock titan rss"}:
        return "Stock Titan"
    return None


def detect_source_filter(query: str) -> str | None:
    query_l = query.lower()
    matches = detect_source_mentions(query_l)
    if len(matches) == 1:
        return matches[0]
    return None


def detect_source_mentions(query_l: str) -> list[str]:
    matches = []
    if re.search(r"\b(walter|deitaone|walter bloomberg)\b", query_l):
        matches.append("Walter")
    if re.search(r"\b(trump|truth social|realdonaldtrump)\b", query_l):
        matches.append("Trump")
    if re.search(r"\b(stock\s*titan|stocktitan)\b", query_l):
        matches.append("Stock Titan")
    return list(dict.fromkeys(matches))


def extract_ticker_mentions(text: str) -> list[str]:
    query = extract_lucian_query(text) or str(text or "")
    excluded = {
        "CAT",
        "SGT",
        "P",
        "PL",
        "PNL",
        "USA",
        "IBKR",
        "TWS",
        "YES",
        "THE",
        "THEM",
        "THOSE",
        "THIS",
        "IT",
        "DO",
        "GO",
        "SURE",
        "ALL",
    }
    matches = []
    for match in re.finditer(r"\$?\b[A-Za-z]{1,5}\b", query):
        token = match.group(0)
        if not token.startswith("$") and token != token.upper():
            continue
        cleaned = token.upper().replace("$", "").strip()
        if cleaned and cleaned not in excluded and not cleaned.isdigit():
            matches.append(cleaned)
    return list(dict.fromkeys(matches))


def is_trade_review_confirmation(text: str) -> bool:
    query = (extract_lucian_query(text) or str(text or "")).strip().lower()
    query = re.sub(r"[.!?,]+$", "", query).strip()
    if not query:
        return False
    confirmations = {
        "yes",
        "yes please",
        "yeah",
        "yep",
        "sure",
        "sure please",
        "go ahead",
        "do it",
        "please do",
        "review it",
        "review them",
        "review the trades",
        "yes review it",
        "yes review them",
        "yes review the trades",
    }
    return query in confirmations


def is_trade_deep_review_request(text: str) -> bool:
    query = (extract_lucian_query(text) or str(text or "")).strip().lower()
    if not query:
        return False

    behavior_terms = (
        "after entry",
        "after exit",
        "post entry",
        "post-exit",
        "post exit",
        "bounced",
        "bounce",
        "recover",
        "recovered",
        "continued crashing",
        "continue crashing",
        "kept falling",
        "mfe",
        "mae",
        "entry late",
        "late entry",
        "exit protective",
    )
    review_terms = ("review", "analyse", "analyze", "check")
    trade_terms = ("trade", "trades", "failed trade", "exit", "entry")

    if any(term in query for term in behavior_terms):
        return True
    if any(term in query for term in review_terms) and any(term in query for term in trade_terms):
        return True
    if extract_ticker_mentions(text) and any(term in query for term in ("behave", "after", "review", "check")):
        return True
    return False


def trade_review_request_needs_pending(text: str) -> bool:
    query = (extract_lucian_query(text) or str(text or "")).strip().lower()
    if has_explicit_date_range(query) or extract_ticker_mentions(query):
        return False
    return any(term in query for term in ("them", "those", "it", "the trades", "these trades"))


def parse_lucian_request(text: str, now: datetime | None = None) -> LucianRequest | None:
    query = extract_lucian_query(text)
    if query is None:
        return None

    query_l = query.lower()
    source_filter = detect_source_filter(query)
    intents: list[str] = []

    source_terms = (
        "source stats",
        "source",
        "assessed",
        "assessment",
        "assessments",
        "post assessed",
        "posts assessed",
        "post stats",
        "cat_1",
        "cat 1",
        "cat_2",
        "cat 2",
    )
    trade_terms = (
        "trade",
        "trades",
        "pnl",
        "p/l",
        "profit",
        "loss",
        "losses",
        "performance",
        "perform",
    )
    combined_terms = (
        "all stats",
        "overall stats",
        "full stats",
        "everything",
        "all the stats",
        "combined",
        "how did xander do",
        "xander do",
        "weekly review",
        "weekly summary",
        "week review",
    )
    vague_review_terms = (
        "what about",
        "how are we",
        "how did we",
        "how was",
        "how were",
        "status",
        "update",
        "review",
        "summary",
    )

    weekly_summary = "weekly summary" in query_l or "week summary" in query_l
    date_summary = any(term in query_l for term in ("last 7 days", "past week", "this week", "last week", "yesterday", "today", "this month", "last month", "whole month")) and any(
        term in query_l for term in ("show", "summary", "review")
    )
    parse_note = None

    combined_requested = any(term in query_l for term in combined_terms)
    vague_requested = any(term in query_l for term in vague_review_terms)
    date_requested = has_explicit_date_range(query)
    source_requested = any(term in query_l for term in source_terms)
    trade_requested = any(term in query_l for term in trade_terms)
    generic_stats_with_window = bool(re.search(r"\bstats?\b", query_l) and date_requested and not source_requested and not trade_requested)

    if weekly_summary or combined_requested or generic_stats_with_window:
        intents.extend(["trade_performance", "source_stats"])
    if date_summary:
        intents.extend(["trade_performance", "source_stats"])
    if "compare" in query_l and len(detect_source_mentions(query_l)) >= 2:
        intents.append("source_stats")
    if source_requested:
        intents.append("source_stats")
    if trade_requested:
        intents.append("trade_performance")
    if source_filter and ("perform" in query_l or "how did" in query_l):
        intents.extend(["trade_performance", "source_stats"])

    ordered_intents = tuple(dict.fromkeys(intents))
    if not ordered_intents:
        return None

    if not date_requested and ("trade_performance" in ordered_intents and "source_stats" in ordered_intents) and (combined_requested or vague_requested):
        date_requested = True
        parse_note = parse_note or "No reporting window was specified, so I used the latest completed Xander window."

    source_date_range = resolve_source_date_range(query, now)
    trade_date_range = resolve_trade_date_range(query, now)
    primary_date_range = trade_date_range if ordered_intents == ("trade_performance",) else source_date_range

    return LucianRequest(
        raw_text=text,
        query=query,
        intents=ordered_intents,
        date_range=primary_date_range,
        source_filter=source_filter,
        date_range_requested=date_requested,
        trade_date_range=trade_date_range,
        source_date_range=source_date_range,
        parse_note=parse_note,
    )


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TRADE HISTORY HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def load_trade_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed_count = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            if isinstance(parsed, dict):
                parsed["_source_line"] = line_number
                records.append(parsed)
            else:
                malformed_count += 1

    return records, malformed_count


def summarize_trade_for_json(trade: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trade:
        return None
    return {
        "Ticker": trade.get("Ticker"),
        "TriggeredDate": trade.get("TriggeredDate"),
        "SoldDate": trade.get("SoldDate"),
        "EntryPrice": trade.get("EntryPrice"),
        "ExitPrice": trade.get("ExitPrice"),
        "PNL": trade.get("PNL"),
        "URL": trade.get("URL"),
        "FGI": trade.get("FGI"),
        "source_line": trade.get("_source_line"),
    }


def trade_label(trade: dict[str, Any] | None) -> str:
    if not trade:
        return "N/A"
    ticker = trade.get("Ticker") or "UNKNOWN"
    pnl = parse_float(trade.get("PNL"))
    date = trade.get("TriggeredDate") or "unknown date"
    if pnl is None:
        return f"{ticker} ({date}, P/L unavailable)"
    return f"{ticker} ({date}, P/L {pnl:.2f})"


def calculate_metrics(
    records: list[dict[str, Any]], start: datetime, end: datetime, malformed_count: int, window_note: str
) -> dict[str, Any]:
    included: list[dict[str, Any]] = []
    skipped_outside = 0
    skipped_missing_trade_date = 0

    for record in records:
        trade_date = parse_date_time(record.get("TriggeredDate"))
        if trade_date is None:
            skipped_missing_trade_date += 1
            skipped_outside += 1
            continue
        if start <= trade_date <= end:
            included.append(record)
        else:
            skipped_outside += 1

    numeric_pnls = [parse_float(trade.get("PNL")) for trade in included]
    numeric_pnls = [pnl for pnl in numeric_pnls if pnl is not None]

    winning = sum(1 for pnl in numeric_pnls if pnl > 0)
    losing = sum(1 for pnl in numeric_pnls if pnl < 0)
    flat = sum(1 for pnl in numeric_pnls if pnl == 0)
    total_trades = len(included)
    win_rate = (winning / total_trades * 100) if total_trades else 0.0
    net_pnl = sum(numeric_pnls)
    average_pnl = (net_pnl / len(numeric_pnls)) if numeric_pnls else 0.0

    trades_with_pnl = [
        trade for trade in included if parse_float(trade.get("PNL")) is not None
    ]
    best_trade = max(trades_with_pnl, key=lambda trade: parse_float(trade.get("PNL")), default=None)
    worst_trade = min(trades_with_pnl, key=lambda trade: parse_float(trade.get("PNL")), default=None)
    ticker_counts = Counter(str(trade.get("Ticker") or "UNKNOWN").upper() for trade in included)

    return {
        "reviewer": "Lucian",
        "generated_at_sgt": datetime.now(SINGAPORE_TZ).isoformat(timespec="seconds"),
        "timezone": "Asia/Singapore",
        "window": {
            "start_sgt": start.isoformat(timespec="seconds"),
            "end_sgt": end.isoformat(timespec="seconds"),
            "note": window_note,
        },
        "source": {
            "trade_history_path": str(Path(os.environ.get("XANDER_TRADE_HISTORY_PATH", str(DEFAULT_TRADE_HISTORY_PATH)))),
            "read_only": True,
        },
        "metrics": {
            "total_trades": total_trades,
            "winning_trades": winning,
            "losing_trades": losing,
            "flat_breakeven_trades": flat,
            "win_rate_percent": round(win_rate, 2),
            "net_pnl": round(net_pnl, 4),
            "average_pnl": round(average_pnl, 4),
            "best_trade_by_pnl": summarize_trade_for_json(best_trade),
            "worst_trade_by_pnl": summarize_trade_for_json(worst_trade),
            "most_traded_tickers": [
                {"ticker": ticker, "trades": count}
                for ticker, count in ticker_counts.most_common(10)
            ],
        },
        "counts": {
            "total_parsed_records": len(records),
            "malformed_line_count": malformed_count,
            "records_skipped_because_outside_window": skipped_outside,
            "records_skipped_missing_triggered_date": skipped_missing_trade_date,
        },
        "trades_included": [summarize_trade_for_json(trade) for trade in included],
    }


def load_trade_metrics_for_range(date_range: DateRange) -> dict[str, Any]:
    trade_history_path = Path(
        os.environ.get("XANDER_TRADE_HISTORY_PATH", str(DEFAULT_TRADE_HISTORY_PATH))
    )
    if not trade_history_path.exists():
        return {
            "available": False,
            "reason": f"Trade history file not found: {trade_history_path}",
            "trade_history_path": str(trade_history_path),
        }

    try:
        records, malformed_count = load_trade_records(trade_history_path)
    except OSError as exc:
        return {
            "available": False,
            "reason": f"Trade history file could not be read: {exc}",
            "trade_history_path": str(trade_history_path),
        }

    metrics_doc = calculate_metrics(
        records,
        date_range.start,
        date_range.end,
        malformed_count,
        date_range.note,
    )
    metrics_doc["available"] = True
    return metrics_doc


def load_default_trade_records() -> tuple[list[dict[str, Any]], int, Path]:
    trade_history_path = Path(
        os.environ.get("XANDER_TRADE_HISTORY_PATH", str(DEFAULT_TRADE_HISTORY_PATH))
    )
    records, malformed_count = load_trade_records(trade_history_path)
    return records, malformed_count, trade_history_path


def filter_trade_records_for_range(
    records: list[dict[str, Any]],
    date_range: DateRange,
    tickers: list[str] | None = None,
) -> list[dict[str, Any]]:
    ticker_filter = {ticker.upper() for ticker in tickers or []}
    included = []
    for record in records:
        ticker = str(record.get("Ticker") or "").upper()
        if ticker_filter and ticker not in ticker_filter:
            continue
        trade_date = parse_date_time(record.get("TriggeredDate"))
        if trade_date is None:
            continue
        if date_range.start <= trade_date <= date_range.end:
            included.append(record)
    return included


def serialize_date_range(date_range: DateRange) -> dict[str, Any]:
    return {
        "start_sgt": date_range.start.isoformat(timespec="seconds"),
        "end_sgt": date_range.end.isoformat(timespec="seconds"),
        "label": date_range.label,
        "note": date_range.note,
        "display": describe_date_range(date_range),
        "completed": date_range.completed,
        "config_sources": date_range.config_sources,
        "config_fallback_used": date_range.config_fallback_used,
    }


def deserialize_date_range(payload: dict[str, Any]) -> DateRange:
    start = parse_date_time(payload.get("start_sgt")) or datetime.now(SINGAPORE_TZ)
    end = parse_date_time(payload.get("end_sgt")) or start
    return DateRange(
        start=start,
        end=end,
        label=str(payload.get("label") or "selected Xander trading window"),
        note=str(payload.get("note") or ""),
        completed=bool(payload.get("completed", True)),
        config_sources=payload.get("config_sources"),
        config_fallback_used=bool(payload.get("config_fallback_used", False)),
    )


def trade_review_context_from_trades(
    trades: list[dict[str, Any]],
    date_range: DateRange,
    trade_history_path: Path,
    reason: str,
) -> dict[str, Any] | None:
    clean_trades = [summarize_trade_for_json(trade) for trade in trades if trade]
    clean_trades = [trade for trade in clean_trades if trade]
    if not clean_trades:
        return None
    tickers = sorted({str(trade.get("Ticker") or "").upper() for trade in clean_trades if trade.get("Ticker")})
    return {
        "type": "trade_review_offer",
        "reason": reason,
        "date_range": serialize_date_range(date_range),
        "tickers": tickers,
        "trade_count": len(clean_trades),
        "trades": clean_trades,
        "trade_history_path": str(trade_history_path),
        "created_at_sgt": datetime.now(SINGAPORE_TZ).isoformat(timespec="seconds"),
    }


def build_trade_review_context_for_request(request: LucianRequest) -> dict[str, Any] | None:
    if "trade_performance" not in request.intents:
        return None
    date_range = request.trade_date_range or request.date_range
    trade_history_path = Path(
        os.environ.get("XANDER_TRADE_HISTORY_PATH", str(DEFAULT_TRADE_HISTORY_PATH))
    )
    if not trade_history_path.exists():
        return None
    try:
        records, _malformed_count = load_trade_records(trade_history_path)
    except OSError:
        return None
    trades = filter_trade_records_for_range(records, date_range)
    return trade_review_context_from_trades(trades, date_range, trade_history_path, "normal_stats_offer")


def build_trade_review_context_from_text(text: str, now: datetime | None = None) -> tuple[dict[str, Any] | None, str | None]:
    query = extract_lucian_query(text) or str(text or "")
    tickers = extract_ticker_mentions(query)
    date_requested = has_explicit_date_range(query)

    trade_history_path = Path(
        os.environ.get("XANDER_TRADE_HISTORY_PATH", str(DEFAULT_TRADE_HISTORY_PATH))
    )
    if not trade_history_path.exists():
        return None, f"Trade history file not found: {trade_history_path}"

    try:
        records, _malformed_count = load_trade_records(trade_history_path)
    except OSError as exc:
        return None, f"Trade history file could not be read: {exc}"

    if tickers and not date_requested:
        ticker_filter = {ticker.upper() for ticker in tickers}
        matches = [
            record for record in records
            if str(record.get("Ticker") or "").upper() in ticker_filter
            and parse_date_time(record.get("TriggeredDate")) is not None
        ]
        matches.sort(key=lambda record: parse_date_time(record.get("TriggeredDate")), reverse=True)
        if not matches:
            return None, f"I could not find a matching closed trade for {', '.join(tickers)}."
        latest = matches[0]
        trade_time = parse_date_time(latest.get("TriggeredDate"))
        date_range = xander_window_containing(trade_time, f"latest {latest.get('Ticker')} Xander trade window")
        trades = [latest]
        return trade_review_context_from_trades(trades, date_range, trade_history_path, "explicit_ticker_review"), None

    date_range = resolve_trade_date_range(query, now)
    trades = filter_trade_records_for_range(records, date_range, tickers)
    if not trades:
        ticker_text = f" for {', '.join(tickers)}" if tickers else ""
        return None, f"I found no closed trades{ticker_text} in {describe_date_range(date_range)}."
    return trade_review_context_from_trades(trades, date_range, trade_history_path, "explicit_window_review"), None


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# SOURCE STATS HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def empty_source_counts() -> dict[str, int]:
    return {field: 0 for field in SOURCE_STATS_FIELDS}


def iter_dates_in_range(start: datetime, end: datetime) -> list[str]:
    start_date = ensure_singapore_time(start).date()
    end_date = ensure_singapore_time(end).date()
    if end_date < start_date:
        return []
    days = []
    current = start_date
    while current <= end_date:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def normalize_source_counts(raw_counts: Any) -> tuple[dict[str, int], list[str]]:
    counts = empty_source_counts()
    missing = []
    if not isinstance(raw_counts, dict):
        return counts, list(SOURCE_STATS_FIELDS)

    for field in SOURCE_STATS_FIELDS:
        value = coerce_int(raw_counts.get(field))
        if value is None:
            missing.append(field)
        else:
            counts[field] = value
    return counts, missing


def add_source_counts(target: dict[str, int], counts: dict[str, int]) -> None:
    for field in SOURCE_STATS_FIELDS:
        target[field] = target.get(field, 0) + counts.get(field, 0)


def load_source_stats(source_filter: str | None = None, date_range: DateRange | None = None) -> dict[str, Any]:
    stats_path = Path(os.environ.get("SOCIALMARKET_STATS_PATH", str(DEFAULT_SOURCE_STATS_PATH)))
    if not stats_path.exists():
        return {
            "available": False,
            "reason": f"Source stats file not found: {stats_path}",
            "stats_path": str(stats_path),
        }

    try:
        with stats_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "reason": f"Source stats file could not be parsed: {exc}",
            "stats_path": str(stats_path),
        }

    if not isinstance(payload, dict):
        return {
            "available": False,
            "reason": "Source stats JSON must contain an object.",
            "stats_path": str(stats_path),
        }

    sources_payload = payload.get("sources")
    daily_payload = payload.get("daily")
    has_sources_payload = isinstance(sources_payload, dict)
    has_daily_payload = isinstance(daily_payload, dict)
    if not has_sources_payload and not has_daily_payload:
        return {
            "available": False,
            "reason": "Source stats JSON does not contain sources or daily buckets.",
            "stats_path": str(stats_path),
        }

    sources: dict[str, dict[str, Any]] = {}
    missing_fields: dict[str, list[str]] = {}
    date_keys: list[str] = []
    missing_date_keys: list[str] = []
    used_daily = False

    if has_daily_payload and (date_range is not None or not has_sources_payload):
        used_daily = True
        date_keys = iter_dates_in_range(date_range.start, date_range.end) if date_range is not None else sorted(str(key) for key in daily_payload.keys())
        for date_key in date_keys:
            raw_day = daily_payload.get(date_key)
            if raw_day is None:
                missing_date_keys.append(date_key)
                continue
            if not isinstance(raw_day, dict):
                missing_date_keys.append(date_key)
                continue

            for raw_source, raw_counts in raw_day.items():
                source = normalize_source_name(str(raw_source)) or str(raw_source)
                if source_filter and source != source_filter:
                    continue
                counts, missing = normalize_source_counts(raw_counts)
                sources.setdefault(source, empty_source_counts())
                add_source_counts(sources[source], counts)
                if missing:
                    missing_fields.setdefault(source, [])
                    missing_fields[source].extend(field for field in missing if field not in missing_fields[source])
    else:
        for raw_source, raw_counts in sources_payload.items():
            source = normalize_source_name(str(raw_source)) or str(raw_source)
            if source_filter and source != source_filter:
                continue
            counts, missing = normalize_source_counts(raw_counts)
            sources[source] = counts
            if missing:
                missing_fields[source] = missing

    totals = {field: sum(counts.get(field, 0) for counts in sources.values()) for field in SOURCE_STATS_FIELDS}
    date_filter_applied = bool(date_range is not None and used_daily)
    if date_range is None and used_daily:
        date_filter_note = "No source stats date range was requested; SOCIALMARKET_Stats.json daily buckets were aggregated."
    elif date_range is None:
        date_filter_note = "No source stats date range was requested; lifetime totals are shown."
    elif used_daily:
        date_filter_note = "Source stats date range was applied using SOCIALMARKET_Stats.json daily buckets."
    else:
        date_filter_note = "SOCIALMARKET_Stats.json has no daily buckets yet; date filtering is unavailable for older aggregate-only data."

    return {
        "available": True,
        "stats_path": str(stats_path),
        "last_updated": payload.get("last_updated"),
        "source_filter": source_filter,
        "aggregate_only": not used_daily,
        "date_filter_applied": date_filter_applied,
        "date_filter_note": date_filter_note,
        "date_keys": date_keys,
        "missing_date_keys": missing_date_keys,
        "totals": totals,
        "sources": sources,
        "missing_fields": missing_fields,
    }


def compact_trade_doc(metrics_doc: dict[str, Any]) -> dict[str, Any]:
    if not metrics_doc.get("available"):
        return metrics_doc
    return {
        "available": True,
        "window": metrics_doc["window"],
        "metrics": metrics_doc["metrics"],
        "counts": metrics_doc["counts"],
    }


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# REVIEWER INTEGRATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def build_lucian_context(request: LucianRequest, logger=None) -> dict[str, Any]:
    trade_date_range = request.trade_date_range or request.date_range
    source_date_range = request.source_date_range or request.date_range
    context: dict[str, Any] = {
        "assistant": "Lucian",
        "query": request.query,
        "intents": list(request.intents),
        "source_filter": request.source_filter,
        "parse_note": request.parse_note,
        "date_range": {
            "label": request.date_range.label,
            "display": describe_date_range(request.date_range),
            "start_sgt": request.date_range.start.isoformat(timespec="seconds"),
            "end_sgt": request.date_range.end.isoformat(timespec="seconds"),
            "note": request.date_range.note,
            "completed": request.date_range.completed,
            "config_sources": request.date_range.config_sources,
            "config_fallback_used": request.date_range.config_fallback_used,
        },
        "trade_date_range": {
            "label": trade_date_range.label,
            "display": describe_date_range(trade_date_range),
            "start_sgt": trade_date_range.start.isoformat(timespec="seconds"),
            "end_sgt": trade_date_range.end.isoformat(timespec="seconds"),
            "note": trade_date_range.note,
            "completed": trade_date_range.completed,
            "config_sources": trade_date_range.config_sources,
            "config_fallback_used": trade_date_range.config_fallback_used,
        },
        "source_date_range": {
            "label": source_date_range.label,
            "display": describe_date_range(source_date_range),
            "start_sgt": source_date_range.start.isoformat(timespec="seconds"),
            "end_sgt": source_date_range.end.isoformat(timespec="seconds"),
            "note": source_date_range.note,
            "completed": source_date_range.completed,
            "config_sources": source_date_range.config_sources,
            "config_fallback_used": source_date_range.config_fallback_used,
        },
    }

    if "trade_performance" in request.intents:
        trade_doc = load_trade_metrics_for_range(trade_date_range)
        if request.source_filter:
            trade_doc["source_filter_note"] = (
                "Trade history records do not store source names, so source-specific "
                "trade filtering is unavailable."
            )
        context["trade_performance"] = compact_trade_doc(trade_doc)

    if "source_stats" in request.intents:
        stats_range = source_date_range if request.date_range_requested else None
        stats_doc = load_source_stats(request.source_filter, stats_range)
        context["source_stats"] = stats_doc

    if logger:
        config_status = xander_trading_time_config_status()
        if config_status["used_default"]:
            logger.warning(
                "Lucian active-window config fallback used: open_source=%s close_source=%s open=%s close=%s",
                config_status["open_source"],
                config_status["close_source"],
                config_status["open_time"],
                config_status["close_time"],
            )
        logger.info(
            "Lucian selected intents=%s trade_window=%s source_window=%s note=%s",
            ",".join(request.intents),
            describe_date_range(trade_date_range),
            describe_date_range(source_date_range),
            request.parse_note,
        )

    return context


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# DEEP TRADE REVIEW HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def parse_bar_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SINGAPORE_TZ)
    return parsed.astimezone(SINGAPORE_TZ)


def parse_price(value: Any) -> float | None:
    parsed = parse_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def format_price(value: Any) -> str:
    parsed = parse_price(value)
    if parsed is None:
        return "N.A."
    return f"${parsed:.2f}"


def format_signed_percent(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "N.A."
    sign = "+" if parsed > 0 else ""
    return f"{sign}{parsed:.2f}%"


def percentage_change(new_value: float | None, base_value: float | None) -> float | None:
    if new_value is None or base_value is None or base_value <= 0:
        return None
    return round((new_value - base_value) / base_value * 100, 4)


def estimate_quantity(trade: dict[str, Any]) -> int | None:
    pnl = parse_float(trade.get("PNL"))
    entry = parse_price(trade.get("EntryPrice"))
    exit_price = parse_price(trade.get("ExitPrice"))
    if pnl is None or entry is None or exit_price is None:
        return None
    per_share = exit_price - entry
    if abs(per_share) < 0.0001:
        return None
    qty = abs(pnl / per_share)
    if qty <= 0:
        return None
    return int(round(qty))


def run_ibkr_historical_helper(symbols: list[str], start: datetime, end: datetime, logger=None) -> dict[str, Any]:
    helper_path = Path(os.environ.get("LUCIAN_IBKR_HISTORICAL_HELPER_PATH", str(DEFAULT_IBKR_HISTORICAL_HELPER_PATH)))
    if not helper_path.exists():
        return {"ok": False, "error": f"IBKR historical helper not found: {helper_path}", "symbols": {}}

    request_payload = {
        "symbols": sorted({symbol.upper() for symbol in symbols if symbol}),
        "start_sgt": ensure_singapore_time(start).isoformat(timespec="seconds"),
        "end_sgt": ensure_singapore_time(end).isoformat(timespec="seconds"),
    }
    timeout = int(os.environ.get("LUCIAN_IBKR_HISTORICAL_TOTAL_TIMEOUT_SECONDS", "75"))
    python_exe = os.environ.get("XANDER_PYTHON_EXE", sys.executable)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp_file:
        json.dump(request_payload, tmp_file)
        request_path = tmp_file.name

    try:
        if logger:
            logger.info(
                "Lucian IBKR historical data request started | symbols=%s window=%s to %s",
                ",".join(request_payload["symbols"]),
                request_payload["start_sgt"],
                request_payload["end_sgt"],
            )
        completed = subprocess.run(
            [python_exe, str(helper_path), request_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        try:
            result = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError:
            result = {"ok": False, "error": f"IBKR helper returned malformed JSON: {stdout[:200]}", "symbols": {}}
        if stderr:
            result["stderr"] = stderr[-500:]
        result["returncode"] = completed.returncode
        if logger:
            logger.info(
                "Lucian IBKR historical data returned | ok=%s returncode=%s",
                result.get("ok"),
                completed.returncode,
            )
        return result
    except subprocess.TimeoutExpired:
        if logger:
            logger.warning("Lucian IBKR historical data request timed out after %ss", timeout)
        return {"ok": False, "error": f"IBKR historical data request timed out after {timeout}s.", "symbols": {}}
    finally:
        try:
            os.remove(request_path)
        except OSError:
            pass


def ibkr_error_is_connection_related(error: str | None) -> bool:
    text = str(error or "").lower()
    return any(term in text for term in ("connect", "connection", "refused", "not connected", "timeout", "timed out", "socket"))


def normalize_bars(raw_bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bars = []
    for raw in raw_bars or []:
        bar_time = parse_bar_time(raw.get("time_sgt"))
        if bar_time is None:
            continue
        close = parse_price(raw.get("close"))
        high = parse_price(raw.get("high"))
        low = parse_price(raw.get("low"))
        if close is None or high is None or low is None:
            continue
        bars.append({
            "time": bar_time,
            "open": parse_price(raw.get("open")),
            "high": high,
            "low": low,
            "close": close,
            "volume": parse_float(raw.get("volume")),
        })
    bars.sort(key=lambda bar: bar["time"])
    return bars


def bars_between(bars: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    return [bar for bar in bars if start <= bar["time"] <= end]


def first_close_at_or_after(bars: list[dict[str, Any]], target: datetime) -> float | None:
    for bar in bars:
        if bar["time"] >= target:
            return bar["close"]
    return None


def classify_trade_replay(metrics: dict[str, Any]) -> tuple[str, str]:
    if not metrics.get("has_sufficient_data"):
        return "Data insufficient", "IBKR did not return enough bars to make a useful post-exit read."

    pnl = parse_float(metrics.get("pnl"))
    recovered_above_entry = metrics.get("recovered_above_entry")
    post_exit_low = parse_price(metrics.get("post_exit_min_low"))
    exit_price = parse_price(metrics.get("exit_price"))
    mfe = parse_float(metrics.get("mfe_pct"))
    mae = parse_float(metrics.get("mae_pct"))

    if recovered_above_entry:
        return "Exited early; price recovered", "Price traded back above entry after exit, so the exit may deserve a closer look."
    if exit_price is not None and post_exit_low is not None and post_exit_low < exit_price:
        return "Exit protected downside", "Price continued below the exit afterward, so the exit likely reduced further downside."
    if pnl is not None and pnl < 0 and mfe is not None and mfe <= 0.2 and mae is not None and mae < -0.5:
        return "No follow-through after signal", "The trade did not move meaningfully in favor before fading."
    if pnl is not None and pnl < 0:
        return "Trade thesis failed", "The trade closed red and did not recover enough after exit to change the read."
    if pnl is not None and pnl > 0:
        return "Exit protected gain", "The trade closed green; the post-exit move did not invalidate the exit."
    return "Data reviewed", "IBKR bars were available, but the post-exit move was mixed."


def calculate_trade_replay_metrics(trade: dict[str, Any], bars: list[dict[str, Any]], review_end: datetime) -> dict[str, Any]:
    ticker = str(trade.get("Ticker") or "UNKNOWN").upper()
    entry_time = parse_date_time(trade.get("TriggeredDate"))
    exit_time = parse_date_time(trade.get("SoldDate"))
    entry_price = parse_price(trade.get("EntryPrice"))
    exit_price = parse_price(trade.get("ExitPrice"))
    pnl = parse_float(trade.get("PNL"))

    metrics: dict[str, Any] = {
        "ticker": ticker,
        "entry_time": entry_time.isoformat(timespec="seconds") if entry_time else None,
        "exit_time": exit_time.isoformat(timespec="seconds") if exit_time else None,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "quantity": trade.get("Quantity") or estimate_quantity(trade),
        "source_line": trade.get("source_line"),
        "has_sufficient_data": False,
    }

    if entry_time is not None and exit_time is not None and exit_time >= entry_time:
        metrics["duration_seconds"] = (exit_time - entry_time).total_seconds()
    else:
        metrics["duration_unavailable_reason"] = "missing_or_invalid_entry_exit_timestamp"

    if entry_time is None or exit_time is None or entry_price is None or exit_price is None:
        metrics["label"], metrics["read"] = "Data insufficient", "Trade history is missing entry/exit time or price."
        return metrics

    entry_to_exit = bars_between(bars, entry_time, exit_time)
    post_exit = bars_between(bars, exit_time, review_end)
    if not entry_to_exit and not post_exit:
        metrics["label"], metrics["read"] = "Data insufficient", "IBKR returned no bars around the trade window."
        return metrics

    if entry_to_exit:
        max_high_entry_exit = max(bar["high"] for bar in entry_to_exit)
        min_low_entry_exit = min(bar["low"] for bar in entry_to_exit)
    else:
        max_high_entry_exit = None
        min_low_entry_exit = None

    if post_exit:
        max_high_after_exit = max(bar["high"] for bar in post_exit)
        min_low_after_exit = min(bar["low"] for bar in post_exit)
        session_close_price = post_exit[-1]["close"]
    else:
        max_high_after_exit = None
        min_low_after_exit = None
        session_close_price = None

    metrics.update({
        "has_sufficient_data": bool(post_exit),
        "price_5m_after_exit": first_close_at_or_after(bars, exit_time + timedelta(minutes=5)),
        "price_15m_after_exit": first_close_at_or_after(bars, exit_time + timedelta(minutes=15)),
        "price_30m_after_exit": first_close_at_or_after(bars, exit_time + timedelta(minutes=30)),
        "price_60m_after_exit": first_close_at_or_after(bars, exit_time + timedelta(minutes=60)),
        "session_close_price": session_close_price,
        "max_high_after_entry_before_exit": max_high_entry_exit,
        "min_low_after_entry_before_exit": min_low_entry_exit,
        "post_exit_max_high": max_high_after_exit,
        "post_exit_min_low": min_low_after_exit,
        "mfe_pct": percentage_change(max_high_entry_exit, entry_price),
        "mae_pct": percentage_change(min_low_entry_exit, entry_price),
        "post_exit_recovery_pct": percentage_change(max_high_after_exit, exit_price),
        "post_exit_continuation_pct": percentage_change(min_low_after_exit, exit_price),
        "recovered_above_entry": bool(max_high_after_exit is not None and max_high_after_exit >= entry_price),
    })
    metrics["label"], metrics["read"] = classify_trade_replay(metrics)
    return metrics


def format_trade_review_metric_line(metrics: dict[str, Any]) -> str:
    checkpoints = [
        f"5m {format_price(metrics.get('price_5m_after_exit'))}",
        f"15m {format_price(metrics.get('price_15m_after_exit'))}",
        f"30m {format_price(metrics.get('price_30m_after_exit'))}",
        f"60m {format_price(metrics.get('price_60m_after_exit'))}",
        f"close {format_price(metrics.get('session_close_price'))}",
    ]
    return ", ".join(checkpoints)


def format_trade_duration(metrics: dict[str, Any]) -> str:
    seconds = parse_float(metrics.get("duration_seconds"))
    if seconds is None or seconds < 0:
        return "unavailable"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


def deterministic_deep_trade_review_text(review_doc: dict[str, Any]) -> str:
    if not review_doc.get("available"):
        return review_doc.get("reason") or TRADE_REVIEW_IBKR_UNAVAILABLE_MESSAGE

    trades = review_doc.get("trades") or []
    if not trades:
        return "⚠️ I found the trade window, but there were no closed trades to review."

    header_count = len(trades)
    lines = [
        f"🔎 I reviewed {header_count} trade{'s' if header_count != 1 else ''} after exit using IBKR historical data.",
        "",
    ]

    labels = Counter()
    for metrics in trades:
        ticker = metrics.get("ticker") or "UNKNOWN"
        label = metrics.get("label") or "Data reviewed"
        labels[label] += 1
        lines.extend(
            [
                ticker,
                f"- Closed at {format_money(metrics.get('pnl'))}. Entry {format_price(metrics.get('entry_price'))}, exit {format_price(metrics.get('exit_price'))}.",
                f"- Duration: {format_trade_duration(metrics)}.",
                f"- MFE / MAE before exit: {format_signed_percent(metrics.get('mfe_pct'))} / {format_signed_percent(metrics.get('mae_pct'))}.",
                f"- After exit: {format_trade_review_metric_line(metrics)}.",
                f"- Label: {label}.",
                f"- Read: {metrics.get('read')}",
                "",
            ]
        )

    if labels:
        summary = "; ".join(f"{count} {label}" for label, count in labels.items())
        lines.extend(["📊 Overall:", f"- {summary}."])

    return "\n".join(lines).strip()


def answer_deep_trade_review(review_context: dict[str, Any], logger=None) -> str:
    if not review_context or not review_context.get("trades"):
        return "⚠️ I can review it, but I need the trade window again. Try: \"Lucian, review yesterday's trades.\""

    date_range = deserialize_date_range(review_context.get("date_range") or {})
    trades = review_context.get("trades") or []
    parsed_entry_times = [parse_date_time(trade.get("TriggeredDate")) for trade in trades]
    parsed_exit_times = [parse_date_time(trade.get("SoldDate")) for trade in trades]
    parsed_entry_times = [value for value in parsed_entry_times if value is not None]
    parsed_exit_times = [value for value in parsed_exit_times if value is not None]
    if not parsed_entry_times or not parsed_exit_times:
        return "⚠️ I found the trade, but the trade history is missing entry or exit timestamps needed for replay."

    request_start = min(parsed_entry_times) - timedelta(minutes=5)
    request_end = max(date_range.end, max(parsed_exit_times) + timedelta(minutes=60))
    symbols = review_context.get("tickers") or [trade.get("Ticker") for trade in trades]

    if logger:
        logger.info(
            "Lucian deep trade review started | tickers=%s window=%s",
            ",".join(symbols),
            review_context.get("date_range", {}).get("display"),
        )
        logger.info("Lucian IBKR availability check started")

    historical = run_ibkr_historical_helper(symbols, request_start, request_end, logger=logger)
    if not historical.get("ok"):
        error = historical.get("error") or historical.get("stderr") or "IBKR historical data unavailable."
        if logger:
            logger.warning("Lucian deep trade review aborted because IBKR historical data failed: %s", error)
        if ibkr_error_is_connection_related(error):
            return TRADE_REVIEW_IBKR_UNAVAILABLE_MESSAGE
        return f"⚠️ I found the trade, but IBKR did not return enough historical price data to review the post-exit move. Detail: {error}"

    if logger:
        logger.info("IBKR connected / historical data usable for Lucian review")

    symbol_payload = historical.get("symbols") or {}
    review_metrics = []
    for trade in trades:
        ticker = str(trade.get("Ticker") or "").upper()
        symbol_data = symbol_payload.get(ticker) or {}
        bars = normalize_bars(symbol_data.get("bars") or [])
        if not bars:
            entry_time = parse_date_time(trade.get("TriggeredDate"))
            exit_time = parse_date_time(trade.get("SoldDate"))
            duration_seconds = (
                (exit_time - entry_time).total_seconds()
                if entry_time is not None and exit_time is not None and exit_time >= entry_time
                else None
            )
            if logger and duration_seconds is None:
                logger.warning("Lucian trade duration unavailable | ticker=%s | reason=missing_or_invalid_entry_exit_timestamp", ticker or "UNKNOWN")
            review_metrics.append({
                "ticker": ticker or "UNKNOWN",
                "entry_time": entry_time.isoformat(timespec="seconds") if entry_time else None,
                "exit_time": exit_time.isoformat(timespec="seconds") if exit_time else None,
                "pnl": parse_float(trade.get("PNL")),
                "entry_price": parse_price(trade.get("EntryPrice")),
                "exit_price": parse_price(trade.get("ExitPrice")),
                "duration_seconds": duration_seconds,
                "duration_unavailable_reason": None if duration_seconds is not None else "missing_or_invalid_entry_exit_timestamp",
                "has_sufficient_data": False,
                "label": "Data insufficient",
                "read": f"IBKR did not return enough historical data for {ticker or 'this trade'}.",
            })
            continue
        metrics = calculate_trade_replay_metrics(trade, bars, request_end)
        if logger and metrics.get("duration_seconds") is None:
            logger.warning(
                "Lucian trade duration unavailable | ticker=%s | reason=%s",
                metrics.get("ticker") or "UNKNOWN",
                metrics.get("duration_unavailable_reason") or "unknown",
            )
        review_metrics.append(metrics)

    if logger:
        logger.info("Lucian trade replay metrics calculated | trades=%s", len(review_metrics))

    return deterministic_deep_trade_review_text({
        "available": True,
        "date_range": review_context.get("date_range"),
        "trades": review_metrics,
    })


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# RESPONSE FORMATTING
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def format_money(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "P/L unavailable"
    if parsed < 0:
        return f"-${abs(parsed):,.2f}"
    return f"${parsed:,.2f}"


def format_signed_money(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "P/L unavailable"
    if parsed < 0:
        return f"-${abs(parsed):,.2f}"
    if parsed > 0:
        return f"+${parsed:,.2f}"
    return "$0.00"


def format_percent(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "N.A."
    if parsed.is_integer():
        return f"{int(parsed)}%"
    return f"{parsed:.2f}%"


def trade_short_label(trade: dict[str, Any] | None) -> str:
    if not trade:
        return "N.A."
    ticker = trade.get("Ticker") or "UNKNOWN"
    return f"{ticker} ({format_signed_money(trade.get('PNL'))})"


def trade_extreme_text(metrics: dict[str, Any]) -> str | None:
    total = metrics.get("total_trades", 0)
    wins = metrics.get("winning_trades", 0)
    losses = metrics.get("losing_trades", 0)
    flat = metrics.get("flat_breakeven_trades", 0)
    best = metrics.get("best_trade_by_pnl")
    worst = metrics.get("worst_trade_by_pnl")

    if total == 0:
        return None
    if wins == 0 and losses == 0 and flat > 0:
        return "No winning or losing trades were recorded; all trades were flat."
    if wins == 0 and losses > 0:
        if losses == 1 and total == 1:
            return f"The only trade closed red: {trade_short_label(worst)}."
        if losses == total:
            if total == 2:
                return (
                    f"Both trades closed red. The smaller loss was {trade_short_label(best)}, "
                    f"while the larger loss was {trade_short_label(worst)}."
                )
            return (
                f"All {total} trades closed red. The smaller loss was {trade_short_label(best)}, "
                f"while the larger loss was {trade_short_label(worst)}."
            )
        return (
            f"There were no winning trades. The smaller loss was {trade_short_label(best)}, "
            f"while the larger loss was {trade_short_label(worst)}."
        )
    if wins > 0 and losses == 0:
        return f"Best trade: {trade_short_label(best)}. No losing trades were recorded."
    return f"Best trade: {trade_short_label(best)}. Worst trade: {trade_short_label(worst)}."


def format_source_breakdown(source: str, counts: dict[str, Any]) -> str:
    total = counts.get("assessed_total", 0)
    bull = counts.get("cat_1_bullish", 0)
    bear = counts.get("cat_1_bearish", 0)
    na = counts.get("cat_1_na", 0)
    cat2 = counts.get("cat_2", 0)

    parts = []
    if bull:
        parts.append(f"{bull} bullish")
    if bear:
        parts.append(f"{bear} bearish")
    if na:
        parts.append(f"{na} N.A.")
    if cat2:
        parts.append(f"{cat2} CAT_2")
    detail = "all CAT_2" if cat2 == total and total else ", ".join(parts) or "no CAT buckets recorded"
    return f"- {source}: {total} assessed, {detail}"


def sentence_window_text(window_text: str) -> str:
    lowered = window_text.lower()
    if lowered.startswith(("latest ", "past ")):
        return f"the {window_text}"
    return window_text


def deterministic_source_stats_text(stats_doc: dict[str, Any], request: LucianRequest, window_text: str | None = None) -> list[str]:
    if not stats_doc.get("available"):
        return [f"⚠️ For post assessment, I could not read the source stats yet: {stats_doc.get('reason', 'unknown issue')}."]

    lines = []
    totals = stats_doc["totals"]
    scope = request.source_filter or "all tracked sources"
    prefix = f"For {sentence_window_text(window_text)}" if window_text else "For post assessment"

    total = totals.get("assessed_total", 0)
    if total == 0:
        if stats_doc.get("date_filter_applied"):
            lines.append(f"🧾 {prefix}, I did not find any recorded source assessments.")
        elif stats_doc.get("date_filter_applied") is False:
            lines.append(
                "⚠️ For post assessment, the stats file does not have usable daily buckets for this window, so I cannot produce a clean historical breakdown yet."
            )
        else:
            lines.append("🧾 For post assessment, I did not find any recorded source assessments.")
    else:
        lines.append(
            "🧾 {prefix}, Xander assessed {total} post{plural}{scope_text}.\n\n"
            "CAT_1 bullish: {bull}\n"
            "CAT_1 bearish: {bear}\n"
            "CAT_1 N.A.: {na}\n"
            "CAT_2: {cat2}".format(
                prefix=prefix,
                scope=scope,
                total=total,
                plural="" if total == 1 else "s",
                scope_text=f" from {scope}" if request.source_filter else "",
                bull=totals.get("cat_1_bullish", 0),
                bear=totals.get("cat_1_bearish", 0),
                na=totals.get("cat_1_na", 0),
                cat2=totals.get("cat_2", 0),
            )
        )

    source_lines = []
    for source, counts in stats_doc.get("sources", {}).items():
        source_lines.append(format_source_breakdown(source, counts))
    if source_lines and not request.source_filter:
        lines.append("\nBy source:\n" + "\n".join(source_lines))

    if stats_doc.get("date_filter_applied") is False and stats_doc.get("date_filter_note"):
        lines.append(stats_doc["date_filter_note"])
    elif stats_doc.get("missing_date_keys") and total == 0:
        lines.append("Some daily buckets for the selected range are not present yet, which may simply mean no live assessments were recorded on those dates.")

    if stats_doc.get("missing_fields"):
        lines.append(f"Some source stat fields are missing: {stats_doc['missing_fields']}")
    return lines


def deterministic_lucian_summary(context: dict[str, Any], request: LucianRequest) -> str:
    trade_window_text = context.get("trade_date_range", {}).get("display") or describe_date_range(request.trade_date_range or request.date_range)
    source_window_text = context.get("source_date_range", {}).get("display") or describe_date_range(request.source_date_range or request.date_range)
    combined = "trade_performance" in request.intents and "source_stats" in request.intents
    lines = []
    data_notes = []

    if request.parse_note:
        data_notes.append(request.parse_note)

    if combined:
        combined_window_text = context.get("date_range", {}).get("display") or describe_date_range(request.date_range)
        lines.append(f"📊 For {sentence_window_text(combined_window_text)}, here's the combined Xander view.")

    trade_doc = context.get("trade_performance")
    if trade_doc:
        if not trade_doc.get("available"):
            lines.append(f"⚠️ Trading: I could not read the trade data yet: {trade_doc.get('reason', 'unknown issue')}.")
        else:
            metrics = trade_doc["metrics"]
            total = metrics["total_trades"]
            trade_lines = []
            if total == 0:
                if combined:
                    trade_lines.extend(
                        [
                            f"- Window: {sentence_window_text(trade_window_text)}",
                            "- Trades: 0",
                            "- Net P/L: $0.00",
                            "- Wins / Losses / Flat: 0 / 0 / 0",
                        ]
                    )
                else:
                    trade_lines.append(f"💼 For {trade_window_text}, no trades were recorded, so there is no P/L to review.")
                counts = trade_doc.get("counts", {})
                parsed = counts.get("total_parsed_records", 0)
                skipped = counts.get("records_skipped_because_outside_window", 0)
                if parsed and skipped >= parsed:
                    data_notes.append(f"I saw {parsed} trade-history record{'s' if parsed != 1 else ''}, but none landed inside the selected trade window.")
            else:
                intro = f"For {sentence_window_text(trade_window_text)}, there {'was' if total == 1 else 'were'} {total} trade{'' if total == 1 else 's'}."
                if not combined:
                    intro = f"💼 {intro}"
                extreme_text = trade_extreme_text(metrics)
                if combined:
                    trade_lines.extend(
                        [
                            f"- Window: {sentence_window_text(trade_window_text)}",
                            f"- Trades: {total}",
                            f"- Net P/L: {format_money(metrics['net_pnl'])}",
                            (
                                "- Wins / Losses / Flat: "
                                f"{metrics['winning_trades']} / {metrics['losing_trades']} / {metrics['flat_breakeven_trades']}"
                            ),
                            f"- Win rate: {format_percent(metrics['win_rate_percent'])}",
                        ]
                    )
                    if extreme_text:
                        trade_lines.append(f"- {extreme_text}")
                else:
                    trade_lines.extend(
                        [
                            intro,
                            "",
                            f"Net P/L: {format_money(metrics['net_pnl'])}",
                            (
                                "Wins / Losses / Flat: "
                                f"{metrics['winning_trades']} / {metrics['losing_trades']} / {metrics['flat_breakeven_trades']}"
                            ),
                            f"Win rate: {format_percent(metrics['win_rate_percent'])}",
                        ]
                    )
                    if extreme_text:
                        trade_lines.extend(["", extreme_text])
            if combined:
                lines.extend(["", "💼 Trading", *trade_lines])
            else:
                lines.extend(trade_lines)
            if trade_doc.get("source_filter_note"):
                data_notes.append(trade_doc["source_filter_note"])

    stats_doc = context.get("source_stats")
    if stats_doc:
        source_lines = deterministic_source_stats_text(stats_doc, request, source_window_text)
        if combined and source_lines and source_lines[0].startswith("🧾 "):
            source_lines[0] = source_lines[0][2:]
        if combined:
            lines.extend(["", "🧾 Post assessment", *source_lines])
        else:
            lines.extend(source_lines)

    if not lines:
        return FALLBACK_UNKNOWN_RESPONSE

    if data_notes:
        lines.extend(["", "⚠️ Data note:", *[f"- {note}" for note in data_notes]])

    return "\n".join(lines)


def clean_lucian_output(text: str) -> str:
    cleaned_lines = []
    skip_prefixes = (
        "lucian here",
        "here's a review",
        "here is a review",
    )
    filler_fragments = (
        "the data is usable for this window",
        "the counts above are the cleanest view",
        "note that there were",
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower().strip("* ")
        if any(lower.startswith(prefix) for prefix in skip_prefixes):
            continue
        if any(fragment in lower for fragment in filler_fragments):
            continue
        cleaned_lines.append(raw_line.rstrip())

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned or text.strip()


def rule_based_lucian_clarification(text: str) -> str:
    query = extract_lucian_query(text) or str(text or "").strip()
    query_l = query.lower()
    source_mentions = detect_source_mentions(query_l)
    date_hint = None
    for label in ("yesterday", "today", "this week", "last week", "past week", "this month", "last month"):
        if label in query_l:
            date_hint = label
            break

    if re.search(r"\bstats?\b", query_l):
        if date_hint:
            return f"I can pull the stats for {date_hint}. Do you want trades, post assessment, or the full combined view?"
        return "I can pull the stats. Do you want trades, post assessment, or the full combined view?"

    if date_hint:
        return f"I can read {date_hint} a couple of ways. Do you want trade performance, post assessment stats, or both?"

    if source_mentions:
        sources = ", ".join(source_mentions)
        return f"I can check {sources}, but I need the angle: source assessment stats, trade performance, or the full combined view?"

    if any(term in query_l for term in ("look", "things", "how was", "how did", "how are")):
        return (
            "I can check that, but I need to know which side you mean: "
            "trades, posts assessed, or the full Xander view."
        )

    return (
        "I can help with that, but I need one more detail. "
        "Do you want trade performance, post assessment stats, or the full combined view?"
    )


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# OLLAMA RESPONSE HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def ollama_text(prompt: str, ollama_url: str, model: str, timeout: int = 20) -> tuple[str | None, str | None]:
    endpoint = ollama_url.rstrip("/") + "/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return None, str(exc)

    text = str(body.get("response", "")).strip()
    if not text:
        return None, "Ollama returned an empty response."
    return text, None


def ollama_lucian_summary(
    context: dict[str, Any], fallback: str, ollama_url: str, model: str, timeout: int = 20
) -> tuple[str, bool, str | None]:
    compact_context = json.dumps(context, indent=2)
    prompt = (
        "You are Lucian, a calm and concise review assistant for the Xander trading bot.\n"
        "Use only the provided JSON. Do not invent missing numbers. Do not give financial advice.\n"
        "If a field says data is unavailable, say that plainly. Keep the response useful and compact.\n"
        "Write like a reviewer speaking to Shawn, not like a raw parser or weekly report template.\n"
        "Use the date_range.display text for the window. Do not put raw ISO timestamps in the Telegram response.\n"
        "Do not say 'Lucian here'. Do not use generic filler such as 'the data is usable for this window'.\n"
        "Do not dump parser field names, skipped-record counters, JSON keys, tables, or bold markdown headings.\n"
        "Do not add a second paragraph that merely repeats the same trade counts or P/L.\n"
        "Use only minimal formal section-level emojis: 📊 for combined summary, 💼 for trading, 🧾 for post assessment, "
        "🔎 for deep review, and ⚠️ for missing data or unavailable dependencies. Do not put emojis on every metric, ticker, or bullet.\n"
        "Use two to four short paragraphs when possible. Exact counts are welcome when they are useful.\n"
        "For trade performance mention number of trades, wins/losses, net P/L, best/worst if available, "
        "and a quick interpretation. For source stats mention assessed counts and CAT buckets.\n"
        "If all data is outside the window or missing, explain that naturally and suggest checking the reporting window.\n\n"
        f"Filtered review JSON:\n{compact_context}\n\n"
        f"Use this fallback summary as the factual floor and style guide:\n{fallback}"
    )
    text, error = ollama_text(prompt, ollama_url, model, timeout=timeout)
    if error or not text:
        return fallback, False, error
    return clean_lucian_output(text), True, None


def ollama_lucian_clarification(
    text: str, fallback: str, ollama_url: str, model: str, timeout: int = 8
) -> tuple[str, bool, str | None]:
    query = extract_lucian_query(text) or str(text or "").strip()
    prompt = (
        "You are Lucian, a concise review assistant for the Xander trading bot.\n"
        "The user asked an unclear request. Do not invent data and do not answer with stats.\n"
        "Ask one short natural clarification question. Offer these choices if useful: "
        "trade performance, post assessment stats, or the full combined Xander view.\n"
        "Keep it under 45 words. Do not say 'Lucian here'.\n\n"
        f"User request: {query}\n\n"
        f"Rule-based fallback to preserve intent:\n{fallback}"
    )
    response, error = ollama_text(prompt, ollama_url, model, timeout=timeout)
    if error or not response:
        return fallback, False, error
    return clean_lucian_output(response), True, None


def answer_lucian_request(
    request: LucianRequest,
    skip_ollama: bool = False,
    logger=None,
) -> str:
    context = build_lucian_context(request, logger=logger)
    fallback = deterministic_lucian_summary(context, request)

    if skip_ollama:
        return fallback

    ollama_url = os.environ.get("LUCIAN_OLLAMA_URL", os.environ.get("XANDER_OLLAMA_URL", DEFAULT_OLLAMA_URL))
    ollama_model = os.environ.get("LUCIAN_OLLAMA_MODEL", os.environ.get("XANDER_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    try:
        ollama_timeout = int(os.environ.get("LUCIAN_OLLAMA_TIMEOUT_SECONDS", "20"))
    except ValueError:
        ollama_timeout = 20
    summary, used, error = ollama_lucian_summary(context, fallback, ollama_url, ollama_model, timeout=ollama_timeout)
    if logger and not used:
        logger.warning("Ollama unavailable; deterministic summary used: %s", error)
    return summary


def answer_unclear_lucian_query(text: str, skip_ollama: bool = False, logger=None) -> str:
    fallback = rule_based_lucian_clarification(text)

    if skip_ollama:
        if logger:
            logger.info("Lucian clarification path=rule_based skip_ollama=True")
        return fallback

    ollama_url = os.environ.get("LUCIAN_OLLAMA_URL", os.environ.get("XANDER_OLLAMA_URL", DEFAULT_OLLAMA_URL))
    ollama_model = os.environ.get("LUCIAN_OLLAMA_MODEL", os.environ.get("XANDER_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    try:
        ollama_timeout = int(os.environ.get("LUCIAN_OLLAMA_CLARIFY_TIMEOUT_SECONDS", os.environ.get("LUCIAN_OLLAMA_TIMEOUT_SECONDS", "8")))
    except ValueError:
        ollama_timeout = 8

    clarification, used, error = ollama_lucian_clarification(text, fallback, ollama_url, ollama_model, timeout=ollama_timeout)
    if logger:
        if used:
            logger.info("Lucian clarification path=ollama")
        else:
            logger.info("Lucian clarification path=rule_based ollama_error=%s", error)
    return clarification


def answer_lucian_query(text: str, skip_ollama: bool = False, logger=None) -> str:
    request = parse_lucian_request(text)
    if request is None:
        return answer_unclear_lucian_query(text, skip_ollama=skip_ollama, logger=logger)
    return answer_lucian_request(request, skip_ollama=skip_ollama, logger=logger)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# WEEKLY CLI OUTPUT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def deterministic_summary(metrics_doc: dict[str, Any]) -> str:
    metrics = metrics_doc["metrics"]
    counts = metrics_doc["counts"]
    window = metrics_doc["window"]
    total = metrics["total_trades"]
    net = metrics["net_pnl"]
    tone = "green" if net > 0 else "red" if net < 0 else "flat"
    best = trade_label(metrics["best_trade_by_pnl"])
    worst = trade_label(metrics["worst_trade_by_pnl"])
    tickers = metrics["most_traded_tickers"]
    ticker_text = ", ".join(f"{item['ticker']} x{item['trades']}" for item in tickers[:5]) or "N/A"

    return "\n".join(
        [
            "Xander Weekly Desk",
            f"Desk check complete for {window['start_sgt']} to {window['end_sgt']}.",
            (
                f"Xander took {total} trade{'s' if total != 1 else ''}; "
                f"{metrics['winning_trades']} won, {metrics['losing_trades']} lost, "
                f"and {metrics['flat_breakeven_trades']} finished flat."
            ),
            (
                f"Net P/L was {net:.2f}, average P/L was {metrics['average_pnl']:.2f}, "
                f"and win rate was {metrics['win_rate_percent']:.2f}%. The week was {tone}."
            ),
            f"Best trade: {best}. Worst trade: {worst}.",
            f"Most traded tickers: {ticker_text}.",
            (
                f"Parsed {counts['total_parsed_records']} records, skipped "
                f"{counts['records_skipped_because_outside_window']} outside the window, "
                f"and ignored {counts['malformed_line_count']} malformed line(s)."
            ),
            "No financial advice, no live trade suggestions. Just the tape, neatly folded.",
        ]
    )


def ollama_summary(
    metrics_doc: dict[str, Any], fallback: str, ollama_url: str, model: str
) -> tuple[str, bool, str | None]:
    compact_metrics = {
        "window": metrics_doc["window"],
        "metrics": metrics_doc["metrics"],
        "counts": metrics_doc["counts"],
    }
    prompt = (
        "You are Xander Weekly Desk, a concise practical trading desk assistant. "
        "Summarize the provided already-calculated weekly trade metrics.\n\n"
        "Rules:\n"
        "- Do not invent numbers.\n"
        "- Use only the provided metrics.\n"
        "- Do not give financial advice.\n"
        "- Do not suggest live trades.\n"
        "- Keep the summary compact.\n\n"
        f"Metrics JSON:\n{json.dumps(compact_metrics, indent=2)}\n\n"
        f"Fallback summary to preserve all facts:\n{fallback}"
    )
    text, error = ollama_text(prompt, ollama_url, model)
    if error or not text:
        return fallback, False, error
    return text, True, None


def main() -> int:
    args = parse_args()

    if args.query:
        print(answer_lucian_query(args.query, skip_ollama=args.skip_ollama))
        return 0

    trade_history_path = Path(
        os.environ.get("XANDER_TRADE_HISTORY_PATH", str(DEFAULT_TRADE_HISTORY_PATH))
    )
    output_dir = Path(
        os.environ.get("XANDER_WEEKLY_REVIEW_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
    )
    ollama_url = os.environ.get("LUCIAN_OLLAMA_URL", os.environ.get("XANDER_OLLAMA_URL", DEFAULT_OLLAMA_URL))
    ollama_model = os.environ.get("LUCIAN_OLLAMA_MODEL", os.environ.get("XANDER_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))

    try:
        override_window = override_review_window(args.start_date, args.end_date)
    except ValueError as exc:
        print(f"Date error: {exc}", file=sys.stderr)
        return 2

    start, end, window_note = override_window or default_review_window()

    if not trade_history_path.exists():
        print(f"Trade history file not found: {trade_history_path}", file=sys.stderr)
        return 1

    records, malformed_count = load_trade_records(trade_history_path)
    metrics_doc = calculate_metrics(records, start, end, malformed_count, window_note)
    fallback = deterministic_summary(metrics_doc)

    ollama_used = False
    ollama_error = None
    summary = fallback
    if not args.skip_ollama:
        summary, ollama_used, ollama_error = ollama_summary(
            metrics_doc, fallback, ollama_url, ollama_model
        )

    metrics_doc["ollama"] = {
        "attempted": not args.skip_ollama,
        "used": ollama_used,
        "url": ollama_url,
        "model": ollama_model,
        "error": ollama_error,
    }

    if not ollama_used:
        summary += "\n\nSummary source: deterministic fallback."
        if ollama_error:
            summary += f"\nOllama note: {ollama_error}"
    else:
        summary += "\n\nSummary source: local Ollama."

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "weekly_trade_metrics.json"
    summary_path = output_dir / "weekly_trade_summary.txt"

    metrics_path.write_text(json.dumps(metrics_doc, indent=2), encoding="utf-8")
    summary_path.write_text(summary + "\n", encoding="utf-8")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Ollama used: {ollama_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
