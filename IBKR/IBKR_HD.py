#!/usr/bin/env python
"""IBKR-only historical bar helper for Lucian trade review.

This script intentionally has no public market-data fallback. If IBKR is not
connected or does not return bars, the caller receives a structured error.

AI status: Created with AI.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

if sys.version_info >= (3, 10):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

try:
    from ib_insync import IB, Stock
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"IBKR Python dependency unavailable: {exc}", "symbols": {}}))
    raise SystemExit(1)


HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORT = int(os.getenv("IBKR_PORT", "4001"))
CLIENT_ID = int(os.getenv("LUCIAN_IBKR_HISTORICAL_CLIENT_ID", str(7100 + (os.getpid() % 800))))
SGT = ZoneInfo(os.getenv("XANDER_TIMEZONE", "Asia/Singapore"))


def parse_sgt(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SGT)
    return parsed.astimezone(SGT)


def duration_for_range(start, end):
    seconds = max(60, int((end - start).total_seconds()) + 300)
    days = max(1, int(seconds / 86400) + 1)
    if days <= 7:
        return f"{days} D"
    weeks = max(1, int(days / 7) + 1)
    return f"{weeks} W"


def bar_to_dict(bar):
    bar_time = bar.date
    if isinstance(bar_time, datetime):
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=ZoneInfo("UTC"))
        bar_time = bar_time.astimezone(SGT).isoformat(timespec="seconds")
    else:
        bar_time = str(bar_time)
    return {
        "time_sgt": bar_time,
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(getattr(bar, "volume", 0) or 0),
    }


def fetch_symbol_bars(ib, symbol, start_sgt, end_sgt):
    contract = Stock(symbol.strip().upper(), "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        return {"ok": False, "error": "IBKR contract qualification returned no match.", "bars": []}

    end_utc = end_sgt.astimezone(ZoneInfo("UTC"))
    bars = ib.reqHistoricalData(
        qualified[0],
        endDateTime=end_utc.strftime("%Y%m%d %H:%M:%S UTC"),
        durationStr=duration_for_range(start_sgt, end_sgt),
        barSizeSetting=os.getenv("LUCIAN_IBKR_HISTORICAL_BAR_SIZE", "1 min"),
        whatToShow=os.getenv("LUCIAN_IBKR_HISTORICAL_WHAT_TO_SHOW", "TRADES"),
        useRTH=False,
        formatDate=2,
        keepUpToDate=False,
        timeout=float(os.getenv("LUCIAN_IBKR_HISTORICAL_REQUEST_TIMEOUT_SECONDS", "20")),
    )

    filtered = []
    for bar in bars:
        bar_time = bar.date
        if isinstance(bar_time, datetime):
            if bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=ZoneInfo("UTC"))
            bar_sgt = bar_time.astimezone(SGT)
            if start_sgt <= bar_sgt <= end_sgt:
                filtered.append(bar_to_dict(bar))
        else:
            filtered.append(bar_to_dict(bar))

    return {"ok": bool(filtered), "error": None if filtered else "IBKR returned no bars in the requested window.", "bars": filtered}


def main(request_path):
    with open(request_path, "r", encoding="utf-8") as handle:
        request = json.load(handle)

    symbols = request.get("symbols") or []
    start_sgt = parse_sgt(request["start_sgt"])
    end_sgt = parse_sgt(request["end_sgt"])

    result = {
        "ok": False,
        "client_id": CLIENT_ID,
        "host": HOST,
        "port": PORT,
        "symbols": {},
    }

    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=float(os.getenv("LUCIAN_IBKR_CONNECT_TIMEOUT_SECONDS", "8")))
        if not ib.isConnected():
            result["error"] = "IBKR connection check failed."
            print(json.dumps(result))
            return 2

        for symbol in symbols:
            clean_symbol = str(symbol).strip().upper()
            if not clean_symbol:
                continue
            try:
                result["symbols"][clean_symbol] = fetch_symbol_bars(ib, clean_symbol, start_sgt, end_sgt)
            except Exception as exc:
                result["symbols"][clean_symbol] = {"ok": False, "error": str(exc), "bars": []}

        result["ok"] = any(item.get("ok") for item in result["symbols"].values())
        if not result["ok"]:
            result["error"] = "IBKR historical data returned no usable bars."
        print(json.dumps(result))
        return 0 if result["ok"] else 3
    except Exception as exc:
        result["error"] = str(exc)
        print(json.dumps(result))
        return 1
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "Usage: python IBKR_HD.py <request.json>"}))
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
