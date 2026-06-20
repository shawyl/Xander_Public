"""Broker market-data quote helper for Xander.

Requests a short-lived quote snapshot from the broker integration layer and
returns structured data to the caller. This helper is operational plumbing and
does not contain public strategy guidance.

AI status: Maintained with AI.
"""

import sys
import asyncio
import os
if sys.version_info >= (3, 10):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import *
import json

def _client_id():
    return 2000 + (os.getpid() % 6000)

def _wait_for_quote(ib: IB, ticker, timeout):
    waited = 0.0
    step = 0.2
    while waited < timeout:
        if (ticker.bid and ticker.bid > 0) or (ticker.ask and ticker.ask > 0) or (ticker.last and ticker.last > 0):
            return True
        ib.sleep(step)
        waited += step
    return False

def main(ticker_symbol):
    ib = IB()
    client_id = _client_id()
    try:
        ib.connect('127.0.0.1', 4001, clientId=client_id)

        contract = Stock(ticker_symbol.strip().upper(), "SMART", "USD")
        ib.qualifyContracts(contract)

        ib.reqMarketDataType(1)
        t_live = ib.reqMktData(contract, '', False, False)

        got_live = _wait_for_quote(ib, t_live, 10.0)

        ib.cancelMktData(contract)
        ib.disconnect()

        if got_live:
            result = {
                "ticker": ticker_symbol.upper(),
                "client_id": client_id,
                "bid": t_live.bid or 0.0,
                "ask": t_live.ask or 0.0,
                "last": t_live.last or 0.0
            }
            print(json.dumps(result))  # Output clean JSON to stdout
            sys.exit(0)
        else:
            print(json.dumps({"error": "Quote timeout", "ticker": ticker_symbol.upper(), "client_id": client_id}))
            sys.exit(2)

    except Exception as e:
        print(json.dumps({"error": str(e), "ticker": ticker_symbol.upper(), "client_id": client_id}))
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python IBKR_MarketData.py <TICKER>"}))
        sys.exit(1)

    main(sys.argv[1])
