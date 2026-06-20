"""Fear and Greed Index helper for Xander.

Fetches a simple market-sentiment value for use by the broader workflow and
prints a safe fallback when the value is unavailable.

AI status: Maintained with AI.
"""

import fear_and_greed

def get_fgi_value():
    try:
        value = fear_and_greed.get().value
        if value is None or str(value).strip() == "":
            print("N.A.", flush=True)
        else:
            print(value, flush=True)
    except Exception:
        print("N.A.", flush=True)

if __name__ == "__main__":
    get_fgi_value()
