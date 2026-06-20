"""Test-only Lucian wrapper.

Production Lucian is initialized, routed, scheduled, and controlled by
launch_xander.py / Launch Center. This file intentionally does not start a
Telegram polling listener, so it cannot create a second competing Lucian
runtime path.

AI status: Created with AI.
"""

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TEST WRAPPER CONFIGURATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(SCRIPT_DIR / ".env")

import lucian_reviewer as reviewer


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# CLI HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-shot Lucian reviewer query for testing. "
            "Production Telegram handling lives in launch_xander.py."
        )
    )
    parser.add_argument(
        "--query",
        help='Lucian-style query, for example: "Lucian, all stats this week".',
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Use deterministic reviewer output without calling local Ollama.",
    )
    return parser.parse_args()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TEST ENTRYPOINT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def main():
    args = parse_args()

    if not args.query:
        print("Lucian production runtime is owned by launch_xander.py / Launch Center.")
        print('For a one-shot local test, run: launch_lucian.py --query "Lucian, all stats this week"')
        return 0

    print(reviewer.answer_lucian_query(args.query, skip_ollama=args.skip_ollama))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ENTRYPOINT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    raise SystemExit(main())
