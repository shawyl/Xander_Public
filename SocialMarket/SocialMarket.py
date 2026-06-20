"""Main market-processing workflow for Xander.

Handles source processing, AI assessment orchestration, alert generation, and
downstream handoff. Sensitive classification and execution details are kept
inside the implementation and should not be exposed in public documentation.

AI status: Maintained with AI.
"""

import requests
from datetime import datetime, time, timedelta
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import time as time_module
import os
import threading
import logging
import random
import platform
import shutil
import re
import ast
import pytz
import string
import hashlib
import warnings
import contextlib
import sys
import importlib
import importlib.util

sys.dont_write_bytecode = True

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# IMPORT / WARNING SUPPRESSION HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
HF_HUB_UNAUTH_WARNING = "You are sending unauthenticated requests to the HF Hub"
_original_logging_log = logging.Logger._log

def _suppress_hf_hub_warning_log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
    if HF_HUB_UNAUTH_WARNING in str(msg):
        return
    return _original_logging_log(self, level, msg, args, exc_info, extra, stack_info, stacklevel)

logging.Logger._log = _suppress_hf_hub_warning_log
_original_showwarning = warnings.showwarning

def _suppress_hf_hub_showwarning(message, category, filename, lineno, file=None, line=None):
    if HF_HUB_UNAUTH_WARNING in str(message):
        return
    return _original_showwarning(message, category, filename, lineno, file, line)

warnings.showwarning = _suppress_hf_hub_showwarning

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

warnings.filterwarnings(
    "ignore",
    message=r".*You are sending unauthenticated requests to the HF Hub.*",
)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import huggingface_hub
    from huggingface_hub.cli import _output as hf_output
    hf_output.Output.warning = lambda self, message: None
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The sentence_transformers package is required. Install it with 'pip install sentence-transformers' "
        "and restart the bot."
    ) from e
import pandas as pd
from rapidfuzz import process, fuzz
import queue
import traceback
import yfinance as yf
import subprocess
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv
try:
    from SocialMarket_DPB import build_dynamic_prompt
except ModuleNotFoundError:
    from SocialMarket.SocialMarket_DPB import build_dynamic_prompt


def clear_python_bytecode_cache(root_path):
    removed = 0
    if not root_path or not os.path.exists(root_path):
        return removed

    for current_root, dirs, files in os.walk(root_path):
        for dirname in list(dirs):
            if dirname != "__pycache__":
                continue
            cache_path = os.path.join(current_root, dirname)
            try:
                shutil.rmtree(cache_path)
                removed += 1
            except Exception:
                pass
        for filename in files:
            if not filename.endswith(".pyc"):
                continue
            pyc_path = os.path.join(current_root, filename)
            try:
                os.remove(pyc_path)
                removed += 1
            except Exception:
                pass

    return removed


def clear_package_bytecode_cache(package_name):
    try:
        spec = importlib.util.find_spec(package_name)
    except Exception:
        return 0

    if spec is None:
        return 0

    roots = []
    if spec.submodule_search_locations:
        roots.extend(str(path) for path in spec.submodule_search_locations)
    elif spec.origin:
        roots.append(os.path.dirname(spec.origin))

    return sum(clear_python_bytecode_cache(root) for root in roots)


def import_telethon_with_bytecode_recovery():
    try:
        from telethon import TelegramClient as ImportedTelegramClient, events as imported_events
        return ImportedTelegramClient, imported_events
    except EOFError as exc:
        if "marshal data too short" not in str(exc).lower():
            raise

        removed = clear_package_bytecode_cache("telethon")
        removed += clear_python_bytecode_cache(os.path.dirname(__file__))
        for module_name in list(sys.modules):
            if module_name == "telethon" or module_name.startswith("telethon."):
                sys.modules.pop(module_name, None)
        importlib.invalidate_caches()

        try:
            from telethon import TelegramClient as ImportedTelegramClient, events as imported_events
            print(f"Recovered Telethon import after clearing {removed} Python bytecode cache item(s).")
            return ImportedTelegramClient, imported_events
        except EOFError as retry_exc:
            raise RuntimeError(
                "Telethon import failed because Python read corrupted bytecode "
                "('EOFError: marshal data too short'). This is usually a truncated .pyc file, "
                "often caused by interrupted writes or OneDrive syncing __pycache__. "
                "Recovery: delete __pycache__ folders under .BOT_Launch and Telethon's site-packages, "
                "then reinstall Telethon if the import still fails. Also confirm launch_xander.py is "
                f"starting SocialMarket with the intended interpreter: {sys.executable}"
            ) from retry_exc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Telethon is not installed in the Python environment running SocialMarket. "
            f"Interpreter: {sys.executable}. Install it in this environment with: "
            f'"{sys.executable}" -m pip install --upgrade telethon'
        ) from exc


TelegramClient, events = import_telethon_with_bytecode_recovery()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# CONFIGURATION / ENVIRONMENT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Load environment variables from a .env file (if present)
load_dotenv()

# PROD ON / OFF ?
isLocal = True # Not required anymore since running on local PC

BOTHUB_ROOT = os.getenv("BOTHUB_ROOT")
if BOTHUB_ROOT is None or BOTHUB_ROOT.strip() == "":
    raise RuntimeError("Missing required environment variable: BOTHUB_ROOT")

def bot_path(*parts):
    return os.path.join(BOTHUB_ROOT, *parts)

def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

def require_int_env(name: str) -> int:
    value = require_env(name)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc

def get_env_time(name):
    value = require_env(name).strip()
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be HH:MM, got {value!r}") from exc

# Global Times (no need for trading hours)
XANDER_TIMEZONE = require_env("XANDER_TIMEZONE")
marketOpen = get_env_time("XANDER_MARKET_OPEN_SGT")
marketClose = get_env_time("XANDER_MARKET_CLOSE_SGT")
marketHalfDay = get_env_time("XANDER_MARKET_HALF_DAY_CLOSE_SGT")
stockTitanDumpStart = get_env_time("XANDER_STOCKTITAN_DUMP_START_SGT")

# OpenAi API key
client = OpenAI(api_key=require_env("OPENAI_API_KEY"))

# Telegram bot (Xander)
bot_token = require_env("XANDER_TELEGRAM_BOT_TOKEN")
chat_id = require_int_env("XANDER_TELEGRAM_CHAT_ID")
MAX_WORKERS = 1          # max concurrent Telegram sends
QUEUE_SIZE = 3000         # max queued messages
telegram_queue = queue.Queue(maxsize=QUEUE_SIZE)
telegram_session = requests.Session()
telegram_session_lock = threading.Lock()
telegram_health_lock = threading.Lock()
telegram_worker_threads = []
telegram_force_restart_last_requested = 0.0
TELEGRAM_CONNECT_TIMEOUT_SECONDS = int(os.getenv("SOCIALMARKET_TELEGRAM_CONNECT_TIMEOUT_SECONDS", "5"))
TELEGRAM_SEND_TIMEOUT_SECONDS = int(os.getenv("SOCIALMARKET_TELEGRAM_SEND_TIMEOUT_SECONDS", "10"))
TELEGRAM_MAX_SEND_ATTEMPTS = int(os.getenv("SOCIALMARKET_TELEGRAM_MAX_SEND_ATTEMPTS", "3"))
TELEGRAM_MAX_CONSECUTIVE_FAILURES = int(os.getenv("SOCIALMARKET_TELEGRAM_MAX_CONSECUTIVE_FAILURES", "5"))
TELEGRAM_FORCE_RESTART_COOLDOWN_SECONDS = int(os.getenv("SOCIALMARKET_TELEGRAM_FORCE_RESTART_COOLDOWN_SECONDS", "300"))
TELEGRAM_QUEUE_WARNING_SIZE = int(os.getenv("SOCIALMARKET_TELEGRAM_QUEUE_WARNING_SIZE", str(int(QUEUE_SIZE * 0.8))))
TELEGRAM_WORKER_STUCK_SECONDS = int(os.getenv("SOCIALMARKET_TELEGRAM_WORKER_STUCK_SECONDS", "180"))
TELEGRAM_HEALTH_CHECK_SECONDS = int(os.getenv("SOCIALMARKET_TELEGRAM_HEALTH_CHECK_SECONDS", "30"))
telegram_health = {
    "last_success_monotonic": None,
    "last_success_at": None,
    "last_failure_at": None,
    "last_exception": None,
    "consecutive_failures": 0,
    "in_flight_started_at": None,
    "in_flight_label": None,
}

# Process structure (X ==> X, TS ==> Truth Social)
processes = [
    {"user" : "DeItaone" , "display" : "*Walter Bloomberg" , "type" : "X" , "platform" : "X" , "intervals" : [0, 15, 30, 45], "instance" : ["http://localhost:8081/"] , "scrapeFailCount" : 0},
    #{"user" : "FirstSquawk" , "display" : "First Squawk" , "type" : "X" , "platform" : "X" , "intervals" : [0, 10, 20, 30, 40, 50], "instance" : ["http://localhost:8083/"] , "scrapeFailCount" : 0},
    {"user" : "realDonaldTrump" , "display" : "Donald J. Trump" , "type" : "X" , "platform" : "X" , "intervals" : [0, 15, 30, 45], "instance" : ["http://localhost:8083/"] , "scrapeFailCount" : 0}, 
    {"user" : "realDonaldTrump" , "display" : "Donald J. Trump" , "type" : "TS" , "platform" : "Truth Social" , "intervals" : [0, 15, 30, 45], "instance" : ["https://trumpstruth.org/"] , "scrapeFailCount" : 0},
    #{"user" : "Benzinga" , "display" : "Benzinga" , "type" : "Benzinga" , "platform" : "Benzinga" , "intervals" : [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], "instance" : ["https://api.benzinga.com/api/v2/news/"] , "scrapeFailCount" : 0},
    {"user" : "StockTitan" , "display" : "StockTitan" , "type" : "StockTitan" , "platform" : "StockTitan" , "intervals" : [0, 20, 40], "instance" : ["https://www.stocktitan.net/rss"] , "scrapeFailCount" : 0}
]

# Telegram Polling
API_ID = require_int_env("TELEGRAM_API_ID")
API_HASH = require_env("TELEGRAM_API_HASH")

TELEGRAM_SOURCE = "WalterBloomberg"
telegram_client = TelegramClient("session_walter", API_ID, API_HASH)

# Folder directories
functionsFolder_path = bot_path(".BOT_Launch", "Functions")
validationFolder_path = bot_path("_OUTPUT", "XANDER", "VALIDATE")
logsFolder_path = bot_path("_OUTPUT", "XANDER", "LOGS")
ibkrFolder_path = bot_path(".BOT_Launch", "IBKR", "INCOMING")
nrFolder_path = bot_path("_OUTPUT", "XANDER", "NON-REGRESSION")
nrReviewFolder_path = bot_path("_OUTPUT", "XANDER", "NR_REVIEW")
feeds_path = bot_path(".BOT_Launch", "SocialMarket", "FEEDS")
cat1_posts_path = bot_path(".BOT_Launch", "SocialMarket", "POSTS")
prompts_path = bot_path(".BOT_Launch", "SocialMarket", "PROMPTS")
marketContext_snapshot_path = bot_path(".BOT_Launch", "Functions", "XANDER_MarketContext_Snapshot.txt")

# Establish logging (socialmarket_logger.info(f''))
os.makedirs(logsFolder_path, exist_ok=True)
current_date = datetime.now().strftime('%d-%m-%Y')
log_filename = f"SocialMarket-{current_date}.log"
log_filepath = os.path.join(logsFolder_path, log_filename)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
socialmarket_logger = logging.getLogger('socialmarket_logger')
socialmarket_logger.setLevel(logging.INFO)
socialmarket_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
socialmarket_handler.setLevel(logging.INFO)
socialmarket_handler.setFormatter(formatter)
socialmarket_logger.addHandler(socialmarket_handler)

# Stage 2 GPT Prompts
stage2_gpt4omini_prompts = ["OPTIONS", "CONTRACTS",  "INDEX_INCLUSION"]
stage2_gpt41mini_prompts = ["CAPITAL_STRUCTURE", "PARTNERSHIPS", "MACRODATA", "MACROINDICATOR", "ACQUISITIONS", "VALIDATIONS"]
stage2_gpt4o_prompts = ["EARNINGS", "DEADLINE", "FED", "POLICY", "GEOPOLITICAL", "TRUMP", "OTHERS"]
stage2_gpt4o_custom_prompts = ["DEADLINE", "FED", "POLICY", "GEOPOLITICAL", "TRUMP", "OTHERS"]
stage2_marketContext_prompts = ["DEADLINE", "FED", "POLICY", "GEOPOLITICAL", "TRUMP"]

# Misc
sent_cache = set()
sent_cache_lock = threading.Lock()
assessed_posts = set()
assessed_posts_lock = threading.Lock()
MAX_ASSESSED = 5000
current_files = []
lock = threading.Lock()
validation_lock = threading.Lock()
socialmarket_stats_lock = threading.Lock()
socialmarket_stats_queue = queue.Queue(maxsize=1000)
SOCIALMARKET_STATS_FILENAME = "SOCIALMARKET_Stats.json"
SOCIALMARKET_STATS_PATH = os.path.join(functionsFolder_path, SOCIALMARKET_STATS_FILENAME)
SOCIALMARKET_STATS_SOURCES = ("Walter", "Trump", "Stock Titan")
SOCIALMARKET_STATS_COUNTERS = (
    "assessed_total",
    "cat_1_bullish",
    "cat_1_bearish",
    "cat_1_na",
    "cat_2",
)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# SOCIALMARKET STATS HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def initialize_socialmarket_stats():
    return {
        "last_updated": None,
        "sources": {
            source: {counter: 0 for counter in SOCIALMARKET_STATS_COUNTERS}
            for source in SOCIALMARKET_STATS_SOURCES
        },
        "daily": {},
    }

def initialize_socialmarket_source_counters():
    return {counter: 0 for counter in SOCIALMARKET_STATS_COUNTERS}

def initialize_socialmarket_daily_bucket():
    return {
        source: initialize_socialmarket_source_counters()
        for source in SOCIALMARKET_STATS_SOURCES
    }

def normalize_socialmarket_source(source):
    if isinstance(source, dict):
        candidates = [
            source.get("user"),
            source.get("display"),
            source.get("platform"),
            source.get("type"),
        ]
    else:
        candidates = [source]

    normalized_candidates = [
        str(candidate or "").strip().lower().replace("*", "").replace("_", " ")
        for candidate in candidates
    ]

    if any(candidate in ("deitaone", "walter", "walter bloomberg", "walterbloomberg") for candidate in normalized_candidates):
        return "Walter"
    if any(candidate in ("realdonaldtrump", "trump", "donald j. trump", "truth social", "ts") for candidate in normalized_candidates):
        return "Trump"
    if any(candidate in ("stocktitan", "stock titan") for candidate in normalized_candidates):
        return "Stock Titan"

    return None

def normalize_assessment_direction(direction):
    if isinstance(direction, list):
        sentiments = [
            str(item.get("Sentiment", "")).strip().lower()
            for item in direction
            if isinstance(item, dict)
        ]
    else:
        sentiments = [str(direction or "").strip().lower()]

    directional_sentiments = {
        sentiment
        for sentiment in sentiments
        if sentiment in ("bullish", "bearish")
    }

    if directional_sentiments == {"bullish"}:
        return "bullish"
    if directional_sentiments == {"bearish"}:
        return "bearish"
    return "na"

def get_socialmarket_counter_bucket(final_category, direction):
    category = str(final_category or "").strip().upper()

    if category == "CAT_2":
        return "cat_2"
    if category != "CAT_1":
        return None

    direction = normalize_assessment_direction(direction)
    if direction == "bullish":
        return "cat_1_bullish"
    if direction == "bearish":
        return "cat_1_bearish"
    return "cat_1_na"

def should_record_socialmarket_stats(is_live_scraping, source, final_category):
    if not is_live_scraping:
        return False
    if normalize_socialmarket_source(source) is None:
        return False
    return get_socialmarket_counter_bucket(final_category, None) is not None

def load_socialmarket_stats(path):
    stats = initialize_socialmarket_stats()

    if not os.path.exists(path):
        return stats

    try:
        with open(path, "r", encoding="utf-8") as f:
            existing_stats = json.load(f)
    except Exception as exc:
        socialmarket_logger.warning(f"Failed to load {SOCIALMARKET_STATS_FILENAME}; reinitializing stats: {exc}")
        return stats

    stats["last_updated"] = existing_stats.get("last_updated")
    existing_sources = existing_stats.get("sources", {})
    existing_daily = existing_stats.get("daily", {})

    if isinstance(existing_sources, dict):
        for source in SOCIALMARKET_STATS_SOURCES:
            existing_counters = existing_sources.get(source, {})
            if not isinstance(existing_counters, dict):
                continue
            for counter in SOCIALMARKET_STATS_COUNTERS:
                try:
                    stats["sources"][source][counter] = int(existing_counters.get(counter, 0))
                except (TypeError, ValueError):
                    stats["sources"][source][counter] = 0

    if isinstance(existing_daily, dict):
        for date_key, raw_day in existing_daily.items():
            try:
                datetime.strptime(str(date_key), "%Y-%m-%d")
            except ValueError:
                continue
            if not isinstance(raw_day, dict):
                continue

            day_bucket = initialize_socialmarket_daily_bucket()
            for source in SOCIALMARKET_STATS_SOURCES:
                raw_counts = raw_day.get(source, {})
                if not isinstance(raw_counts, dict):
                    continue
                for counter in SOCIALMARKET_STATS_COUNTERS:
                    try:
                        day_bucket[source][counter] = int(raw_counts.get(counter, 0))
                    except (TypeError, ValueError):
                        day_bucket[source][counter] = 0
            stats["daily"][str(date_key)] = day_bucket

    return stats

def atomic_write_json(path, payload, retries=3, retry_delay=0.1):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(retries):
        tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.{attempt}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")

            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time_module.sleep(retry_delay)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

def record_socialmarket_stats(source, final_category, direction, is_live_scraping=False):
    try:
        normalized_source = normalize_socialmarket_source(source)
        bucket = get_socialmarket_counter_bucket(final_category, direction)

        if not is_live_scraping or normalized_source is None or bucket is None:
            return

        socialmarket_stats_queue.put_nowait((normalized_source, bucket))
    except queue.Full:
        socialmarket_logger.warning(f"Skipped {SOCIALMARKET_STATS_FILENAME} update: stats queue is full")
    except Exception as exc:
        socialmarket_logger.warning(f"Failed to queue {SOCIALMARKET_STATS_FILENAME} update: {exc}")

def socialmarket_stats_worker():
    while True:
        normalized_source, bucket = socialmarket_stats_queue.get()
        try:
            increment_socialmarket_stats(normalized_source, bucket)
        except Exception as exc:
            socialmarket_logger.warning(f"Failed to update {SOCIALMARKET_STATS_FILENAME}: {exc}")
        finally:
            socialmarket_stats_queue.task_done()

def increment_socialmarket_stats(normalized_source, bucket):
    try:
        with socialmarket_stats_lock:
            stats = load_socialmarket_stats(SOCIALMARKET_STATS_PATH)
            now = datetime.now()
            date_key = now.date().isoformat()
            day_bucket = stats.setdefault("daily", {}).setdefault(date_key, initialize_socialmarket_daily_bucket())
            day_source = day_bucket.setdefault(normalized_source, initialize_socialmarket_source_counters())

            stats["sources"][normalized_source]["assessed_total"] += 1
            stats["sources"][normalized_source][bucket] += 1
            day_source["assessed_total"] += 1
            day_source[bucket] += 1
            stats["last_updated"] = now.isoformat(timespec="seconds")
            atomic_write_json(SOCIALMARKET_STATS_PATH, stats)
    except Exception as exc:
        socialmarket_logger.warning(f"Failed to update {SOCIALMARKET_STATS_FILENAME}: {exc}")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# MODEL / TICKER DATA INITIALIZATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def load_sentence_transformer_model(model_name: str):
    @contextlib.contextmanager
    def _suppress_stderr():
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull_fd, 2)
            yield
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
            os.close(devnull_fd)

    with _suppress_stderr():
        return SentenceTransformer(model_name)

MODEL = load_sentence_transformer_model('all-MiniLM-L6-v2')
df = pd.read_csv(bot_path(".BOT_Launch", "SocialMarket", "TICKERS", "combined_unique_ticker_list.csv"))
df = df[df["Security Name"].notna()]
df = df[df["Security Name"].str.strip().str.len() > 3]
name_to_ticker = dict(zip(df["Security Name"], df["Symbol"]))
lower_name_map = {name.lower(): ticker for name, ticker in name_to_ticker.items()}
USER_AGENTS = [
    # Chrome - Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.199 Safari/537.36",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.224 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.160 Safari/537.36",

    # Firefox - Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:110.0) Gecko/20100101 Firefox/110.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",

    # Edge - Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.90 Safari/537.36 Edg/118.0.2088.61",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.110 Safari/537.36 Edg/116.0.1938.62",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.2365.66",
]
approved_sectors = [
    "Technology",
    "Semiconductors",
    "Financials",
    "Healthcare",
    "Consumer",
    "Defense",
    "Industrial",
    "Real Estate",
    "Energy",
    "Evs",
    "Others"
]

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# LOGGING / DATE HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def timePrint(log):
    sgt_now = datetime.now(pytz.timezone(XANDER_TIMEZONE))
    formatted_time = sgt_now.strftime('%d-%b-%y %I:%M:%S %p')  # e.g., 10-May-25 01:22:43 AM
    print(f"[{formatted_time}] {log}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def parse_and_adjust_datetime(date_str, input_format, offset_hours, platformName):
    try:
        dt_obj = datetime.strptime(date_str, input_format)
        dt_adjusted = dt_obj + timedelta(hours=offset_hours)

        # Platform-specific formatting to remove leading zero in hour
        if platform.system() == "Windows":
            formatted = dt_adjusted.strftime("%d-%b-%y %#I:%M%p")
        else:
            formatted = dt_adjusted.strftime("%d-%b-%y %-I:%M%p")

        return formatted
    except Exception as e:
        parse_error = f"[{platformName}] Date Parsing Error {e}"
        socialmarket_logger.info(parse_error)
        timePrint(parse_error)
        return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TELEGRAM DELIVERY HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def log_telegram_warning(message):
    timePrint(message)
    socialmarket_logger.warning(message)


def log_telegram_info(message):
    timePrint(message)
    socialmarket_logger.info(message)


def telegram_message_label(text):
    compact = re.sub(r"\s+", " ", str(text)).strip()
    return compact[:80] if compact else "empty-message"


def reset_telegram_session():
    global telegram_session
    with telegram_session_lock:
        old_session = telegram_session
        telegram_session = requests.Session()
        with contextlib.suppress(Exception):
            old_session.close()


def request_socialmarket_force_restart(reason, detail=None):
    global telegram_force_restart_last_requested
    now_monotonic = time_module.monotonic()

    if now_monotonic - telegram_force_restart_last_requested < TELEGRAM_FORCE_RESTART_COOLDOWN_SECONDS:
        return False

    telegram_force_restart_last_requested = now_monotonic
    target_path = os.path.join(functionsFolder_path, "SOCIALMARKET_FORCE_RESTART.txt")
    restart_payload = {
        "timestamp": datetime.now(pytz.timezone(XANDER_TIMEZONE)).isoformat(),
        "reason": reason,
        "detail": detail,
    }

    try:
        os.makedirs(functionsFolder_path, exist_ok=True)
        if os.path.exists(target_path):
            log_telegram_warning(f"SocialMarket force restart already requested | reason={reason} | detail={detail}")
            return True

        with open(target_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(restart_payload, indent=2))

        log_telegram_warning(f"SocialMarket force restart requested | reason={reason} | detail={detail}")
        return True
    except Exception as exc:
        log_telegram_warning(f"Failed to request SocialMarket force restart | reason={reason} | error={exc}\n{traceback.format_exc()}")
        return False


def mark_telegram_send_success(label, attempts):
    with telegram_health_lock:
        had_failures = telegram_health["consecutive_failures"] > 0
        telegram_health["last_success_monotonic"] = time_module.monotonic()
        telegram_health["last_success_at"] = datetime.now(pytz.timezone(XANDER_TIMEZONE)).isoformat()
        telegram_health["consecutive_failures"] = 0
        telegram_health["last_exception"] = None

    if attempts > 1 or had_failures:
        log_telegram_info(f"Telegram send success | message={label} | attempts={attempts}")


def mark_telegram_send_failure(label, exc_summary):
    with telegram_health_lock:
        telegram_health["last_failure_at"] = datetime.now(pytz.timezone(XANDER_TIMEZONE)).isoformat()
        telegram_health["last_exception"] = exc_summary
        telegram_health["consecutive_failures"] += 1
        consecutive_failures = telegram_health["consecutive_failures"]

    if consecutive_failures >= TELEGRAM_MAX_CONSECUTIVE_FAILURES:
        request_socialmarket_force_restart(
            "TELEGRAM_CONSECUTIVE_FAILURES",
            f"message={label} | failures={consecutive_failures} | last_error={exc_summary}",
        )


def mark_telegram_in_flight(label):
    with telegram_health_lock:
        telegram_health["in_flight_started_at"] = time_module.monotonic()
        telegram_health["in_flight_label"] = label


def clear_telegram_in_flight():
    with telegram_health_lock:
        telegram_health["in_flight_started_at"] = None
        telegram_health["in_flight_label"] = None


def get_telegram_retry_delay(response_json, attempt_number):
    retry_after = None
    if isinstance(response_json, dict):
        retry_after = response_json.get("parameters", {}).get("retry_after")

    try:
        retry_after = int(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        retry_after = None

    if retry_after is not None:
        return min(max(retry_after, 1), 30)

    return min(2 * attempt_number, 10)


def telegram_worker(worker_id=0):

    while True:
        item = None
        try:
            item = telegram_queue.get(timeout=5)
            label = telegram_message_label(item.get("text", ""))
            queue_wait_time = time_module.monotonic() - item["enqueued_at"]
            queued = queue_wait_time >= 0.05
            queue_wait_time = int(queue_wait_time) if queued else 0.0

            mark_telegram_in_flight(label)
            sent = send_telegram_message(
                item["text"],
                item["lockHrs"],
                item["retryCount"],
                queue_wait_time
            )

            if not sent:
                log_telegram_warning(f"Telegram message dropped after retries | message={label}")
        except queue.Empty:
            continue
        except Exception as exc:
            log_telegram_warning(f"Telegram worker crash recovered | worker={worker_id} | error={exc}\n{traceback.format_exc()}")
            request_socialmarket_force_restart("TELEGRAM_WORKER_EXCEPTION", str(exc))
            time_module.sleep(5)
        finally:
            clear_telegram_in_flight()
            if item is not None:
                telegram_queue.task_done()


def start_telegram_workers():
    for worker_id in range(MAX_WORKERS):
        worker = threading.Thread(target=telegram_worker, args=(worker_id,), daemon=True)
        worker.start()
        telegram_worker_threads.append(worker)


def telegram_health_monitor():
    while True:
        try:
            time_module.sleep(TELEGRAM_HEALTH_CHECK_SECONDS)
            qsize = telegram_queue.qsize()
            alive_workers = [worker for worker in telegram_worker_threads if worker.is_alive()]

            if len(alive_workers) < MAX_WORKERS:
                log_telegram_warning(f"Telegram sender worker not alive | alive={len(alive_workers)} | expected={MAX_WORKERS}")
                request_socialmarket_force_restart("TELEGRAM_WORKER_DEAD", f"alive={len(alive_workers)} expected={MAX_WORKERS}")

            if qsize >= TELEGRAM_QUEUE_WARNING_SIZE:
                log_telegram_warning(f"Telegram queue size warning | queue_size={qsize} | max_size={QUEUE_SIZE}")

            with telegram_health_lock:
                in_flight_started_at = telegram_health.get("in_flight_started_at")
                in_flight_label = telegram_health.get("in_flight_label")
                consecutive_failures = telegram_health.get("consecutive_failures", 0)
                last_exception = telegram_health.get("last_exception")

            if in_flight_started_at is not None:
                stuck_seconds = time_module.monotonic() - in_flight_started_at
                if stuck_seconds >= TELEGRAM_WORKER_STUCK_SECONDS:
                    request_socialmarket_force_restart(
                        "TELEGRAM_WORKER_STUCK",
                        f"message={in_flight_label} | stuck_seconds={int(stuck_seconds)} | queue_size={qsize}",
                    )

            if consecutive_failures >= TELEGRAM_MAX_CONSECUTIVE_FAILURES:
                request_socialmarket_force_restart(
                    "TELEGRAM_CONSECUTIVE_FAILURES",
                    f"failures={consecutive_failures} | last_error={last_exception}",
                )
        except Exception as exc:
            log_telegram_warning(f"Telegram health monitor error | error={exc}\n{traceback.format_exc()}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def send_telegram_message_thread(text, lockHrs, retryCount, dedupe_override=None):

    dedupe_key = dedupe_override if dedupe_override else text.strip()
    cache_added = False

    if "⚠️" not in text:
        with sent_cache_lock:
            if dedupe_key in sent_cache:
                return
            sent_cache.add(dedupe_key)
            cache_added = True

    item = {
        "text": text,
        "lockHrs": lockHrs,
        "retryCount": retryCount,
        "enqueued_at": time_module.monotonic(),
    }

    try:
        telegram_queue.put_nowait(item)
        qsize = telegram_queue.qsize()
        if qsize >= TELEGRAM_QUEUE_WARNING_SIZE:
            log_telegram_warning(f"Telegram queue size warning | queue_size={qsize} | max_size={QUEUE_SIZE}")
    except queue.Full:
        if cache_added:
            with sent_cache_lock:
                sent_cache.discard(dedupe_key)
        label = telegram_message_label(text)
        log_telegram_warning(f"Telegram queue full - dropping message | message={label} | max_size={QUEUE_SIZE}")
        request_socialmarket_force_restart("TELEGRAM_QUEUE_FULL", f"message={label}")
        timePrint("⚠️ Telegram queue full — dropping message")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def send_telegram_message(text, lockHrs, retryCount, queue_wait_time):
    
    toProceed = True
    label = telegram_message_label(text)

    if lockHrs > 0:
        with lock:
            base_text = text.strip()
            tele_folder = os.path.join(validationFolder_path, "TELE")
            messages_file_path = os.path.join(tele_folder, "messages.txt")
            os.makedirs(tele_folder, exist_ok=True)

            if os.path.exists(messages_file_path):
                with open(messages_file_path, 'r', encoding='utf-8') as file:
                    try:
                        message_records = json.load(file)
                    except json.JSONDecodeError:
                        message_records = []
            else:
                message_records = []

            now = datetime.now()
            found = False

            for record in message_records:
                if record["text"] == base_text:
                    found = True
                    recorded_time = datetime.fromisoformat(record["datetime"])
                    if now - recorded_time < timedelta(hours=lockHrs):
                        toProceed = False
                    else:
                        record["datetime"] = now.isoformat()
                    break

            if not found:
                message_records.append({
                    "text": base_text,
                    "datetime": now.isoformat()
                })

            # Append notification 
            alert_msg = f"* Alert above will be paused for {lockHrs}HR(S) to prevent spam"
            alert_pattern = r"\* Alert above will be paused for \d+HR\(S\) to prevent spam"

            if not re.search(alert_pattern, text):
                text += f"\n{alert_msg}"

            with open(messages_file_path, 'w', encoding='utf-8') as file:
                json.dump(message_records, file, indent=2)

    if not toProceed:
        return True

    if retryCount > 0 or (retryCount == 0 and queue_wait_time > 0):
        delay_time = (5 * retryCount) + queue_wait_time

        mins, secs = divmod(int(delay_time), 60)
        if mins > 0:
            delay_display = f"{mins}m {secs}s"
        else:
            delay_display = f"{secs}s"

        delay_msg = f"* Delayed for ~{delay_display} due to rate limiting"

        delay_pattern = r"^\* Delayed for ~\d+m \d+s due to rate limiting$|^\* Delayed for ~\d+s due to rate limiting$"

        if re.search(delay_pattern, text, flags=re.MULTILINE):
            text = re.sub(delay_pattern, delay_msg, text, flags=re.MULTILINE)
        else:
            text += f"\n{delay_msg}"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True
    }

    total_attempts = max(1, TELEGRAM_MAX_SEND_ATTEMPTS)

    for attempt_index in range(total_attempts):
        attempt_number = attempt_index + 1
        response_json = None

        try:
            with telegram_session_lock:
                response = telegram_session.post(
                    url,
                    json=payload,
                    timeout=(TELEGRAM_CONNECT_TIMEOUT_SECONDS, TELEGRAM_SEND_TIMEOUT_SECONDS),
                )
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as exc:
            exc_summary = f"{type(exc).__name__}: {exc}"
            log_telegram_warning(f"Telegram send failure | message={label} | attempt={attempt_number}/{total_attempts} | error={exc_summary}\n{traceback.format_exc()}")
            mark_telegram_send_failure(label, exc_summary)
            if attempt_number >= 2:
                reset_telegram_session()
            if attempt_number < total_attempts:
                retry_delay = get_telegram_retry_delay(None, attempt_number)
                log_telegram_warning(f"Telegram retry scheduled | message={label} | attempt={attempt_number + 1}/{total_attempts} | delay={retry_delay}s")
                time_module.sleep(retry_delay)
            continue
        except ValueError as exc:
            exc_summary = f"{type(exc).__name__}: {exc}"
            log_telegram_warning(f"Telegram send response parse failure | message={label} | attempt={attempt_number}/{total_attempts} | error={exc_summary}\n{traceback.format_exc()}")
            mark_telegram_send_failure(label, exc_summary)
            if attempt_number < total_attempts:
                retry_delay = get_telegram_retry_delay(None, attempt_number)
                log_telegram_warning(f"Telegram retry scheduled | message={label} | attempt={attempt_number + 1}/{total_attempts} | delay={retry_delay}s")
                time_module.sleep(retry_delay)
            continue
        except Exception as exc:
            exc_summary = f"{type(exc).__name__}: {exc}"
            log_telegram_warning(f"Telegram send unexpected failure | message={label} | attempt={attempt_number}/{total_attempts} | error={exc_summary}\n{traceback.format_exc()}")
            mark_telegram_send_failure(label, exc_summary)
            reset_telegram_session()
            if attempt_number < total_attempts:
                retry_delay = get_telegram_retry_delay(None, attempt_number)
                log_telegram_warning(f"Telegram retry scheduled | message={label} | attempt={attempt_number + 1}/{total_attempts} | delay={retry_delay}s")
                time_module.sleep(retry_delay)
            continue

        if response_json.get("ok"):
            mark_telegram_send_success(label, attempt_number)
            return True

        description = response_json.get("description")
        exc_summary = f"Telegram API rejected message: {description}"
        log_telegram_warning(f"Telegram API send failure | message={label} | attempt={attempt_number}/{total_attempts} | response={response_json}")
        mark_telegram_send_failure(label, exc_summary)

        if attempt_number < total_attempts:
            retry_delay = get_telegram_retry_delay(response_json, attempt_number)
            log_telegram_warning(f"Telegram retry scheduled | message={label} | attempt={attempt_number + 1}/{total_attempts} | delay={retry_delay}s")
            time_module.sleep(retry_delay)

    request_socialmarket_force_restart("TELEGRAM_SEND_FAILED", f"message={label}")
    return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TELEGRAM MESSAGE FORMATTERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def transformJsonToTeleMessage(platform_name, post_id, json_result, ai_model, gpt_cat, duration):

    if (gpt_cat in stage2_gpt4o_custom_prompts):
        gpt_cat = "DYNAMIC"
        
    category = json_result.get("Category", "N.A.")
    region = json_result.get("Region", "N.A.")
    types = json_result.get("Type", [])
    if not isinstance(types, list):
        types = []
    details = json_result.get("Details", "")
    fgi = json_result.get("FGI_Value", "N.A.")

    # Format the types
    type_lines = []
    for t in types:
        if not isinstance(t, dict):
            continue
        asset = t.get("Asset", "N.A.")
        sentiment = t.get("Sentiment", "N.A.")

        if sentiment == "Bullish":
            type_lines.append(f"{asset} 🟩")
        elif sentiment == "Bearish":
            type_lines.append(f"{asset} 🟥")
        else:
            type_lines.append(f"{asset} 🟧")
    
    # Join types as comma-separated string
    type_str = ", ".join(type_lines)

    if category == "CAT_1":

        fgi_val = None

        if fgi is not None and fgi != "" and fgi != "N.A.":
            try:
                fgi = int(float(fgi))

                if fgi < 45:
                    fgi_val = f"🔴 ({fgi}) Risk Off"  # Fear
                elif fgi > 55:
                    fgi_val = f"🟢 ({fgi}) Risk On"  # Greed
                else:
                    fgi_val = f"🔘 ({fgi}) Neutral"  # Neutral
            except (TypeError, ValueError):
                fgi_val = None

        msg_title = f"[{platform_name}]" + (f" - {post_id}" if post_id else "") + "\n"

        lines = [
            f"✅ {category} TRIGGER",
            f"🌍 {region}",
            f"🔹 {type_str}",
        ]

        if fgi_val:
            lines.append(fgi_val)

        if details:
            lines.append(f"📝 {details}")
        
        lines.append(f"* {ai_model} | {gpt_cat} | ({duration})")

        message = msg_title + "\n".join(lines)
    else:

        msg_title = f"[{platform_name}]" + (f" - {post_id}" if post_id else "") + "\n"

        lines = [
            f"📊 {category}",
            f"* {ai_model} | {gpt_cat} | ({duration})"
        ]

        message = msg_title + "\n".join(lines)

    return message
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ORDER FILE CREATION HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def process_ibkr_order_legacy(targetted_assets, region, post_content, company_mentions, resolved_ticker_details):

    errorOccurred = False
    position = 0

    for item in targetted_assets:

        try:
            asset = item.get("Asset", "").upper()
            sentiment = item.get("Sentiment", "").upper()
            toProceed = True

            if region and asset and sentiment and (sentiment == "BULLISH" or sentiment == "BEARISH"):
                
                sectors = [
                    "general",
                    "technology",
                    "semiconductors",
                    "financials",
                    "healthcare",
                    "consumer",
                    "defense",
                    "industrial",
                    "real estate"
                ]
                
                sectors_ignore = [
                    "energy",
                    "evs",
                    "others"
                ]

                if (asset.strip().lower() in sectors_ignore) or (asset.strip().lower() not in sectors + sectors_ignore and "$" not in asset):
                    send_telegram_message_thread(f'⚠️ Skipping: Execution not permitted for asset "{asset}"', 0, 0)
                    toProceed = False
                elif asset.strip().lower() in sectors and region.strip().upper() not in ["GENERAL", "USA"]:
                    send_telegram_message_thread(f"⚠️ Skipping: Execution not permitted for {region} region.", 0, 0)
                    toProceed = False

                # Verify if SPY is up >= 1% for the day
                if asset.strip().lower() == "general":
                    spy = yf.Ticker("SPY")
                    data = spy.history(period="1d", interval="1m")  # today's intraday data

                    # Get latest price and today's open
                    current_price = data['Close'][-1]
                    open_price = data['Open'].iloc[0]

                    pct_change = (current_price - open_price) / open_price * 100

                    if pct_change >= 1:
                        send_telegram_message_thread(f"⚠️ Skipping: SPY is already up ≥ 1% today — further upside may be limited.", 0, 0)
                        toProceed = False

                if toProceed:

                    if (asset.strip().lower() in sectors and region in ["GENERAL", "USA"]) or (asset.strip().lower() not in sectors + sectors_ignore and "$" in asset):

                        final_post_content = post_content

                        sector_to_ticker = {
                            "SPY": "SPXL",
                            "GENERAL": "SPXL",
                            "TECHNOLOGY": "TQQQ",
                            "SEMICONDUCTORS": "SOXL",
                            "FINANCIALS": "FAS",
                            "HEALTHCARE": "CURE",
                            "CONSUMER": "WANT",
                            "DEFENSE": "DFEN",
                            "INDUSTRIAL": "DUSL",
                            "REAL ESTATE": "DRN"
                        }
                    
                        action = "BUY" if sentiment == "BULLISH" else "SELL"
                        ticker = sector_to_ticker.get(asset.upper(), asset.replace("$", ""))

                        toProceed = True
                        isEarningsRequest = "**EARNINGS REQUEST**" in post_content

                        if ("$" in asset):
                            marketCap = get_market_cap_yahoo(ticker, None)
                            if marketCap is not None:
                                fill_line = f"\n**MARKETCAP AT {marketCap}**"
                                final_post_content = post_content + fill_line
                                if "**EARNINGS REQUEST**" in post_content:
                                    if (marketCap < 500_000_00_000_000_000):
                                        toProceed = False
                                elif marketCap < 50_000_000:
                                    toProceed = False

                        elif (len(company_mentions) > 0):
                            ticker_details = resolved_ticker_details.get(ticker)
                            marketCap = ticker_details.get("marketCap") if ticker_details else None
                            # Add market cap amount into file content
                            if marketCap is None:
                                marketCap = get_market_cap_yahoo(ticker, company_mentions[position], validate_name=True)
                            if marketCap is not None:
                                fill_line = f"\n**MARKETCAP AT {marketCap}**"
                                final_post_content = post_content + fill_line
                                if "**EARNINGS REQUEST**" in post_content:
                                    if (marketCap < 500_000_00_000_000_000):
                                        toProceed = False
                                elif marketCap < 50_000_000:
                                    toProceed = False

                        if toProceed:

                            unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)).upper()

                            filename = f"XANDER_{action}_{ticker}_{unique_id}_0.txt"

                            full_path = os.path.join(ibkrFolder_path, filename)

                            with open(full_path, 'w', encoding='utf-8') as f:
                                f.write(final_post_content)

                            order_created_msg = f"{filename} sent to {ibkrFolder_path}"
                            socialmarket_logger.info(order_created_msg)
                            timePrint(order_created_msg)

                        elif action == "BUY" and isEarningsRequest:
                            send_telegram_message_thread(f'⚠️ Skipping: Insufficient market cap requirement to place EARNINGS order for "{asset}"', 0, 0)

                        elif action == "BUY":
                            send_telegram_message_thread(f'⚠️ Skipping: Insufficient market cap requirement to place order for "{asset}"', 0, 0)

            position += 1
            
        except Exception as e:
            errorOccurred = True
            error_msg = f"Failed to write order file: {e}"
            socialmarket_logger.info(error_msg)
            timePrint(error_msg)

    if errorOccurred:
        error_msg = f"⚠️ An error occurred while creating the order. Review the logs for further information."
        send_telegram_message_thread(error_msg, 0, 0)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def process_ibkr_order(targetted_assets, region, post_content, company_mentions, resolved_ticker_details):

    expected_orders = []
    created_orders = []
    skipped_orders = []
    failed_orders = []

    sectors = [
        "general",
        "technology",
        "semiconductors",
        "financials",
        "healthcare",
        "consumer",
        "defense",
        "industrial",
        "real estate",
    ]

    sectors_ignore = [
        "energy",
        "evs",
        "others",
    ]

    sector_to_ticker = {
        "SPY": "SPXL",
        "GENERAL": "SPXL",
        "TECHNOLOGY": "TQQQ",
        "SEMICONDUCTORS": "SOXL",
        "FINANCIALS": "FAS",
        "HEALTHCARE": "CURE",
        "CONSUMER": "WANT",
        "DEFENSE": "DFEN",
        "INDUSTRIAL": "DUSL",
        "REAL ESTATE": "DRN",
    }

    if not isinstance(targetted_assets, list):
        error_msg = f"IBKR order creation skipped | reason=INVALID_TARGET_ASSETS | detail={type(targetted_assets).__name__}"
        socialmarket_logger.error(error_msg)
        timePrint(error_msg)
        send_telegram_message_thread("WARNING: Unable to create order files because CAT_1 assets were malformed. Review SocialMarket logs.", 0, 0)
        return

    if not isinstance(company_mentions, list):
        company_mentions = []

    if not isinstance(resolved_ticker_details, dict):
        resolved_ticker_details = {}

    def format_market_cap(market_cap):
        try:
            return f"{int(market_cap):,}"
        except (TypeError, ValueError):
            return str(market_cap)

    def record_skip(asset, ticker, action, reason_code, detail, notify_text=None):
        record = {
            "asset": asset,
            "ticker": ticker,
            "action": action,
            "reason": reason_code,
            "detail": detail,
        }
        skipped_orders.append(record)
        log_msg = (
            f"IBKR order skipped | asset={asset or 'N.A.'} | ticker={ticker or 'N.A.'} | "
            f"action={action or 'N.A.'} | reason={reason_code} | detail={detail}"
        )
        socialmarket_logger.warning(log_msg)
        timePrint(log_msg)
        if notify_text:
            send_telegram_message_thread(notify_text, 0, 0)

    def record_failure(asset, ticker, action, exc):
        exc_summary = f"{type(exc).__name__}: {exc}"
        record = {
            "asset": asset,
            "ticker": ticker,
            "action": action,
            "error": exc_summary,
        }
        failed_orders.append(record)
        error_msg = (
            f"IBKR order failed | asset={asset or 'N.A.'} | ticker={ticker or 'N.A.'} | "
            f"action={action or 'N.A.'} | error={exc_summary}\n{traceback.format_exc()}"
        )
        socialmarket_logger.error(error_msg)
        timePrint(error_msg)
        send_telegram_message_thread(
            f"WARNING: Failed to create order file for {('$' + ticker) if ticker else asset}. Review SocialMarket logs.",
            0,
            0,
        )

    def build_unique_order_path(action, ticker):
        os.makedirs(ibkrFolder_path, exist_ok=True)
        for _ in range(10):
            unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)).upper()
            filename = f"XANDER_{action}_{ticker}_{unique_id}_0.txt"
            full_path = os.path.join(ibkrFolder_path, filename)
            if not os.path.exists(full_path):
                return filename, full_path
        raise RuntimeError(f"Unable to generate unique order filename for {action} {ticker}")

    def atomic_write_order_file(full_path, content):
        tmp_path = f"{full_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, full_path)
        finally:
            if os.path.exists(tmp_path):
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

    def maybe_apply_market_cap_gate(asset, ticker, action, final_post_content, position):
        marketCap = None
        isEarningsRequest = "**EARNINGS REQUEST**" in post_content
        min_market_cap = 500_000_00_000_000_000 if isEarningsRequest else 50_000_000

        if "$" in asset:
            marketCap = get_market_cap_yahoo(ticker, None)
        elif len(company_mentions) > 0:
            ticker_details = resolved_ticker_details.get(ticker)
            marketCap = ticker_details.get("marketCap") if ticker_details else None
            company_keyword = company_mentions[position] if position < len(company_mentions) else None
            if marketCap is None:
                marketCap = get_market_cap_yahoo(ticker, company_keyword, validate_name=True)

        if marketCap is None:
            socialmarket_logger.info(
                f"IBKR order market cap unavailable; proceeding with existing behavior | ticker={ticker} | asset={asset}"
            )
            return True, final_post_content

        final_post_content = post_content + f"\n**MARKETCAP AT {marketCap}**"

        try:
            numeric_market_cap = int(marketCap)
        except (TypeError, ValueError):
            socialmarket_logger.warning(
                f"IBKR order market cap value invalid; proceeding with existing behavior | ticker={ticker} | marketCap={marketCap}"
            )
            return True, final_post_content

        if numeric_market_cap < min_market_cap:
            request_type = "EARNINGS " if isEarningsRequest else ""
            detail = (
                f"marketCap={format_market_cap(numeric_market_cap)} below "
                f"minimum={format_market_cap(min_market_cap)}"
            )
            record_skip(
                asset,
                ticker,
                action,
                "MARKET_CAP_BELOW_MINIMUM",
                detail,
                f"WARNING: Skipping {request_type}order for ${ticker}: {detail}.",
            )
            return False, final_post_content

        return True, final_post_content

    def log_reconciliation():
        expected_tickers = [item["ticker"] for item in expected_orders if item.get("ticker")]
        created_tickers = [item["ticker"] for item in created_orders if item.get("ticker")]
        skipped_tickers = [item["ticker"] for item in skipped_orders if item.get("ticker")]
        failed_tickers = [item["ticker"] for item in failed_orders if item.get("ticker")]
        accounted = set(created_tickers + skipped_tickers + failed_tickers)
        missing = [ticker for ticker in expected_tickers if ticker not in accounted]

        summary_msg = (
            f"IBKR order reconciliation | expected={expected_tickers} | "
            f"created={created_orders} | skipped={skipped_orders} | failed={failed_orders}"
        )
        socialmarket_logger.info(summary_msg)

        if missing:
            error_msg = f"IBKR order reconciliation missing outcome | missing={missing} | expected={expected_orders}"
            socialmarket_logger.error(error_msg)
            timePrint(error_msg)

    for position, item in enumerate(targetted_assets):

        asset = None
        ticker = None
        action = None

        try:
            if not isinstance(item, dict):
                record_skip(None, None, None, "INVALID_TYPE_ITEM", f"item={item!r}")
                continue

            asset = str(item.get("Asset", "")).strip().upper()
            sentiment = str(item.get("Sentiment", "")).strip().upper()
            region_upper = str(region or "").strip().upper()
            action = "BUY" if sentiment == "BULLISH" else "SELL" if sentiment == "BEARISH" else None

            if not region_upper:
                record_skip(asset, None, action, "MISSING_REGION", f"region={region!r}")
                continue

            if not asset:
                record_skip(asset, None, action, "MISSING_ASSET", f"item={item!r}")
                continue

            if sentiment not in ("BULLISH", "BEARISH"):
                record_skip(asset, None, action, "NON_DIRECTIONAL_SENTIMENT", f"sentiment={sentiment or 'N.A.'}")
                continue

            asset_key = asset.strip().lower()
            is_sector = asset_key in sectors
            is_ignored_sector = asset_key in sectors_ignore
            is_direct_ticker = "$" in asset

            if is_ignored_sector or (asset_key not in sectors + sectors_ignore and not is_direct_ticker):
                record_skip(
                    asset,
                    None,
                    action,
                    "EXECUTION_NOT_PERMITTED_ASSET",
                    f"asset={asset}",
                    f'WARNING: Skipping order: execution not permitted for asset "{asset}".',
                )
                continue

            if is_sector and region_upper not in ["GENERAL", "USA"]:
                record_skip(
                    asset,
                    None,
                    action,
                    "EXECUTION_NOT_PERMITTED_REGION",
                    f"region={region}",
                    f"WARNING: Skipping order: execution not permitted for {region} region.",
                )
                continue

            if asset_key == "general":
                try:
                    spy = yf.Ticker("SPY")
                    data = spy.history(period="1d", interval="1m")
                    current_price = data['Close'][-1]
                    open_price = data['Open'].iloc[0]
                    pct_change = (current_price - open_price) / open_price * 100

                    if pct_change >= 1:
                        record_skip(
                            asset,
                            "SPXL",
                            action,
                            "SPY_UPSIDE_LIMIT",
                            f"SPY intraday change={pct_change:.2f}%",
                            "WARNING: Skipping order: SPY is already up >= 1% today; further upside may be limited.",
                        )
                        continue
                except Exception as exc:
                    socialmarket_logger.warning(
                        f"IBKR order SPY guard check failed; continuing with existing behavior | error={exc}\n{traceback.format_exc()}"
                    )

            if not ((is_sector and region_upper in ["GENERAL", "USA"]) or (asset_key not in sectors + sectors_ignore and is_direct_ticker)):
                record_skip(asset, None, action, "NOT_ORDER_CANDIDATE", f"asset={asset} region={region}")
                continue

            ticker = sector_to_ticker.get(asset.upper(), asset.replace("$", "").strip().upper())
            if not ticker or ticker in ("N.A.", "NA"):
                record_skip(asset, ticker, action, "INVALID_TICKER", f"asset={asset}")
                continue

            expected_orders.append({"asset": asset, "ticker": ticker, "action": action})

            final_post_content = post_content
            may_create, final_post_content = maybe_apply_market_cap_gate(asset, ticker, action, final_post_content, position)
            if not may_create:
                continue

            filename, full_path = build_unique_order_path(action, ticker)
            socialmarket_logger.info(
                f"IBKR order file create attempt | asset={asset} | ticker={ticker} | action={action} | path={full_path}"
            )
            atomic_write_order_file(full_path, final_post_content)

            created_record = {
                "asset": asset,
                "ticker": ticker,
                "action": action,
                "file": filename,
                "path": full_path,
            }
            created_orders.append(created_record)

            order_created_msg = f"{filename} sent to {ibkrFolder_path}"
            socialmarket_logger.info(order_created_msg)
            timePrint(order_created_msg)

        except Exception as e:
            try:
                record_failure(asset, ticker, action, e)
            except Exception:
                error_msg = f"IBKR order failure logging failed | error={e}\n{traceback.format_exc()}"
                socialmarket_logger.error(error_msg)
                timePrint(error_msg)

    log_reconciliation()

    if failed_orders:
        error_msg = "WARNING: An error occurred while creating one or more order files. Review SocialMarket logs."
        send_telegram_message_thread(error_msg, 0, 0)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# POST STORAGE / SCRAPING HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def getSavedPostTime(platform, user):
    validation_folder = os.path.join(validationFolder_path, platform, user)
    file_path = os.path.join(validation_folder, "previous_post.txt")

    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # JSON only
            return data.get("time")

    except Exception as e:
        timePrint(f"Corrupted saved post detected. Resetting file. Error: {e}")

        try:
            os.remove(file_path)
        except OSError:
            pass

        return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def save_previous_post(file_path, post_obj):
    atomic_write_json(file_path, post_obj)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def load_stocktitan_seen_ids(validation_folder):
    file_path = os.path.join(validation_folder, "seen_urls.json")
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def save_stocktitan_seen_ids(validation_folder, seen_ids, max_items=1000):
    file_path = os.path.join(validation_folder, "seen_urls.json")
    unique_seen_ids = list(dict.fromkeys(seen_ids))[-max_items:]
    atomic_write_json(file_path, unique_seen_ids)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def validate_post(post_obj, platform, user):
    validation_folder = os.path.join(validationFolder_path, platform, user)
    file_path = os.path.join(validation_folder, "previous_post.txt")

    os.makedirs(validation_folder, exist_ok=True)

    if platform == "StockTitan":
        dedupe_id = post_obj.get("dedupe_id") or post_obj.get("url")
        if dedupe_id:
            with validation_lock:
                seen_ids = load_stocktitan_seen_ids(validation_folder)
                if dedupe_id in set(seen_ids):
                    return False

                first_scan = not os.path.exists(file_path)
                seen_ids.append(dedupe_id)
                save_stocktitan_seen_ids(validation_folder, seen_ids)
                save_previous_post(file_path, post_obj)
                return None if first_scan else True

    if not os.path.exists(file_path):
        save_previous_post(file_path, post_obj)
        return None  # first scan

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            saved_post = json.load(f)
    except Exception:
        save_previous_post(file_path, post_obj)
        return True

    if saved_post.get("url") == post_obj.get("url"):
        return False

    time_format = "%d-%b-%y %I:%M%p"
    current_dt = datetime.strptime(post_obj["time"], time_format)
    saved_dt = datetime.strptime(saved_post["time"], time_format)

    is_new = current_dt > saved_dt or (
        current_dt == saved_dt and post_obj.get("idx", 0) > saved_post.get("idx", 0)
    )

    if is_new:
        save_previous_post(file_path, post_obj)
        return True

    return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def purgeUrlHistory():
    validation_folder = os.path.join(validationFolder_path, "Benzinga", "URL")
    file_path = os.path.join(validation_folder, "URL_Archives.txt")

    # Nothing to purge if file does not exist
    if not os.path.exists(file_path):
        return

    # Read all lines
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file]

    # If 30 or fewer lines, nothing to purge
    if len(lines) <= 30:
        return

    # Keep only the last 30 lines
    latest_30 = lines[-30:]

    # Rewrite file with the latest 30 entries
    with open(file_path, "w", encoding="utf-8") as file:
        for line in latest_30:
            file.write(line + "\n")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def verifyUrlHistory(postLink):
    postLink = postLink.strip()  # normalize input
    
    validation_folder = os.path.join(validationFolder_path, "Benzinga", "URL")
    file_path = os.path.join(validation_folder, "URL_Archives.txt")

    os.makedirs(validation_folder, exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(postLink + "\n")
        return True

    with open(file_path, "r", encoding="utf-8") as file:
        for row in file:
            if postLink == row.strip():
                return False

    with open(file_path, "a", encoding="utf-8") as file:
        file.write(postLink + "\n")

    return True
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def getTrumpNewPosts_TS(displayName, process, lastSaved):
    # Relies on 'status', 'status__content', 'status-info__account-name'
    instances = process.get("instance", [])
    if len(instances) == 1:
        target_url = instances[0]
    else:
        target_url = random.choice(instances)
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://google.com",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    try:

        with requests.Session() as session:
            response = session.get(target_url, headers=headers)

        if (response.status_code != 200):
            process['scrapeFailCount'] += 1
            error_msg = f"[TS] ({process['scrapeFailCount']}) [{process['user']}] Scraping error status code received: {response.status_code}"
            socialmarket_logger.info(error_msg)
            timePrint(error_msg)
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        status_blocks = soup.find_all('div', class_='status')

        post_obj_list = []
        isLaterThanSaved = False
        idx = 0

        for status in status_blocks:

            # Extract content within this status block
            post_html = status.find('div', class_='status__content')
            if post_html:
                lines = []

                for p in post_html.find_all('p'):

                    for a_tag in p.find_all('a'):
                        a_tag.decompose()

                    text = p.get_text(separator="\n", strip=True).replace('\xa0', ' ')
                    if text:
                        lines.append(text)

                if len(lines) > 1:
                    post = lines[0] + "\n\n" + "\n".join(lines[1:])
                else:
                    post = "\n".join(lines)
            else:
                post = None

            # Extract the date & time of Post (Convert to SGT --> 12HRS Ahead)
            post_items = status.find_all('a', class_='status-info__meta-item')
            if len(post_items) >= 2:
                post_datetime_raw = post_items[1].get_text(strip=True)
                post_datetime = parse_and_adjust_datetime(post_datetime_raw, "%B %d, %Y, %I:%M %p", 12, "TS")
            else:
                post_datetime = None

            # Verify if extracted date & time of post is later than previously saved in local
            isLaterThanSaved = post_datetime is not None and lastSaved is None

            if post_datetime and lastSaved:
                time_format = "%d-%b-%y %I:%M%p"
                post_dt = datetime.strptime(post_datetime, time_format)
                last_saved_dt = datetime.strptime(lastSaved, time_format)

                if post_dt >= last_saved_dt:
                    isLaterThanSaved = True

            # Extract the URL within this status block
            post_link_html = status.find('a', class_='status__external-link')
            post_link = None
            if (post_link_html):
                href = post_link_html.get('href')
                if (href):
                    post_link = href # Gives Truth Social actual URL

            # Extract owner within this status block
            owner_html = status.find('a', class_='status-info__account-name')
            is_owner = owner_html and owner_html.get_text(strip=True) == displayName

            # Get all links in this post block (to detect attachment URLs), IMG & VID not included into TrumpTruth
            links = [a['href'] for a in post_html.find_all('a', href=True)] if post_html else []
            has_attachments = len(links) > 0

            valid = bool(post and post.strip()) and (isLaterThanSaved or lastSaved is None)

            if (valid):
                process['scrapeFailCount'] = 0
                post_obj_list.append({
                    "idx": idx,
                    "content": post,
                    "url": post_link,
                    "time": post_datetime,
                    "isOwner": is_owner,
                    "hasAttachments": has_attachments,
                    "valid": valid
                })
                idx += 1
            
            if not (isLaterThanSaved) or (valid and lastSaved is None):
                return post_obj_list
            
        if (isLaterThanSaved):
            return post_obj_list
        else:
            # If no valid Post found
            process['scrapeFailCount'] += 1
            no_valid_msg = f"[TS] ({process['scrapeFailCount']}) [{process['user']}] Scraping for posts failed"
            socialmarket_logger.info(no_valid_msg)
            timePrint(no_valid_msg)
            return None
        
    except Exception as e:
        process['scrapeFailCount'] += 1
        error_msg = f"[TS] ({process['scrapeFailCount']}) [{process['user']}] Scraping exception occured:\n{e}"
        socialmarket_logger.info(error_msg)
        timePrint(error_msg)
        return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#   

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def getNewPosts_X(username, displayName, process, lastSaved):
    # Relies on 'timeline-item', 'tweet-content', 'tweet-date', 'fullname'
    instances = process.get("instance", [])
    if len(instances) == 1:
        target_url = instances[0] + username
    else:
        target_url = random.choice(instances) + username
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://google.com",
        "DNT": "1",  
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    try:

        with requests.Session() as session:
            response = session.get(target_url, headers=headers)

        if (response.status_code != 200):
            process['scrapeFailCount'] += 1
            error_msg = f"[X] ({process['scrapeFailCount']}) [{process['user']}] Scraping error status code received: {response.status_code}"
            socialmarket_logger.info(error_msg)
            timePrint(error_msg)
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        tweet_blocks = soup.find_all('div', class_='timeline-item')

        post_obj_list = []
        isLaterThanSaved = False
        idx = 0

        for tweet in tweet_blocks:

            # Skip pinned tweets
            if tweet.find('div', class_='pinned'):
                continue
            
            # Extract content within this tweet block
            post_html = tweet.find('div', class_='tweet-content')
            if post_html:
                # Replace <br> with newlines
                for br in post_html.find_all('br'):
                    br.replace_with('\n')

                # Get text with spacing preserved between tags
                post = post_html.get_text(separator=' ', strip=True).replace('\xa0', ' ')
            else:
                post = None

            # Extract the date & time of Post (Convert UTC to SGT --> 8HRS Ahead)
            tweet_date_block = tweet.find('span', class_='tweet-date')
            if tweet_date_block:
                date_anchor = tweet_date_block.find('a', title=True)
                if date_anchor:
                    datetime_raw = date_anchor['title']  # e.g., "Apr 16, 2025 · 1:44 PM UTC"
                    post_datetime = parse_and_adjust_datetime(datetime_raw, "%b %d, %Y · %I:%M %p UTC", 8, "X")
                else:
                    post_datetime = None
            else:
                post_datetime = None

            # Verify if extracted date & time of post is later than previously saved in local
            isLaterThanSaved = post_datetime is not None and lastSaved is None

            if post_datetime and lastSaved:
                time_format = "%d-%b-%y %I:%M%p"
                post_dt = datetime.strptime(post_datetime, time_format)
                last_saved_dt = datetime.strptime(lastSaved, time_format)

                if post_dt >= last_saved_dt:
                    isLaterThanSaved = True
                    
            # Extract the URL within this status block
            post_link_html = tweet.find('a', class_='tweet-link')
            post_link = None
            if (post_link_html):
                href = post_link_html.get('href')
                if (href):
                    post_link = "https://x.com" + href  # Gives relative URL, need to add X URL

            # Extract owner within this status block
            owner_html = tweet.find('a', class_='fullname')
            is_owner = owner_html and owner_html.get_text(strip=True) == displayName

            # Get all links in this tweet block (to detect attachment URLs, Images & Videos as well for X)
            links = [a['href'] for a in post_html.find_all('a', href=True)] if post_html else []
            has_attachments = (
                len(links) > 0 or
                tweet.find('a', class_='still-image') is not None or
                tweet.find('div', class_='video-container') is not None or
                tweet.find('video') is not None
            )

            valid = bool(post and post.strip()) and (isLaterThanSaved or lastSaved is None)

            if (valid):
                process['scrapeFailCount'] = 0
                post_obj_list.append({
                    "idx": idx,
                    "content": post,
                    "url": post_link,
                    "time": post_datetime,
                    "isOwner": is_owner, 
                    "hasAttachments": has_attachments,
                    "valid": valid
                })
                idx += 1

            if not (isLaterThanSaved) or (valid and lastSaved is None):
                return post_obj_list


        if (isLaterThanSaved):
            return post_obj_list
        else:
            # If no valid Post found
            process['scrapeFailCount'] += 1
            no_valid_msg = f"[X] ({process['scrapeFailCount']}) [{process['user']}] Scraping for posts failed"
            socialmarket_logger.info(no_valid_msg)
            socialmarket_logger.info(soup)
            timePrint(no_valid_msg)
            return None

    except Exception as e:
        process['scrapeFailCount'] += 1
        error_msg = f"[X] ({process['scrapeFailCount']}) [{process['user']}] Scraping exception occured:\n{e}"
        socialmarket_logger.info(error_msg)
        timePrint(error_msg)
        return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#   

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def getNewPosts_Benzinga(process, lastSaved):
    instances = process.get("instance", [])
    target_url = instances[0] + ""

    with open(os.path.join(functionsFolder_path, "XANDER_Benzinga_API.txt"), 'r', encoding='utf-8') as file:
        benzinga_token = file.read()

    querystring = {"token":benzinga_token,"pageSize":"15","displayOutput":"headline","sort":"created:desc"}
    headers = {"accept": "application/json"}

    try:

        response = requests.get(target_url, headers=headers, params=querystring, timeout=15)

        if (response.status_code != 200):
            process['scrapeFailCount'] += 1
            error_msg = f"[Benzinga] ({process['scrapeFailCount']}) [{process['user']}] Benzinga API error status code received: {response.status_code}"
            socialmarket_logger.info(error_msg)
            timePrint(error_msg)
            return None
        
        post_obj_list = []
        isLaterThanSaved = False
        idx = 0
        try:
            data = response.json()
        except ValueError as e:
            process['scrapeFailCount'] += 1
            error_msg = f"[Benzinga] ({process['scrapeFailCount']}) [{process['user']}] Invalid JSON response received:\n{e}"
            socialmarket_logger.info(error_msg)
            timePrint(error_msg)
            return None

        if not isinstance(data, list):
            process['scrapeFailCount'] += 1
            error_msg = f"[Benzinga] ({process['scrapeFailCount']}) [{process['user']}] Unexpected JSON response type received"
            socialmarket_logger.info(error_msg)
            timePrint(error_msg)
            return None

        for item in data:
            if not isinstance(item, dict):
                continue

            # Extract content within the item
            post_block = item.get('title')

            # Extract the date & time of Post (Convert UTC to SGT --> 8HRS Ahead)
            date_block = item.get('updated') # e.g., "Tue, 26 Aug 2025 08:10:09 -0400"
            
            if date_block:
                dt = parsedate_to_datetime(date_block)
                datetime_raw = dt.strftime("%d-%b-%y %I:%M%p")
                sgt_offset = 8 - (dt.utcoffset().total_seconds() / 3600)
                post_datetime = parse_and_adjust_datetime(datetime_raw, "%d-%b-%y %I:%M%p", sgt_offset, "Benzinga")
            else:
                post_datetime = None

            # Verify if extracted date & time of post is later than previously saved in local
            isLaterThanSaved = post_datetime is not None and lastSaved is None

            if post_datetime and lastSaved:
                time_format = "%d-%b-%y %I:%M%p"
                post_dt = datetime.strptime(post_datetime, time_format)
                last_saved_dt = datetime.strptime(lastSaved, time_format)
                now = datetime.now()
                if post_dt > now:
                    post_dt = now
                    post_datetime = post_dt.strftime("%d-%b-%y %#I:%M%p")

                if post_dt >= last_saved_dt:
                    isLaterThanSaved = True
                    
            # Extract the URL within this block
            post_link = item.get('url', '')

            # Extract the stocks provided by Benzinga
            stocks_list = item.get('stocks') or []
            cleaned_stocks_list = []
            for stock in stocks_list:
                if not isinstance(stock, dict):
                    continue
                stock_name = stock.get('name', '')
                if ":" not in stock_name and "/" not in stock_name:
                    stock['name'] = stock_name.replace("$", "")
                    cleaned_stocks_list.append(stock)

            # Extract news category from URL
            valid_categories = ["markets", "economics", "news", "m-a", "general"]
            category = ""
            if "benzinga.com/" in post_link:
                category = post_link.split("benzinga.com/", 1)[1].split("/")[0]

            valid = (
                post_block 
                and post_block.strip()
                and not post_block.lower().startswith(("reported earlier", "earlier today", "trading halt"))
                and (isLaterThanSaved or lastSaved is None)
                and (category.isdigit() or category in valid_categories)
            )
            
            if (valid):
                valid = (verifyUrlHistory(post_link))

            if (valid):
                
                # Block noise from Channels
                channels_block = item.get('channels') or []
                blocked_channels = ["Analyst Ratings", "Trading Ideas", "Long Ideas", "Short Ideas", "Price Target", "WIIM", "Movers"]
                valid_earnings_channels = ["Guidance", "Earnings Beats", "Earnings Misses"]

                if channels_block:

                    isEarnings = (any(isinstance(channel, dict) and channel.get('name') == "Earnings" for channel in channels_block))
                    hasValidEarningsChannel = False

                    for channel in channels_block:

                        if not isinstance(channel, dict):
                            continue
                        channel_name = channel.get('name')

                        if channel_name in blocked_channels:
                            valid = False
                            break
                            
                        elif isEarnings and channel_name in valid_earnings_channels:
                            hasValidEarningsChannel = True

                    if isEarnings and not hasValidEarningsChannel:
                        valid = False

                if (valid):

                    # Block noise from Tags
                    tags_block = item.get('tags') or []
                    blocked_tags = ["why it's moving"]

                    if tags_block:

                        for tag in tags_block:

                            if not isinstance(tag, dict):
                                continue
                            tag_name = tag.get('name')

                            if tag_name in blocked_tags:
                                valid = False
                                break
                
                if (valid):

                    process['scrapeFailCount'] = 0
                    post_obj_list.append({
                        "idx": idx,
                        "content": post_block,
                        "url": post_link,
                        "time": post_datetime,
                        "isOwner": True, 
                        "hasAttachments": False,
                        "valid": valid,
                        "tickers": cleaned_stocks_list
                    })

                    idx += 1

            if not (isLaterThanSaved) or (valid and lastSaved is None):
                return post_obj_list

        if (isLaterThanSaved):
            return post_obj_list
        else:
            # If no valid Post found
            process['scrapeFailCount'] += 1
            no_valid_msg = f"[Benzinga] ({process['scrapeFailCount']}) [{process['user']}] API Fetching for news failed"
            socialmarket_logger.info(no_valid_msg)
            timePrint(no_valid_msg)
            return None

    except Exception as e:
        process['scrapeFailCount'] += 1
        error_msg = f"[Benzinga] ({process['scrapeFailCount']}) [{process['user']}] API Fetching exception occured:\n{e}"
        socialmarket_logger.info(error_msg)
        timePrint(error_msg)
        return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#   

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# 
def getNewPosts_StockTitanRSS(process, lastSaved):
    instances = process.get("instance", [])
    target_url = instances[0] if instances else None

    if not target_url:
        process["scrapeFailCount"] += 1
        msg = f"[StockTitan] ({process['scrapeFailCount']}) Missing RSS URL in process.instance"
        socialmarket_logger.info(msg)
        timePrint(msg)
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; XanderBot/1.0)",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    }

    try:
        response = requests.get(target_url, headers=headers, timeout=15)

        if response.status_code != 200 or not (response.text and response.text.strip()):
            process["scrapeFailCount"] += 1
            msg = f"[StockTitan] ({process['scrapeFailCount']}) RSS error: status={response.status_code}, len={len(response.text or '')}"
            socialmarket_logger.info(msg)
            timePrint(msg)
            return None

        # Parse XML
        root = ET.fromstring(response.text)
        items = root.findall(".//item")

        if not items:
            process["scrapeFailCount"] += 1
            msg = f"[StockTitan] ({process['scrapeFailCount']}) RSS parsed but no <item> found"
            socialmarket_logger.info(msg)
            timePrint(msg)
            return None

        # Parse lastSaved
        last_saved_dt = None
        if lastSaved:
            try:
                last_saved_dt = datetime.strptime(lastSaved, "%d-%b-%y %I:%M%p")
            except:
                last_saved_dt = None

        post_obj_list = []
        idx = 0
        isLaterThanSaved_any = False

        for item in items:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            guid = (item.findtext("guid") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()

            # Convert pubDate (GMT) -> SGT
            # pubDate example: "Wed, 07 Jan 2026 12:25:00 GMT"
            post_datetime = None
            if pub_date:
                dt = parsedate_to_datetime(pub_date)  # tz-aware datetime (GMT/UTC)
                datetime_raw = dt.strftime("%d-%b-%y %I:%M%p")

                tz_offset_hours = (dt.utcoffset().total_seconds() / 3600) if dt.utcoffset() else 0
                sgt_offset = 8 - tz_offset_hours

                post_datetime = parse_and_adjust_datetime(
                    datetime_raw, "%d-%b-%y %I:%M%p", sgt_offset, "StockTitan"
                )
            # Verify if extracted date & time of post is later than previously saved in local
            isLaterThanSaved = (post_datetime is not None and lastSaved is None)

            if post_datetime and last_saved_dt:
                post_dt = datetime.strptime(post_datetime, "%d-%b-%y %I:%M%p")
                now = datetime.now()

                # Clamp future timestamps (keep your original behavior)
                if post_dt > now:
                    post_dt = now
                    # Format without leading zero hour
                    post_datetime = post_dt.strftime("%d-%b-%y %#I:%M%p") if platform.system() == "Windows" else post_dt.strftime("%d-%b-%y %-I:%M%p")

                if post_dt >= last_saved_dt:
                    isLaterThanSaved = True

            if isLaterThanSaved:
                isLaterThanSaved_any = True

            valid = (title and link and (isLaterThanSaved or lastSaved is None))

            if valid:
                # Extract ticker
                ticker = None
                try:
                    # https://www.stocktitan.net/news/ACI/....html  ->  ACI
                    after = link.split("/news/", 1)[1]
                    ticker_candidate = after.split("/", 1)[0].strip().upper()
                    if ticker_candidate and ticker_candidate.isalnum() and len(ticker_candidate) <= 8:
                        ticker = ticker_candidate
                except:
                    ticker = None

                cleaned_stocks_list = []
                if ticker:
                    cleaned_stocks_list.append({"name": ticker})

                process["scrapeFailCount"] = 0
                post_obj_list.append({
                    "idx": idx,
                    "content": title,
                    "url": link,
                    "dedupe_id": guid or link,
                    "time": post_datetime,
                    "isOwner": True,
                    "hasAttachments": False,
                    "valid": True,
                    "tickers": cleaned_stocks_list
                })
                idx += 1

            if lastSaved and post_datetime and last_saved_dt:
                post_dt = datetime.strptime(post_datetime, "%d-%b-%y %I:%M%p")
                if post_dt < last_saved_dt:
                    continue

            if lastSaved is None and idx > 0:
                return post_obj_list

        if isLaterThanSaved_any:
            return post_obj_list

        process["scrapeFailCount"] += 1
        msg = f"[StockTitan] ({process['scrapeFailCount']}) RSS fetch ok but nothing newer than lastSaved"
        socialmarket_logger.info(msg)
        timePrint(msg)
        return None

    except Exception as e:
        process["scrapeFailCount"] += 1
        msg = f"[StockTitan] ({process['scrapeFailCount']}) RSS exception occured:\n{e}"
        socialmarket_logger.info(msg)
        timePrint(msg)
        return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# MARKET HOURS HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_adjusted_trading_hrs():
    file_name_TradingHoursSet = "XANDER_TradingHours_Set.txt"
    file_path_TradingHoursSet = os.path.join(functionsFolder_path, file_name_TradingHoursSet)
    with open(file_path_TradingHoursSet, "r", encoding="utf-8") as file:
        val = file.read().strip()
        if (val in ["AM","PM"]):
            return val
        else:
            return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def is_market_open_sgt():
    sgt = pytz.timezone(XANDER_TIMEZONE)
    now = datetime.now(sgt)

    market_open_time = marketOpen  
    market_close_time = marketClose

    adjustedHours = check_adjusted_trading_hrs()
    
    if (adjustedHours is not None and adjustedHours == "AM"):
        market_close_time = marketHalfDay
    elif (adjustedHours is not None and adjustedHours == "PM"):
        market_open_time = marketHalfDay

    market_close_time = (datetime.combine(datetime.today(), market_close_time) - timedelta(minutes=1)).time() # Putting 1 min before to add buffer for closing orders

    # Weekday must be Monday (0) to Friday (4)
    # But allow Saturday if it's before market_close_time
    if now.weekday() == 5 and now.time() > market_close_time:
        return False
    elif now.weekday() > 5:
        return False

    # Handle crossing midnight (i.e., time range from 9:30 PM to 4:00 AM next day)
    if now.time() >= market_open_time or now.time() <= market_close_time:
        return True
    else:
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TICKER RESOLUTION HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def resolve_tickers(company_guess):
    guess = company_guess.strip().lower()
    normalized_guess = guess.replace(" ", "").replace(".", "").replace(",", "").replace("-", "")
    threshold = 95

    prefix_matches = []
    substring_matches = []
    fuzzy_matches = []

    # Step 1: Exact name match (before normalization)
    if guess in lower_name_map:
        ticker = lower_name_map[guess]
        if len(guess) > 3 and "$" not in ticker:
            return [ticker]  # Exact match is highest priority

    # Step 2a: Prefix match (normalized)
    for name, ticker in lower_name_map.items():
        name_norm = name.replace(" ", "").replace(".", "").replace(",", "").replace("-", "")
        if name_norm.startswith(normalized_guess) and len(name.strip()) > 3 and "$" not in ticker:
            prefix_matches.append(ticker)
    if prefix_matches:
        return list(set(prefix_matches))  # Deduplicate

    # Step 2b: Substring match (normalized)
    for name, ticker in lower_name_map.items():
        name_norm = name.replace(" ", "").replace(".", "").replace(",", "").replace("-", "")
        if normalized_guess in name_norm and len(name.strip()) > 3 and "$" not in ticker:
            substring_matches.append(ticker)
    if substring_matches:
        return list(set(substring_matches))  # Deduplicate

    # Step 3: Fuzzy match (normalized)
    normalized_name_map = {name.replace(" ", "").replace(".", "").replace(",", "").replace("-", ""): name for name in lower_name_map.keys()}
    matches = process.extract(normalized_guess, normalized_name_map.keys(), limit=5)

    for match_norm, score, _ in matches:
        original_name = normalized_name_map[match_norm]
        ticker = lower_name_map[original_name]
        if len(original_name.strip()) > 3 and "$" not in ticker and score >= threshold:
            fuzzy_matches.append(ticker)

    return list(set(fuzzy_matches)) if fuzzy_matches else []
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def swap_tickers(ticker_mentions, result_q):
    tickers = []
    for stock in ticker_mentions:
        ticker = None
        # Object format: {"name": "SHAZ"}
        if isinstance(stock, dict):
            ticker = stock.get("name")
        # Simple list format: ["SHAZ"]
        else:
            ticker = stock
        ticker = ticker.upper().strip().replace("$", "")
        tickers.append({
            "match": ticker,
            "keyword": None
        })
    result_q.put(tickers)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def identify_tickers(company_guesses, platform, result_q):
    tickers = []
    for company_guess in company_guesses:
        matches = resolve_tickers(company_guess)
        if len(matches) > 0:
            for match in matches:
                tickers.append({"match": match, "keyword": company_guess})
        else:
            socialmarket_logger.info(f"[{platform}] Unable to resolve ticker with: {company_guess}")
    result_q.put(tickers)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_market_cap_yahoo(ticker, extracted_company_name=None, validate_name=False):
    try:
        min_score = 90
        stock = yf.Ticker(ticker)
        info = stock.info

        if validate_name and extracted_company_name:
            socialmarket_logger.info(f"Fetching market cap for {ticker} with Yahoo name validation against: '{extracted_company_name}'")
        else:
            socialmarket_logger.info(f"Fetching market cap for {ticker} without Yahoo name validation")

        # Step 1: Validate company name match if provided
        if validate_name and extracted_company_name:
            official_name = info.get('longName') or info.get('shortName') or ''
            if not official_name:
                msg = f"Missing official name for {ticker}"
                socialmarket_logger.info(msg)
                return None

            extracted_clean = extracted_company_name.lower().strip()
            official_clean = clean_company_name(official_name.strip())

            # Token overlap safeguard
            extracted_tokens = set(re.findall(r'\w+', extracted_clean))
            official_tokens = set(re.findall(r'\w+', official_clean))
            if not (extracted_tokens & official_tokens):
                msg = f"No shared tokens — Extracted: '{extracted_company_name}', Official: '{official_name}'"
                socialmarket_logger.info(msg)
                return None

            # Adjust threshold for short names
            if len(extracted_clean.split()) == 1:
                min_score = 85

            # Fuzzy scores
            full_score = fuzz.ratio(extracted_clean, official_clean)
            partial_score = fuzz.partial_ratio(extracted_clean, official_clean)
            token_score = fuzz.token_sort_ratio(extracted_clean, official_clean)

            # Improved fuzzy match logic
            if full_score >= min_score or token_score >= min_score:
                pass
            elif partial_score == 100 and extracted_clean in official_clean:
                pass
            elif len(extracted_tokens & official_tokens) >= 1 and partial_score >= min_score:
                pass
            else:
                msg = f"Fuzzy match failed (full={full_score}, partial={partial_score}, token={token_score}) — Extracted: '{extracted_company_name}', Official: '{official_name}'"
                socialmarket_logger.info(msg)
                return None

        return info.get("marketCap", None)

    except Exception as e:
        msg = f"FAILED to get market cap for {ticker}: {e}"
        socialmarket_logger.info(msg)
        return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def monitor_ticker_resolution(result_queue, platform, timeout=10) -> dict:

    start_time = time_module.time()
    outputList = []

    while time_module.time() - start_time < timeout:

        if not result_queue.empty():

            resolved_tickers = result_queue.get()

            for resolved_ticker in resolved_tickers:

                ticker = resolved_ticker["match"]
                keyword = resolved_ticker["keyword"]
                socialmarket_logger.info(f"[{platform}] Ticker resolved in background thread: ${ticker}")

                market_cap = get_market_cap_yahoo(ticker, keyword, validate_name=(keyword is not None))

                if market_cap is not None:
                    socialmarket_logger.info(f"[{platform}] Market cap fetched for ${ticker}: ${market_cap:.2f}")
                    output = {'ticker': ticker, 'marketCap': market_cap, 'keyword': keyword, 'status': "SUCCESS"}
                    socialmarket_logger.info(f"[{platform}] Returning {output}")
                    outputList.append(output) 
                else:
                    socialmarket_logger.warning(f"[{platform}] Failed to fetch market cap for ${ticker}")
                    output = {'ticker': ticker, 'marketCap': None, 'keyword': keyword, 'status': "PARTIAL"}
                    socialmarket_logger.info(f"[{platform}] Returning {output}")
                    outputList.append(output) 

            return outputList

        time_module.sleep(0.2)

    socialmarket_logger.warning(f"[{platform}] Timeout waiting for ticker resolution.")
    output = {'ticker': None, 'marketCap': None, 'keyword': None, 'status': "FAIL"}
    socialmarket_logger.info(f"[{platform}] Returning {output}")
    return [output]
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def filter_single_success_candidates_by_keyword(resolved_candidates, platform):
    if not resolved_candidates:
        return resolved_candidates

    grouped_by_keyword = {}
    passthrough = []
    for item in resolved_candidates:
        keyword = item.get("keyword") if isinstance(item, dict) else None
        if keyword is None:
            passthrough.append(item)
            continue
        grouped_by_keyword.setdefault(keyword, []).append(item)

    filtered_candidates = list(passthrough)
    for keyword, candidates in grouped_by_keyword.items():
        success_candidates = [
            item for item in candidates
            if isinstance(item, dict) and item.get("status") == "SUCCESS"
        ]
        if len(success_candidates) == 1:
            socialmarket_logger.info(
                "[%s] Single SUCCESS ticker resolved for keyword '%s': keeping %s and discarding %d non-success candidates",
                platform,
                keyword,
                success_candidates[0].get("ticker"),
                len(candidates) - 1,
            )
            filtered_candidates.extend(success_candidates)
        else:
            filtered_candidates.extend(candidates)

    return filtered_candidates
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def launch_resolution_monitor(result_queue, platform, resolved_ticker_dict):
    result = monitor_ticker_resolution(result_queue, platform)
    if result:
        result = filter_single_success_candidates_by_keyword(result, platform)
        resolved_batch = {}
        for res in result:
            ticker = res.get("ticker")
            if ticker:  # Only add valid tickers
                resolved_batch[ticker] = res
        if resolved_batch:
            resolved_ticker_dict.update(resolved_batch)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def clean_company_name(name):
    return re.sub(
        r'[.,]?\s+(inc|incorporated|ltd|corp|co|llc|plc|corporation|group|holdings?|class a common stock)\.?$',
        '', name.lower(),
        flags=re.IGNORECASE
    ).strip()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def normalize_company_list(value):
    if not isinstance(value, list):
        value = [value]

    normalized = []
    seen = set()
    for item in value:
        item = str(item or "").strip()
        if not item:
            continue
        key = clean_company_name(item)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)

    return normalized
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def preferred_ticker_resolution_companies(result_json_mini, company_mentions):
    acquired_list = normalize_company_list((result_json_mini or {}).get("Acquired", []))
    if acquired_list:
        return "Acquired", acquired_list

    benefited_companies = normalize_company_list((result_json_mini or {}).get("BenefitedCompanies", []))
    if benefited_companies:
        return "BenefitedCompanies", benefited_companies

    return None, normalize_company_list(company_mentions)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def acquired_company_matches_keyword(acquired_company, keyword):
    acquired_clean = clean_company_name(str(acquired_company or ""))
    keyword_clean = clean_company_name(str(keyword or ""))
    return bool(
        acquired_clean
        and keyword_clean
        and (
            acquired_clean == keyword_clean
            or acquired_clean in keyword_clean
            or keyword_clean in acquired_clean
        )
    )
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def acquisition_target_resolved(result_json_mini, type_list, resolved_ticker_details):
    if result_json_mini.get("Category", "CAT_2") != "CAT_1_ACQUISITION":
        return True

    acquired_list = result_json_mini.get("Acquired", [])
    if not isinstance(acquired_list, list):
        acquired_list = [acquired_list]
    acquired_list = [item for item in acquired_list if item]

    if not acquired_list or not type_list:
        return False

    resolved_assets = {
        str(item.get("Asset", "")).replace("$", "").strip().upper()
        for item in type_list
        if isinstance(item, dict) and str(item.get("Asset", "")).strip().startswith("$")
    }
    if not resolved_assets:
        return False

    for ticker, info in resolved_ticker_details.items():
        resolved_ticker = str(info.get("ticker") or ticker or "").strip().upper()
        if resolved_ticker not in resolved_assets:
            continue
        if any(acquired_company_matches_keyword(acquired_company, info.get("keyword")) for acquired_company in acquired_list):
            return True

    return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def normalize_asset_ticker(asset):
    ticker = str(asset or "").replace("$", "").strip().upper()
    if not ticker or ticker in ("N.A.", "NA"):
        return None
    if not re.fullmatch(r"[A-Z0-9.\-]{1,8}", ticker):
        return None
    return ticker
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def is_successful_ticker_info(ticker_info):
    if not isinstance(ticker_info, dict):
        return False
    ticker = normalize_asset_ticker(ticker_info.get("ticker"))
    return bool(ticker and ticker_info.get("status") == "SUCCESS")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def sentiment_for_acquisition_target(ticker_info, original_type_list, fallback_sentiment):
    ticker = normalize_asset_ticker(ticker_info.get("ticker"))
    keyword = ticker_info.get("keyword")

    for item in original_type_list:
        if not isinstance(item, dict):
            continue
        asset = str(item.get("Asset", "")).strip()
        sentiment = item.get("Sentiment")
        if not sentiment:
            continue

        if normalize_asset_ticker(asset) == ticker:
            return sentiment

        if "$" not in asset and acquired_company_matches_keyword(asset, keyword):
            return sentiment

    return fallback_sentiment or "Bullish"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def apply_acquisition_target_filter(platform, result_json, result_json_mini, original_type_list, resolved_ticker_details, gpt_model):
    mini_category = result_json_mini.get("Category", "N.A.")
    acquired_list = normalize_company_list(result_json_mini.get("Acquired", []))

    if mini_category != "CAT_1_ACQUISITION" or not acquired_list:
        return False

    target_ticker_infos = [
        info for info in resolved_ticker_details.values()
        if is_successful_ticker_info(info)
    ]

    if not target_ticker_infos:
        socialmarket_logger.info(
            f"[{platform}] ACQUISITIONS target filter not applied: no successfully resolved acquired target tickers. "
            f"Acquired={acquired_list}; resolved={resolved_ticker_details}"
        )
        return False

    specialist_assets = [
        str(item.get("Asset", "")).strip()
        for item in original_type_list
        if isinstance(item, dict)
    ]
    specialist_tickers = [
        ticker for ticker in (normalize_asset_ticker(asset) for asset in specialist_assets if str(asset).strip().startswith("$"))
        if ticker
    ]
    target_tickers = []
    final_type_list = []
    seen = set()
    sentiments = [
        item.get("Sentiment")
        for item in original_type_list
        if isinstance(item, dict) and item.get("Sentiment")
    ]
    fallback_sentiment = sentiments[0] if len(set(sentiments)) == 1 and sentiments else "Bullish"

    for ticker_info in target_ticker_infos:
        ticker = normalize_asset_ticker(ticker_info.get("ticker"))
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        target_tickers.append(ticker)
        final_type_list.append({
            "Asset": f"${ticker}",
            "Sentiment": sentiment_for_acquisition_target(ticker_info, original_type_list, fallback_sentiment),
        })

    if not final_type_list:
        socialmarket_logger.info(
            f"[{platform}] ACQUISITIONS target filter not applied: resolved acquired targets were invalid. "
            f"Acquired={acquired_list}; resolved={resolved_ticker_details}"
        )
        return False

    excluded_tickers = [ticker for ticker in specialist_tickers if ticker not in target_tickers]
    result_json["Type"] = final_type_list

    socialmarket_logger.info(
        f"[{platform}] ACQUISITIONS target filter active.\n"
        f"Source/article tickers: {specialist_tickers}\n"
        f"Specialist assets: {specialist_assets}\n"
        f"Resolved acquired targets: {target_tickers}\n"
        f"Final actionable tickers after acquisition filter: {target_tickers}\n"
        f"Excluded tickers: {excluded_tickers} reason=not_in_acquired_target_list"
    )

    for ticker in target_tickers:
        socialmarket_logger.info(f"[{platform}] Executing ticker resolution from {gpt_model} Response with {ticker}")

    return True
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def attempt_ticker_resolution(platform, result_json, result_json_mini, company_mentions, resolved_ticker_details, gpt_model):

    type_list = result_json.get("Type", [])
    category = result_json.get("Category", "N.A.")
    mini_category = result_json_mini.get("Category", "N.A.")

    if category == "CAT_1" and len(company_mentions) > 0:

        preferred_source, preferred_companies = preferred_ticker_resolution_companies(result_json_mini, company_mentions)

        # Prefer the intended economic target when Stage 1 identifies one.
        try:
            target_companies = preferred_companies if preferred_source else None
        except Exception:
            target_companies = None

        if target_companies:
            if not isinstance(target_companies, list):
                target_companies = [target_companies]

            target_clean = [clean_company_name(x) for x in target_companies if x]

            filtered = {}
            for tk, info in list(resolved_ticker_details.items()):
                kw = info.get("keyword") or ""
                if any(acquired_company_matches_keyword(target, kw) for target in target_clean):
                    filtered[tk] = info

            if len(filtered) == 0 or len(filtered) > len(target_companies):
                resolved_ticker_details = {
                    "N.A.": {"ticker": target_companies[0], "marketCap": 0, "keyword": "N.A.", "status": "SUCCESS"}
                }
            else:
                resolved_ticker_details = filtered

            socialmarket_logger.info(f"[{platform}] {preferred_source} list detected in mini response: {target_companies} - filtering resolved tickers to {resolved_ticker_details}")

        has_unauth_item = sum(item.get("Asset") not in approved_sectors and "$" not in item.get("Asset") for item in type_list) > 0
        nas = [item.get("Asset") for item in type_list if item.get("Asset") == "N.A."]
        all_nas = len(set(nas)) == 1
        only_one_sector = (
            len(type_list) == 1 and 
            type_list[0].get("Asset") in approved_sectors
        )
        multiple_all_sectors = (
            len(type_list) > 1 and
            all(item.get("Asset") in approved_sectors for item in type_list)
        )
        sentiments = [item.get("Sentiment") for item in type_list if item.get("Sentiment") is not None]
        all_same_sentiment = len(set(sentiments)) == 1
        all_company_names = all(
            item.get("Asset") not in approved_sectors
            for item in type_list
        )

        if not all(item['keyword'] is None for item in resolved_ticker_details.values()):
            # Distinct resolved_ticker_details by taking highest market cap
            keyword_map = {}
            for item in resolved_ticker_details.values():
                keyword = item['keyword']
                market_cap = item['marketCap']
                if keyword not in keyword_map or (keyword_map[keyword]['marketCap'] is not None and market_cap is not None and market_cap > keyword_map[keyword]['marketCap']):
                    keyword_map[keyword] = item

            resolved_ticker_details = {item["ticker"]: item for item in keyword_map.values() if item['status'] == "SUCCESS"}

        else:

            resolved_ticker_details = {item["ticker"]: item for item in resolved_ticker_details.values() if item['status'] == "SUCCESS"}

        if apply_acquisition_target_filter(platform, result_json, result_json_mini, type_list, resolved_ticker_details, gpt_model):
            return
            
        isSkipped = False
        should_attempt_swap = (len(resolved_ticker_details) == 1 and (has_unauth_item or only_one_sector))
        should_attempt_addition = (len(resolved_ticker_details) > 0 and (has_unauth_item or multiple_all_sectors) and all_same_sentiment)
        should_attempt_renewal = (len(resolved_ticker_details) > 0 and (only_one_sector or all_nas or all_company_names or preferred_source))
        
        if should_attempt_swap:

            ticker = next(iter(resolved_ticker_details.values()))["ticker"] # Only when resolved_ticker_details is length : 1

            if not ticker:
                isSkipped = True
            else:
                socialmarket_logger.info(f"[{platform}] Executing ticker resolution from {gpt_model} Response with {ticker}")

                for item in type_list:
                    if (item.get("Asset") not in approved_sectors and "$" not in item.get("Asset")) or item.get("Asset") in approved_sectors:
                        item["Asset"] = ("$" + ticker) if ticker != "N.A." else ticker

                result_json["Type"] = type_list

        elif (should_attempt_addition or should_attempt_renewal):

            base_sentiment = sentiments[0]
            if should_attempt_renewal:
                type_list = []
            else:
                type_list = [item for item in type_list if item.get("Asset") not in approved_sectors and "$" not in item.get("Asset")]

            for ticker_info in resolved_ticker_details.values():
                ticker = ticker_info.get("ticker")
                if ticker:
                    new_item = {
                        "Asset": ("$" + ticker) if (ticker != "N.A." and ticker.isupper()) else ticker,
                        "Sentiment": base_sentiment
                    }
                    type_list.append(new_item)
                    socialmarket_logger.info(f"[{platform}] Executing ticker resolution from {gpt_model} Response with {ticker}")

            result_json["Type"] = type_list

        else:
            isSkipped = True

        if isSkipped:
            socialmarket_logger.info(f"[{platform}] Skipping ticker resolution from 4o Response.")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# AI RESPONSE / CLASSIFICATION HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_fgi_value(timeout=3):
    try:
        result = subprocess.run(
            [sys.executable, "SocialMarket_FGI.py"],
            stdout=subprocess.PIPE,
            timeout=timeout,
            text=True,
            cwd=bot_path(".BOT_Launch", "SocialMarket")
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "N.A."
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_4o_mini_flags(result_json_mini):

    mini_flags = [
        "TickerMentioned",
        "ForeignIndexMention",
        "CryptoRelated",
        "ThirdPartyForecast",
        "ValidThirdParty",
        "CommodityFocused",
        "GeopoliticalConflict",
        "TrumpReference",
        "TrumpStatement",
        "FedSpeaker",
        "USPolicyAction",
        "IsSectorSpecific",
        "ValidOptions",
        "MarketReactionOverview",
        "FDAClinicalTrialRelated",
        "EquityDilutionEvent",
        "ReverseSplitEvent",
        "BuybackEvent",
        "PartnershipCollaborationEvent",
        "IndexInclusionEvent",
        "MacroData",
        "ValidMacroIndicator",
        "HasForwardTimingKeyword"
    ]

    if result_json_mini is None:
        return {flag: False for flag in mini_flags}
    
    present_flags = set(result_json_mini.get("TriggerFlags", []))
    return {flag: (flag in present_flags) for flag in mini_flags}
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def validate_mini(result_json_mini, mini_flags, platform):

    flags_blocker = [
        "ForeignIndexMention",
        "CryptoRelated",
        "CommodityFocused",
        "MarketReactionOverview",
        "FDAClinicalTrialRelated"
    ]

    conditional_flags_blocker = [ # Indexing must match with conditional_flags_escalation
        "ThirdPartyForecast",
        "IsSectorSpecific"
        
    ]

    conditional_flags_escalation = [ # Indexing must match with conditional_flags_blocker
        "ValidThirdParty",
        "TickerMentioned"
    ]

    flags_escalation = [
        "TickerMentioned",
        "ValidThirdParty",
        "GeopoliticalConflict",
        "TrumpReference",
        "TrumpStatement",
        "FedSpeaker",
        "USPolicyAction",
        "ValidOptions",
        "EquityDilutionEvent",
        "ReverseSplitEvent",
        "BuybackEvent",
        "PartnershipCollaborationEvent",
        "IndexInclusionEvent",
        "MacroData",
        "ValidMacroIndicator",
        "HasForwardTimingKeyword"
    ]

    company_mentions = result_json_mini.get("CompanyMentions", [])
    category = result_json_mini.get("Category", "CAT_2")
    decision = result_json_mini.get("Decision", "DROP")

    if decision == "DROP":
        socialmarket_logger.info(f"[{platform}] Downgraded to CAT_2 — Decision field is {decision}")
        result_json_mini["Category"] = "CAT_2"

    elif category != "CAT_2":

        if any(mini_flags.get(flag, False) for flag in flags_blocker):
            socialmarket_logger.info(f"[{platform}] Downgraded to CAT_2 — blocked by non-actionable soft trigger(s)")
            result_json_mini["Category"] = "CAT_2"

        else: 
            
            for i in range(0, len(conditional_flags_blocker)):
                blocker_flag = conditional_flags_blocker[i]
                escalation_flag = conditional_flags_escalation[i]

                if mini_flags.get(blocker_flag, False) and not mini_flags.get(escalation_flag, False):
                    socialmarket_logger.info(f"[{platform}] Downgraded to CAT_2 — blocked by non-actionable soft trigger(s)")
                    result_json_mini["Category"] = "CAT_2"
                    break
        
            if (result_json_mini["Category"] != "CAT_2") and all(not mini_flags[flag] for flag in flags_escalation) and (len(company_mentions) == 0):
                socialmarket_logger.info(f"[{platform}] Downgraded to CAT_2 — no escalation flags triggered and no company mentions")
                result_json_mini["Category"] = "CAT_2"
        
    else:
        result_json_mini["Category"] = "CAT_2"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def classify_gpt_stage2_category(result_json_mini, mini_flags, platform):

    stage2_cat = None
    company_mentions = result_json_mini.get("CompanyMentions", [])
    category = result_json_mini.get("Category", "CAT_2")

    if category == "CAT_1_EARNINGS":
        stage2_cat = "EARNINGS"

    elif category == "CAT_1_CONTRACT":
        stage2_cat = "CONTRACTS"

    elif category == "CAT_1_ACQUISITION":
        stage2_cat = "ACQUISITIONS"

    elif mini_flags['PartnershipCollaborationEvent']:
        stage2_cat = "PARTNERSHIPS"

    elif mini_flags['IndexInclusionEvent']:
        stage2_cat = "INDEX_INCLUSION"

    elif mini_flags['MacroData']:
        stage2_cat = "MACRODATA"
        
    elif mini_flags['ValidOptions']:
        stage2_cat = "OPTIONS"

    elif mini_flags['TrumpStatement']:

        if mini_flags['HasForwardTimingKeyword']:
            stage2_cat = "DEADLINE"

        elif mini_flags['TrumpReference'] and (len(company_mentions) > 0 or mini_flags['TickerMentioned']):
            stage2_cat = "OTHERS"

        else:
            stage2_cat = "TRUMP"

    elif mini_flags['GeopoliticalConflict']:
        stage2_cat = "GEOPOLITICAL"

    elif mini_flags['ValidMacroIndicator']:
        stage2_cat = "MACROINDICATOR"

    elif mini_flags['HasForwardTimingKeyword']:
        stage2_cat = "DEADLINE"

    elif (len(company_mentions) > 0 or mini_flags['TickerMentioned']) and (mini_flags['EquityDilutionEvent'] or mini_flags['ReverseSplitEvent'] or mini_flags['BuybackEvent']):
        stage2_cat = "CAPITAL_STRUCTURE"

    elif (len(company_mentions) > 0 or mini_flags['TickerMentioned']):
        stage2_cat = "OTHERS"

    elif mini_flags['FedSpeaker']:
        stage2_cat = "FED"

    elif mini_flags['USPolicyAction']:
        stage2_cat = "POLICY"

    else:
        stage2_cat = "OTHERS"

    if (len(company_mentions) ==  0): # BLOCKING ALL ASSESSMENTS WITH NO COMPANY / TICKER MENTIONS
        stage2_cat = "BLOCKED"
    
    socialmarket_logger.info(f"[{platform}] Classified into Stage 2 category: {stage2_cat}")
    
    return stage2_cat
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def retry_parsing(response, platform, ai_type):
    attempts = 0
    max_attempts = 3
    previous_response = None  # Variable to store the previous response
    socialmarket_logger.info(f"[{platform}] Parsing below response from OpenAi (gpt-{ai_type.lower()})..")

    while attempts < max_attempts:
        attempts += 1
        try:
            # Attempt to parse the response
            # Remove ```json or ``` wrappers and reduce extra curly braces if needed
            cleaned_response = re.sub(r"^```json\s*|```$", "", response.strip(), flags=re.MULTILINE)
            cleaned_response = re.sub(r"\{\{", "{", cleaned_response)
            cleaned_response = re.sub(r"\}\}", "}", cleaned_response)
            if (ai_type != "5.4-mini-sections"):
                socialmarket_logger.info(str(cleaned_response))
            parsed_response = json.loads(cleaned_response)
            return parsed_response
        
        except json.JSONDecodeError:
            # Increment attempt counter
            if attempts <= max_attempts:
                retry_msg = f"Unable to parse successfully. Retrying... Attempt {attempts}/{max_attempts}"
                socialmarket_logger.info(retry_msg)
                timePrint(retry_msg)
                
                response_text = response if previous_response is None else previous_response
                error_message = f"""
Attempt {attempts}: Failed to parse JSON response. Please review the response provided and provide a corrected version.

Instructions:
- Return **only valid JSON** — no commentary, markdown, or explanation.
- Match the structure and field names exactly.
- Ensure all strings are properly quoted and brackets are closed.

Additional rules:
- Ensure the full response is complete — do not truncate or cut off any part of the JSON.
- Do not include any markdown formatting (e.g., backticks) or explanations.
- If any field is null (e.g., "ThoughtProcess": null), replace it with the correct empty string or value according to the format.
- Do not rename or omit any required fields. Field names must match the expected schema exactly.

"""
                
                if ai_type == "4o-mini-classification":
                  error_message += f"""
Expected format:
{{
  "Category": "CAT_2" or "CAT_1_POSSIBLE" or "CAT_1_EARNINGS" or "CAT_1_ACQUISITION" or "CAT_1_CONTRACT
  "CompanyMentions": ["Apple", "Tesla"]
  "ShortReason": "Only include if the original response included a 'ShortReason' field. Otherwise, leave as an empty string."
}}
""" 
                elif ai_type in ["4o-mini-flags-1", "4o-mini-flags-2"]:
                  error_message += f"""
Expected format:
{{
  "TriggerFlags": ["ThirdPartyForecast", "ValidMacroIndicator", "TrumpStatement"],
  "Tickers": ["Only include if the original response included a 'Tickers' field. Otherwise, leave as an empty list."]
  "BenefitedCompanies": ["Only include if the original response included a 'BenefitedCompanies' field. Otherwise, leave as an empty list."]
}}
"""
                elif ai_type == "4o-mini-triage":
                  error_message += f"""
Expected format:
{{
  "Decision": ["ESCALATE_4O", "DROP", "MINI_ONLY"]
}}
"""
                elif ai_type == "4o-mini-sections":
                  error_message += f"""
Expected format:
{{
  "Sections": ["Systemic Actor Ruleset: Donald Trump", "Funding & Market Structure Signals", "Handling Vague or Ambiguous Posts"],
  "NeedMarketContext": false
}}
"""
                else:
                  error_message += f"""
Expected format:
{{
  "Category": "CAT_1 / CAT_2",
  "Region": "USA" / "EU" / "China" / "Asia" / "Others" / "General",
  "Type": [
    {{
      "Asset": "<Ticker or Sector or 'General' or 'N.A.'>",
      "Sentiment": "Bullish" / "Bearish"
    }}
  ],
  "Details": "Only include if the original response included a 'Details' field. Otherwise, leave as an empty string.",
  "ThoughtProcess": "Only include if the original response included a 'ThoughtProcess' field. Otherwise, leave as an empty string."
}}
"""
                    
                error_message += f"""

Do not return anything else. Output must be a **single, parsable JSON object**.
Do not change or invent any values — only fix formatting issues or required structural corrections.

Response:

-----------------------------

ORIGINAL RESPONSE:

{response_text}

-----------------------------
"""
                
                try:
                    # Get new response from OpenAI
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": error_message}],
                        temperature=0.3  # Parameter that controls the randomness or creativity of the response
                    ).choices[0].message.content.strip()

                    # Update previous response for the next iteration
                    previous_response = response

                except Exception as e:
                    retry_msg = f"Timeout / error occured as below for getting response from OpenAi."
                    socialmarket_logger.info(retry_msg)
                    timePrint(retry_msg)
                    socialmarket_logger.info(e)
                    print(e)
                
            else:
                error_msg = "⚠️ Maximum retry limit reached - unable to parse response given by OpenAi. Check logs for exception."
                socialmarket_logger.info(error_msg)
                send_telegram_message_thread(error_msg, 0, 0)
                timePrint(error_msg)
                break

    return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_openai_with_retries(call_fn, platform, model_label, max_attempts=3):
    attempts = 0
    response = None

    while attempts <= max_attempts:
        attempts += 1
        try:
            response = call_fn()
            break

        except Exception as e:
            if attempts <= max_attempts:
                retry_msg = f"[{platform}] Timeout / error occured as below for getting response from OpenAi ({model_label}). Retrying... Attempt {attempts}/{max_attempts}"
                socialmarket_logger.info(retry_msg)
                timePrint(retry_msg)
                socialmarket_logger.info(e)
                print(e)
                wait_time = 5  # default fallback
                error_text = str(e)
                match = re.search(r"try again in ([\d\.]+)s", error_text)
                if match:
                    wait_time = float(match.group(1))
                time_module.sleep(wait_time)
            else:
                error_msg = f"[{platform}] ⚠️ Maximum retry limit reached - unable to get response from OpenAi ({model_label}). Check logs for exception."
                socialmarket_logger.info(error_msg)
                send_telegram_message_thread(error_msg, 0, 0)
                timePrint(error_msg)

    return response
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_gpt_mini(post_obj, platform, type, model):

    attempts = 0
    max_attempts = 3
    
    socialmarket_logger.info(f"[{platform}] Getting {type} response from OpenAi ({model})..")

    if isinstance(post_obj, dict) and post_obj.get('content'):
        post = post_obj['content']
    else:
        post = post_obj

    if (type == "CLASSIFICATION"):
        full_path = os.path.join(prompts_path, "4O-MINI-CLASSIFICATION.txt")
    elif (type == "FLAGS-1"):
        full_path = os.path.join(prompts_path, "4O-MINI-FLAGS-1.txt")
    elif (type == "FLAGS-2"):
        full_path = os.path.join(prompts_path, "4O-MINI-FLAGS-2.txt")
    elif (type == "TRIAGE"):
        full_path = os.path.join(prompts_path, "4O-MINI-TRIAGE.txt")
    else:
        full_path = os.path.join(prompts_path, "54-MINI-SECTIONS.txt")
    
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as file:
            gpt4ominiPrompt = file.read()

    source_prefix = ""
    if type == "TRIAGE":
        source_user = None
        if isinstance(post_obj, dict):
            source_user = post_obj.get("user") or post_obj.get("display")

        source_names = {
            "DeItaone": "DeItaone",
            "realDonaldTrump": "Trump",
            "StockTitan": "Stock Titan"
        }
        source_label = source_names.get(source_user, source_user or platform)
        source_prefix = f"[{source_label}]\n\n"

    prompt = gpt4ominiPrompt
    prompt += f"""
-----------------------------

POST BELOW:

{source_prefix}{post.upper()}

-----------------------------
    """

    response = run_openai_with_retries(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=20
        ).choices[0].message.content.strip(),
        platform,
        model
    )

    # Pass the response to retry_parsing function
    if response:
        parsed_response = retry_parsing(response, platform, f"{model.replace("gpt-","")}-{type.lower()}")
        if parsed_response:
            return parsed_response
    
    return None  # Return None if all else fail
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def assess_gpt_model(stage2_category):
    if stage2_category.strip().upper() in stage2_gpt4omini_prompts:
        return "gpt-4o-mini"
    elif stage2_category.strip().upper() in stage2_gpt41mini_prompts: 
        return "gpt-4.1-mini"
    else:
        return "gpt-4o"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def validate_sections(sections_response, mini_flags):
    market_context_section = "Market Context Integration Requirement (Global Rule)"
    general_inference_section = "Asset Field Assignment: General Inference Logic"
    ticker_inference_section = "Hard Asset Resolution & Inference Suppression Framework"

    sections = sections_response.get("Sections", [])
    need_market_context = sections_response.get("NeedMarketContext", False)

    has_market_context_section = market_context_section in sections
    has_general_inference_section = general_inference_section in sections
    has_ticker_inference_section = ticker_inference_section in sections

    if not has_general_inference_section and not mini_flags['TickerMentioned']:
        sections.append(general_inference_section)

    elif has_general_inference_section and mini_flags['TickerMentioned']:
        sections.remove(general_inference_section)

    if has_ticker_inference_section and not mini_flags['TickerMentioned']:
        sections.remove(ticker_inference_section)

    elif not has_ticker_inference_section and mini_flags['TickerMentioned']:
        sections.append(ticker_inference_section)

    if has_market_context_section:
        sections_response["NeedMarketContext"] = True

    elif need_market_context is True:
        sections.append(market_context_section)
        sections_response["Sections"] = sections

    else:
        sections_response["NeedMarketContext"] = False

    socialmarket_logger.info(json.dumps(sections_response, indent=2, ensure_ascii=False))

    return sections_response
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_stage2_gpt(post_obj, platform, stage2_category, gpt_model, dynamic_prompt_result, needMarketContext):

    # Initialize
    isDebugMode = False

    # Get prompt
    category_filename = f"{stage2_category}.txt"
    full_path = os.path.join(prompts_path, category_filename)
    
    if (os.path.exists(full_path)) and (dynamic_prompt_result is None):
        with open(full_path, 'r', encoding='utf-8') as file:
            selectedPromptLogic = file.read()
    else:
        if hasattr(dynamic_prompt_result, "prompt"):
            selectedPromptLogic = dynamic_prompt_result.prompt
        else:
            selectedPromptLogic = str(dynamic_prompt_result)
    
    # Log start
    socialmarket_logger.info(f"[{platform}] Getting response from OpenAi ({gpt_model})..")

    # Format posts directly from TRUMP
    if isinstance(post_obj, dict) and post_obj.get('user') == "realDonaldTrump":
      post_obj['content'] = "TRUMP: " + post_obj['content']
    
    brief = f"Please classify the following post into one of the 2 categories below. Your goal is to give repeatable, objective, and deterministic classifications-similar posts should always result in the same category"

    # Handle DEBUG mode
    if isinstance(post_obj, dict) and post_obj.get('content'):
        post = post_obj['content']
    else:
        if platform == "DEBUG":
            isDebugMode = True
        post = post_obj

    date_string = datetime.today().strftime("%d %B %Y")
    # Prepare market context
    if needMarketContext:
      asOfToday = f"As of today, {date_string}."

      with open(marketContext_snapshot_path, 'r', encoding='utf-8') as file:
              market_content = file.read().strip()

    # System Instructions to GPT 4o (AI Agent)
    system_instructions = f"""Today's date is {date_string}. You are a deterministic classification engine for short-form financial related posts.

Your job is to apply the user-supplied classification rules **exactly as written** to decide if a post is:
- CAT_1: actionable (likely to cause market repricing), or
- CAT_2: non-actionable (no clear trading trigger)

🛑 Do not infer, speculate, or apply intuition. Obey the rules, overrides, and gating logic only.

If rules do not clearly support CAT_1, default to CAT_2.

You are not a chatbot — you are a rules processor.

The post has been pre-routed. Do not second-guess its topic.
"""

    # Set up prompt to GPT 4o / 4o-mini
    prompt = brief
    if needMarketContext:
      prompt += f"""

# ------------------------------------------------------------------------------------------

# ============================== MARKET CONTEXT ==============================

{asOfToday}

{market_content}

# ------------------------------------------------------------------------------------------

"""
      
    prompt += selectedPromptLogic

    prompt += f"""
### Post to be CLASSIFIED

-----------------------------

{post.upper()}

-----------------------------

---
"""

    if isDebugMode or gpt_model in ["gpt-4o-mini"]:
        prompt += """
### Model Execution Instructions

- Carefully analyze the post and evaluate it against **all classification rules and conditions**, following the **tiered hierarchy structure (Tier S > A > B > C > D)**  
- If multiple rules match, you must apply **only the highest-priority tier** — lower-tier rules are always overridden  
- Ensure the output fully complies with **all gating, enforcement, and sentiment logic** described in the prompt  
- Respond **only** with the structured JSON output in the exact format below
- If multiple companies or tradeable entities are referenced in the post, **all must be explicitly listed** in the `Type` array.
  - Assign `"Sentiment": "Bullish"` or `"Bearish"` only to entities with clear, tradeable directional impact.
  - Assign `"Sentiment": "N.A."` to any referenced entity with **no clear directional or repricing implication**.
  - Do **not** omit secondary or parent entities solely because they lack directional sentiment.
- Freshness Gate (Mandatory):
  - Any post referencing completed actions, contracts, results, or events **from prior days or earlier periods** (including vague timing such as “in Q4”, “last quarter”, “earlier this week”) must be treated as **outdated** unless explicitly framed as a *new disclosure today*.
  - Outdated information is **ineligible for CAT_1** and must be classified as **CAT_2**, regardless of directional strength.

---

### JSON Format:

Respond in this format only:

{{
  "Category": "CAT_1 / CAT_2",
  "Region": "USA" / "EU" / "China" / "Asia" / "Others" / "General",
  "Type": [
	{{
	  "Asset": "<If a ticker, format as '$TICKER'. Otherwise, use Sector or 'General' or 'N.A.' — follow the priority and gating rules in the Output Format 'Type' section>",
	  "Sentiment": "Bullish" / "Bearish"
	}}
  ],
  "Details": "For options trades, list only tickers with >$500k notional size using the exact sentence format: "Significant options trades indicate potential market moves for [TICKER_1], [TICKER_2], and [TICKER_3].' If not options-related, leave this value empty",
  "ThoughtProcess": "For options trades, leave this field empty. For all other posts, provide a concise and structured explanation detailing:
- Clearly identify **which tier(s)** (S, A, B, C, or D) were triggered during evaluation, and **how many total tiers were flagged**  
  → Explicitly state which rule(s) belong to each tier that matched  
  → Explain which tier was ultimately applied based on the tier hierarchy (S > A > B > C > D)  
  → If multiple rules matched across different tiers, explain **why only the highest-priority tier was used** and how it overrode the others 
- How directional sentiment was inferred using which specific section & prompt rule(s) / condition

The explanation should reflect a clear decision logic path, derived directly from the relevant prompt rules. Avoid summarizing the post — instead, focus on the classification reasoning as a compact rule-based decision tree."
}}
"""

    else:
        prompt += """
### Model Execution Instructions

- Carefully analyze the post and evaluate it against **all classification rules and conditions**, following the **tiered hierarchy structure (Tier S > A > B > C > D)**  
- If multiple rules match, you must apply **only the highest-priority tier** — lower-tier rules are always overridden  
- Ensure the output fully complies with **all gating, enforcement, and sentiment logic** described in the prompt  
- Respond **only** with the structured JSON output in the exact format below
- **DO NOT** include any commentary, explanation, or reasoning 
- If multiple companies or tradeable entities are referenced in the post, **all must be explicitly listed** in the `Type` array.
  - Assign `"Sentiment": "Bullish"` or `"Bearish"` only to entities with clear, tradeable directional impact.
  - Assign `"Sentiment": "N.A."` to any referenced entity with **no clear directional or repricing implication**.
  - Do **not** omit secondary or parent entities solely because they lack directional sentiment. 
- Freshness Gate (Mandatory):
  - Any post referencing completed actions, contracts, results, or events **from prior days or earlier periods** (including vague timing such as “in Q4”, “last quarter”, “earlier this week”) must be treated as **outdated** unless explicitly framed as a *new disclosure today*.
  - Outdated information is **ineligible for CAT_1** and must be classified as **CAT_2**, regardless of directional strength.

---

### JSON Format:

Respond in this format only:

{{
  "Category": "CAT_1 / CAT_2",
  "Region": "USA" / "EU" / "China" / "Asia" / "Others" / "General",
  "Type": [
	{{
	  "Asset": "<If a ticker, format as '$TICKER'. Otherwise, use Sector or 'General' or 'N.A.' — follow the priority and gating rules in the Output Format 'Type' section>",
	  "Sentiment": "Bullish" / "Bearish"
	}}
  ],
  "Details": "For options trades, list only tickers with >$500k notional size using the exact sentence format: "Significant options trades indicate potential market moves for [TICKER_1], [TICKER_2], and [TICKER_3].' If not options-related, leave this value empty"
}}
"""

    
    # Get GPT 4o to ASSESS
    response = run_openai_with_retries(
        lambda: client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            timeout=20
        ).choices[0].message.content.strip(),
        platform,
        gpt_model
    )

    # Pass the response to retry_parsing function
    if response:
        parsed_response = retry_parsing(response, platform, gpt_model.replace("gpt-",""))
        if parsed_response:
            return parsed_response
    
    return None  # Return None if all else fail
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# POST ASSESSMENT PIPELINE
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def post_assessment_key(platform, post_obj):
    if platform == "StockTitan":
        return post_obj["url"]
    return f"{platform}:{post_obj.get('id') or post_obj.get('content')}"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def build_post_info_message(post_obj, platform, platform_name, user, post_id):
    if user in ("Benzinga", "StockTitan"):
        tickers_list = post_obj.get("tickers", [])
        tickers = ""
        for ticker_obj in tickers_list:
            tickers += "$" + str(ticker_obj.get("name", "")) + " , "
        if tickers:
            tickers = tickers[:-3]

        return (
            f"[{platform_name}] - {post_id}\n"
            f"👤 User: {user}\n"
            f"🕒 Date & Time (SGT): {post_obj.get('time')}\n"
            f"🔑 Ticker(s): {tickers}\n"
            f"{post_obj.get('url', '')}\n\n"
            f"📝 {post_obj.get('content')}"
        )

    if platform == "Telegram":
        return (
            f"[{platform_name}] - {post_id}\n"
            f"👤 User: {user}\n"
            f"🕒 Date & Time (SGT): {post_obj.get('time')}\n"
            f"📝 {post_obj.get('content')}"
        )

    return (
        f"[{platform_name}] - {post_id}\n"
        f"👤 User: {user}\n"
        f"🔑 Is Owner: {post_obj.get('isOwner')}\n"
        f"🕒 Date & Time (SGT): {post_obj.get('time')}\n"
        f"{post_obj.get('url', '')}\n\n"
        f"📝 {post_obj.get('content')}"
    )
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def parallel_stage1_4omini(post_obj, platform):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_gpt_mini, post_obj, platform, "CLASSIFICATION", "gpt-4o-mini"),
            executor.submit(run_gpt_mini, post_obj, platform, "FLAGS-1", "gpt-4o-mini"),
            executor.submit(run_gpt_mini, post_obj, platform, "FLAGS-2", "gpt-4o-mini"),
            executor.submit(run_gpt_mini, post_obj, platform, "TRIAGE", "gpt-4o-mini"),
        ]
        results = [future.result() for future in futures]

    merged_results = {}

    for result in results:
        if not isinstance(result, dict):
            continue

        for key, value in result.items():
            if key == "TriggerFlags":
                existing = merged_results.get("TriggerFlags", [])

                if isinstance(value, list):
                    existing.extend(value)
                else:
                    existing.append(value)

                # remove duplicates while preserving order
                merged_results["TriggerFlags"] = list(dict.fromkeys(existing))
            else:
                merged_results[key] = value

    return merged_results
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def emit_or_buffer_tele(text, withholdTele, withholdMessages, aiEnabled, discretionaryAiEnabled):
    if not withholdTele:
        send_telegram_message_thread(text, 0, 0)
        return

    if discretionaryAiEnabled or aiEnabled:
        withholdMessages.append(text)
    else:
        send_telegram_message_thread(text, 0, 0)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def is_post_recent_enough(post_obj, platform, lastSaved):
    if lastSaved is None:
        return True

    try:
        post_time_str = post_obj['time']
        post_time = datetime.strptime(post_time_str, "%d-%b-%y %I:%M%p")
    except Exception as e:
        socialmarket_logger.info(f"[{platform}] Invalid post time format: {e}")
        return False

    now_sgt = datetime.now()
    time_diff = abs((now_sgt - post_time).total_seconds()) / 60.0

    if platform == "StockTitan":
        return time_diff <= 5

    if platform == "Benzinga":
        return time_diff <= 1

    return time_diff <= 10
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def wait_for_ticker_resolution(resolved_ticker_details, interval=0.1, timeout=10):
    start_time = time_module.time()
    while time_module.time() - start_time < timeout:
        if resolved_ticker_details and all("status" in item for item in resolved_ticker_details.values()):
            return True
        time_module.sleep(interval)
    return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def start_ticker_resolution(worker_fn, worker_args, platform_name, resolved_ticker_details):
    result_queue = queue.Queue()
    threading.Thread(target=worker_fn, args=(*worker_args, result_queue), daemon=True).start()
    threading.Thread(target=launch_resolution_monitor, args=(result_queue, platform_name, resolved_ticker_details), daemon=True).start()
    return result_queue
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fetch_and_process_post(process, user, display_name, platform, platform_name, simulation, tele_obj=None):
    try:
        # Scrape & Assess
        if simulation is None and process and process.get('display') not in ["NR", "SIMULATE", "DEBUG"]:
            is_live_scraping = True
                
            if (platform != "Telegram"):
                lastSaved = getSavedPostTime(platform, user)
            else:
                lastSaved = None

            if platform == "X":
                post_obj_list = getNewPosts_X(user, display_name, process, lastSaved)
            elif platform == "TS" and user == "realDonaldTrump":
                post_obj_list = getTrumpNewPosts_TS(display_name, process, lastSaved)
            elif platform == "Benzinga":
                post_obj_list = getNewPosts_Benzinga(process, lastSaved)
                purgeUrlHistory()
            elif platform == "StockTitan":
                post_obj_list = getNewPosts_StockTitanRSS(process, lastSaved)
            elif platform == "Telegram":
                post_obj_list = [tele_obj]
            
            if post_obj_list:

                withholdTele = False

                if (user == "StockTitan"):
                    dumpStart = stockTitanDumpStart
                    dumpEnd = marketClose
                    now_sgt = datetime.now().time()
                    isDumpPeriod = (now_sgt >= dumpStart) or (now_sgt <= dumpEnd)
                    isWeekday = datetime.now().weekday() < 6   # Mon–Sat only
                    if (isDumpPeriod) and (isWeekday):
                        withholdTele = True

                for post_obj in reversed(post_obj_list):

                    if post_obj and post_obj.get('valid') is True:

                        canProceed = True
                        withholdMessages = []

                        if (lastSaved is not None):
                            canProceed = is_post_recent_enough(post_obj, platform, lastSaved)

                        if (platform != "Telegram"):
                            isNewPost = validate_post(post_obj, platform, user)
                        else:
                            isNewPost = True
                        
                        if (canProceed):

                            if isNewPost in (None, True):

                                post_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)).upper()

                                if (isNewPost is None):
                                    info_noti = "Assessing most recent readable post"
                                else:
                                    info_noti = "New post detected for assessment"

                                socialmarket_logger.info(f"[{platform}] {info_noti}..")

                                info_msg = build_post_info_message(post_obj, platform, platform_name, user, post_id)

                                dedupe_key = post_assessment_key(platform, post_obj)

                                with assessed_posts_lock:
                                    if dedupe_key in assessed_posts:
                                        continue
                                    assessed_posts.add(dedupe_key)
                                    if len(assessed_posts) > MAX_ASSESSED:
                                        assessed_posts.clear()

                                discretionaryAiEnabled = check_aiDiscretionaryFlag()
                                aiEnabled = check_aiFlag()
                                emit_or_buffer_tele(info_msg, withholdTele, withholdMessages, aiEnabled, discretionaryAiEnabled)

                                # Start AI Assessment on new post
                                if (discretionaryAiEnabled or aiEnabled) and (platform != "Telegram") and (platform != "Truth Social"):

                                    ai_start = time_module.time()

                                    # Stage 1: 4O-MINI ONLY
                                    result_json_mini = parallel_stage1_4omini(post_obj, platform)
                                    socialmarket_logger.info(result_json_mini)

                                    if result_json_mini is not None:

                                        mini_flags = get_4o_mini_flags(result_json_mini)
                                        validate_mini(result_json_mini, mini_flags, platform_name)
                                        category = result_json_mini.get("Category", "CAT_2")

                                        if category != "CAT_2":

                                            stage2_category = classify_gpt_stage2_category(result_json_mini, mini_flags, platform)
                    
                                            dynamic_prompt_result = None
                                            needMarketContext = False
                            
                                            if (stage2_category in stage2_gpt4o_prompts) and (stage2_category != "EARNINGS"):
                                                sections = run_gpt_mini(post_obj, platform, "SECTIONS", "gpt-5.4-mini")
                                                validate_sections(sections, mini_flags)
                                                needMarketContext = sections["NeedMarketContext"]
                                                dynamic_prompt_result = build_dynamic_prompt(
                                                        sections,
                                                        prompts_dir=prompts_path,
                                                        logger=socialmarket_logger,
                                                        write_temp=True,
                                                    )
                                            
                                            isMarketOpen = is_market_open_sgt()

                                            isEarningsException = (stage2_category == "EARNINGS" and not isMarketOpen)
                                            isBlocked = (stage2_category == "BLOCKED") or (stage2_category == "OPTIONS") or (stage2_category == "EARNINGS" and isMarketOpen)

                                            if (aiEnabled or isEarningsException) and not isBlocked:
                                                company_mentions = result_json_mini.get("CompanyMentions", [])
                                                preferred_source, ticker_resolution_companies = preferred_ticker_resolution_companies(result_json_mini, company_mentions)
                                                if (platform == "X"):
                                                    ticker_mentions = result_json_mini.get("Tickers", [])
                                                else:
                                                    ticker_mentions = post_obj.get("tickers", [])
                                                result_queue = None
                                                resolved_ticker_details = {}

                                                if preferred_source and (len(ticker_resolution_companies) > 0):
                                                    result_queue = start_ticker_resolution(identify_tickers, (ticker_resolution_companies, platform_name), platform_name, resolved_ticker_details)

                                                # Get details base off of ticker provided first before assessing it to AI (Only for SCRAPING)
                                                elif (result_json_mini.get("Category", "CAT_2") == "CAT_1_CONTRACT") and (len(ticker_mentions) > 0):
                                                    result_queue = start_ticker_resolution(swap_tickers, (ticker_mentions,), platform_name, resolved_ticker_details)
                                                    ticker_resolution_ready = wait_for_ticker_resolution(resolved_ticker_details)

                                                    if ticker_resolution_ready:
                                                        parsing_ticker_details = "\n\n" + "\n".join(
                                                                f"${item['ticker']} Market Cap @ {item.get('marketCap', 'N.A.')}"
                                                                for item in resolved_ticker_details.values()
                                                            )

                                                        post_obj['content'] = post_obj['content'] + parsing_ticker_details

                                                        if not withholdTele:
                                                            send_telegram_message_thread(parsing_ticker_details, 0, 0)
                                                        else:
                                                            withholdMessages.append(parsing_ticker_details)
                                                    else:
                                                        socialmarket_logger.info(f"[{platform}] Market-cap resolution timed out before contract assessment; continuing without market-cap attachment.")

                                                # Concurrently get details base off of ticker provided if available
                                                elif (result_json_mini.get("Category", "CAT_2") != "CAT_1_ACQUISITION") and (len(ticker_mentions) > 0):
                                                    result_queue = start_ticker_resolution(swap_tickers, (ticker_mentions,), platform_name, resolved_ticker_details)

                                                # Fallback: Do offline scan concurrently of available tickers listed in NYSE & NASDAQ
                                                elif (len(ticker_resolution_companies) > 0):
                                                    result_queue = start_ticker_resolution(identify_tickers, (ticker_resolution_companies, platform_name), platform_name, resolved_ticker_details)
                                                
                                                # Stage 2: GPT-4O / 4O-MINI / 4.1-MINI
                                                gpt_model = assess_gpt_model(stage2_category)
                                                result_json = run_stage2_gpt(post_obj, platform, stage2_category, gpt_model, dynamic_prompt_result, needMarketContext)

                                                # Continue from result of GPT-4O / 4O-MINI / 4.1-MINI
                                                if result_json is not None:
                                                    ai_end = time_module.time()
                                                    ai_duration = ai_end - ai_start
                                                    duration_str = f"{ai_duration:.2f}s"
                                                    if (len(ticker_resolution_companies) > 0):
                                                        wait_for_ticker_resolution(resolved_ticker_details)
                                                        attempt_ticker_resolution(platform_name, result_json, result_json_mini, ticker_resolution_companies, resolved_ticker_details, gpt_model)
                                                    result_json["FGI_Value"] = get_fgi_value()
                                                    tele_msg = transformJsonToTeleMessage(platform_name, post_id, result_json, gpt_model, stage2_category, duration_str)
                                                    if not withholdTele:
                                                        send_telegram_message_thread(tele_msg, 0, 0)
                                                    else:
                                                        withholdMessages.append(tele_msg)

                                                    category = result_json.get("Category", "N.A.")

                                                    # Escalation filter validation
                                                    if (category == "CAT_1"):

                                                        tradingEnabled = (check_tradingFlag())

                                                        if tradingEnabled:

                                                            region = result_json.get("Region", "N.A.")
                                                            type_list = result_json.get("Type", [])
                                                            
                                                            if result_json_mini.get("Category", "CAT_2") == "CAT_1_ACQUISITION" and not acquisition_target_resolved(result_json_mini, type_list, resolved_ticker_details): 
                                                                if not withholdTele:
                                                                    send_telegram_message_thread("⚠️ Skipping: Incomplete ticker resolution for M&A.", 0, 0)
                                                                else:
                                                                    withholdMessages.append("⚠️ Skipping: Incomplete ticker resolution for M&A.")
                                                                socialmarket_logger.info(f"[{platform}] ⚠️ Skipping: Incomplete ticker resolution for M&A.")

                                                            elif len(type_list) > 3:
                                                                if not withholdTele:
                                                                    send_telegram_message_thread("⚠️ Skipping: Excessive ticker count.", 0, 0)
                                                                else:
                                                                    withholdMessages.append("⚠️ Skipping: Excessive ticker count.")
                                                                socialmarket_logger.info(f"[{platform}] ⚠️ Skipping: Excessive ticker count.")

                                                            else:
                                                                current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                                                order_content = f"[@{user}]\n" + post_obj['content'] + "\n------------------------------------------------" + f"\n**URL: {post_obj['url']}**" + f"\n**FGI: {result_json['FGI_Value']}**" + f"\n**TRIGGERED AT {current_timestamp}**"
                                                                if stage2_category == "EARNINGS":
                                                                    order_content = order_content + "\n**EARNINGS REQUEST**"
                                                                process_ibkr_order(type_list, region, order_content, ticker_resolution_companies, resolved_ticker_details)

                                                    elif category == "CAT_1" and "GEOPOLITICAL" in stage2_category:
                                                        if not withholdTele:
                                                            send_telegram_message_thread(f"⚠️ Skipping: Execution not permitted for {stage2_category} category.", 0, 0)
                                                        else:
                                                            withholdMessages.append(f"⚠️ Skipping: Execution not permitted for {stage2_category} category.")
                                                        socialmarket_logger.info(f"[{platform}] ⚠️ Skipping: Execution not permitted for {stage2_category} category.")

                                                    record_socialmarket_stats(process, category, result_json.get("Type", []), is_live_scraping=is_live_scraping)

                                                    # Save post into POSTS
                                                    os.makedirs(cat1_posts_path, exist_ok=True)
                        
                                                    now = datetime.now()
                                                    timestamp = now.strftime('%Y%m%d_%H%M%S') + f"{now.microsecond // 1000:03d}" 
                                                    filename = f"Post_{platform}_{user}_{timestamp}.txt"
                                                    filepath = os.path.join(cat1_posts_path, filename)

                                                    embedding = MODEL.encode(post_obj['content'], normalize_embeddings=True).tolist()

                                                    data = {
                                                        "timestamp": now.isoformat(),
                                                        "text": post_obj['content'],
                                                        "embedding": embedding,
                                                        "id": hashlib.sha256((post_obj['content']).encode('utf-8')).hexdigest()
                                                    }
                                                    
                                                    with open(filepath, 'w', encoding='utf-8') as f:
                                                            json.dump(data, f, ensure_ascii=False, indent=2)

                                                    # Send CAT 1 messages from Stock Titan if during dump period
                                                    if withholdTele and (category == "CAT_1"):
                                                        for msg in withholdMessages:
                                                            send_telegram_message_thread(msg, 0, 0)

                                            elif aiEnabled:
                                                ai_end = time_module.time()
                                                ai_duration = ai_end - ai_start
                                                duration_str = f"{ai_duration:.2f}s"
                                                result_json_mini["Category"] = "EXCLUDED"
                                                tele_msg = transformJsonToTeleMessage(platform_name, post_id, result_json_mini, "gpt-4o-mini", "N.A.", duration_str)
                                                if not withholdTele:
                                                    send_telegram_message_thread(tele_msg, 0, 0)

                                        else:
                                            ai_end = time_module.time()
                                            ai_duration = ai_end - ai_start
                                            duration_str = f"{ai_duration:.2f}s"

                                            if str(result_json_mini.get("Decision", "")).strip().upper() != "DROP":
                                                record_socialmarket_stats(process, category, None, is_live_scraping=is_live_scraping)

                                            if not aiEnabled:
                                                continue

                                            tele_msg = transformJsonToTeleMessage(platform_name, post_id, result_json_mini, "gpt-4o-mini", "N.A.", duration_str)
                                            if not withholdTele:
                                                send_telegram_message_thread(tele_msg, 0, 0)
                                        
            else:
                if (process['scrapeFailCount'] >= 12):
                    scrape_fail_msg = (
                        f"[{platform_name}]\n"
                        f"🛑 Repeated scraping failures detected for {user}. Attention required."
                    )
                    send_telegram_message_thread(scrape_fail_msg, 24, 0)

        # For Personal Testing
        elif simulation and process and process.get('display') in ["SIMULATE", "DEBUG"]:

            ai_start = time_module.time()

            result_json = None
            result_json_mini = parallel_stage1_4omini(simulation, platform)
                                    
            socialmarket_logger.info(result_json_mini)

            if result_json_mini is not None:

                mini_flags = get_4o_mini_flags(result_json_mini)
                validate_mini(result_json_mini, mini_flags, "TEST")
                category = result_json_mini.get("Category", "CAT_2")

                if category != "CAT_2":

                    stage2_category = classify_gpt_stage2_category(result_json_mini, mini_flags, platform)

                    dynamic_prompt_result = None
                    needMarketContext = False
    
                    if (stage2_category in stage2_gpt4o_prompts) and (stage2_category != "EARNINGS"):
                        sections = run_gpt_mini(simulation, platform, "SECTIONS", "gpt-5.4-mini")
                        validate_sections(sections, mini_flags)
                        needMarketContext = sections["NeedMarketContext"]
                        dynamic_prompt_result = build_dynamic_prompt(
                                sections,
                                prompts_dir=prompts_path,
                                logger=socialmarket_logger,
                                write_temp=True,
                            )

                    stage2Enabled = check_stage2Flag()

                    if (stage2Enabled):

                        company_mentions = result_json_mini.get("CompanyMentions", [])
                        preferred_source, ticker_resolution_companies = preferred_ticker_resolution_companies(result_json_mini, company_mentions)
                        result_queue = None
                        resolved_ticker_details = {}
                        
                        ticker_mentions = result_json_mini.get("Tickers", [])

                        if preferred_source and (len(ticker_resolution_companies) > 0):
                            result_queue = start_ticker_resolution(identify_tickers, (ticker_resolution_companies, "TEST"), "TEST", resolved_ticker_details)

                        # Get directly from mentioned tickers response
                        elif (len(ticker_mentions) > 0):
                            result_queue = start_ticker_resolution(swap_tickers, (ticker_mentions,), "TEST", resolved_ticker_details)

                        # Do offline scan concurrently of available tickers listed in NYSE & NASDAQ
                        elif (len(ticker_resolution_companies) > 0):
                            result_queue = start_ticker_resolution(identify_tickers, (ticker_resolution_companies, "TEST"), "TEST", resolved_ticker_details)

                        testMode = "TEST" if process.get('display') == "SIMULATE" else "DEBUG"
                        gpt_model = assess_gpt_model(stage2_category)
                        result_json = run_stage2_gpt(simulation, testMode, stage2_category, gpt_model, dynamic_prompt_result, needMarketContext)

                        if result_json is not None:
                            ai_end = time_module.time()
                            ai_duration = ai_end - ai_start
                            duration_str = f"{ai_duration:.2f}s"
                            if (len(ticker_resolution_companies) > 0):
                                wait_for_ticker_resolution(resolved_ticker_details)
                                attempt_ticker_resolution("TEST", result_json, result_json_mini, ticker_resolution_companies, resolved_ticker_details, gpt_model)
                            result_json["FGI_Value"] = get_fgi_value()
                            tele_msg = transformJsonToTeleMessage("TEST", None, result_json, gpt_model, stage2_category, duration_str)
                            send_telegram_message_thread(tele_msg, 0, 0)
                
                    else:
                        ai_end = time_module.time()
                        ai_duration = ai_end - ai_start
                        duration_str = f"{ai_duration:.2f}s"
                        tele_msg = transformJsonToTeleMessage("TEST", None, result_json_mini, "gpt-4o-mini", "N.A.", duration_str)
                        send_telegram_message_thread(tele_msg, 0, 0)
                
                else:
                    ai_end = time_module.time()
                    ai_duration = ai_end - ai_start
                    duration_str = f"{ai_duration:.2f}s"
                    tele_msg = transformJsonToTeleMessage("TEST", None, result_json_mini, "gpt-4o-mini", "N.A.", duration_str)
                    send_telegram_message_thread(tele_msg, 0, 0)

                if process.get('display') == "DEBUG" and result_json is not None:

                    thoughtProcess = result_json.get("ThoughtProcess", "N.A.")
                    time_module.sleep(1)
                    send_telegram_message_thread(thoughtProcess, 0, 0)
                
        # For Non-Regression Testing
        elif simulation and process and process.get('display') == "NR":

            result_json_mini = parallel_stage1_4omini(simulation, platform)

            socialmarket_logger.info(result_json_mini)

            if result_json_mini is not None:

                mini_flags = get_4o_mini_flags(result_json_mini)
                validate_mini(result_json_mini, mini_flags, "NR")
                category = result_json_mini.get("Category", "CAT_2")

                if category != "CAT_2":

                    stage2_category = classify_gpt_stage2_category(result_json_mini, mini_flags, platform)
                    
                    dynamic_prompt_result = None
                    needMarketContext = False
    
                    if (stage2_category in stage2_gpt4o_prompts) and (stage2_category != "EARNINGS"):
                        sections = run_gpt_mini(simulation, platform, "SECTIONS", "gpt-5.4-mini")
                        validate_sections(sections, mini_flags)
                        needMarketContext = sections["NeedMarketContext"]
                        dynamic_prompt_result = build_dynamic_prompt(
                                sections,
                                prompts_dir=prompts_path,
                                logger=socialmarket_logger,
                                write_temp=True,
                            )

                    company_mentions = result_json_mini.get("CompanyMentions", [])
                    preferred_source, ticker_resolution_companies = preferred_ticker_resolution_companies(result_json_mini, company_mentions)
                    result_queue = None
                    resolved_ticker_details = {}
                    
                    ticker_mentions = result_json_mini.get("Tickers", [])

                    if preferred_source and (len(ticker_resolution_companies) > 0):
                        result_queue = start_ticker_resolution(identify_tickers, (ticker_resolution_companies, "NR"), "NR", resolved_ticker_details)

                    # Get directly from mentioned tickers response
                    elif (len(ticker_mentions) > 0):
                        result_queue = start_ticker_resolution(swap_tickers, (ticker_mentions,), "NR", resolved_ticker_details)

                    # Do offline scan concurrently of available tickers listed in NYSE & NASDAQ
                    elif (len(ticker_resolution_companies) > 0):
                        result_queue = start_ticker_resolution(identify_tickers, (ticker_resolution_companies, "NR"), "NR", resolved_ticker_details)

                    gpt_model = assess_gpt_model(stage2_category)
                    result_json = run_stage2_gpt(simulation, "NR", stage2_category, gpt_model, dynamic_prompt_result, needMarketContext)

                    if result_json is not None:
                        if (len(ticker_resolution_companies) > 0):
                            wait_for_ticker_resolution(resolved_ticker_details)
                            attempt_ticker_resolution("NR", result_json, result_json_mini, ticker_resolution_companies, resolved_ticker_details, gpt_model)
                        return result_json, stage2_category
                    else:
                        return None, None
                    
                else:
                    return result_json_mini, "N.A."

    except Exception as e:
        error_msg = f"[{platform}] Error processing {platform} - {user}:\n{str(e)}"
        socialmarket_logger.info(error_msg)
        socialmarket_logger.info(traceback.format_exc())
        send_telegram_message_thread(error_msg, 0, 0)
        timePrint(error_msg)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# CONTROL FILE / TEST RUN HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_scrapingFlag():
    xander_ScrapingFlag_fileName = "XANDER_ScrapingFlag.txt"
    file_path_xander_ScrapingFlag = os.path.join(functionsFolder_path, xander_ScrapingFlag_fileName)
    if os.path.isfile(file_path_xander_ScrapingFlag):
        with open(file_path_xander_ScrapingFlag, 'r') as f:
            content = f.read().strip()
        if (content == "0"):
            return False
        else:
            return True
    else:
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_aiFlag():
    xander_AiFlag_fileName = "XANDER_AiFlag.txt"
    file_path_xander_AiFlag = os.path.join(functionsFolder_path, xander_AiFlag_fileName)
    if os.path.isfile(file_path_xander_AiFlag):
        with open(file_path_xander_AiFlag, 'r') as f:
            content = f.read().strip()
        if (content == "0"):
            return False
        else:
            return True
    else:
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_aiDiscretionaryFlag():
    xander_DiscretionaryAiFlag_fileName = "XANDER_DiscretionaryAiFlag.txt"
    file_path_xander_DiscretionaryAiFlag = os.path.join(functionsFolder_path, xander_DiscretionaryAiFlag_fileName)
    if os.path.isfile(file_path_xander_DiscretionaryAiFlag):
        with open(file_path_xander_DiscretionaryAiFlag, 'r') as f:
            content = f.read().strip()
        if (content == "0"):
            return False
        else:
            return True
    else:
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_stage2Flag():
    xander_Stage2Flag_fileName = "XANDER_Stage2Flag.txt"
    file_path_xander_Stage2Flag = os.path.join(functionsFolder_path, xander_Stage2Flag_fileName)
    if os.path.isfile(file_path_xander_Stage2Flag):
        with open(file_path_xander_Stage2Flag, 'r') as f:
            content = f.read().strip()
        if (content == "0"):
            return False
        else:
            return True
    else:
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_tradingFlag():
    xander_TradingFlag_fileName = "XANDER_TradingFlag.txt"
    file_path_xander_TradingFlag = os.path.join(functionsFolder_path, xander_TradingFlag_fileName)
    if os.path.isfile(file_path_xander_TradingFlag):
        with open(file_path_xander_TradingFlag, 'r') as f:
            content = f.read().strip()
        if (content == "0"):
            return False
        else:
            return True
    else:
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def initiate_simulate(file_content, simulate_process):

    lines = file_content.strip().splitlines()

    if lines:
        first_line = lines[0].strip()
        # Match [@username] and keep the @
        match = re.match(r'\[\s*(\@[^\s\[\]]+)\s*\]', first_line)
        if match:
            user = match.group(1)
            targetted_process = next((p for p in processes if p["user"].lower() == user.replace("@","").lower()), None)
            if (targetted_process):
                simulate_process['user'] = targetted_process['user']
                simulate_process['type'] = targetted_process['type']
                simulate_process['platform'] = targetted_process['platform']
                file_content = "\n".join(lines[1:]).strip()

    fetch_and_process_post(simulate_process, None, None, "TEST", "TEST", file_content)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def initiate_nr(nr_process):

    notify_start_msg = "Non-regression run initiated. This may take some time to complete."
    send_telegram_message_thread(notify_start_msg, 0, 0)

    txt_files = [f for f in os.listdir(nrFolder_path) if f.endswith(".txt")]
    total = len(txt_files)
    passed = 0
    failed = 0
    manual_review = 0

    for filename in txt_files:

        # Example: @Username_CAT1_BULLISH_ABCDEF.txt
        parts = filename.replace(".txt", "").split("_")
        category = parts[0]        # CAT1 or CAT2
        if category == "CAT1":
            sentiment = parts[1]       # BULLISH / BEARISH / NA (only for CAT1)
            asset = parts[2]         # SECTOR / GENERAL / TICKER (only for CAT1)
            gpt4oSubCat = parts[3] 
        else:
            gpt4oSubCat = parts[1] 

        file_path = os.path.join(nrFolder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        nr_process['user'] = "Non-Regression Run"
        nr_process['type'] = "X"
        nr_process['platform'] = "X"

        result_json, gpt4oSubCat_result = fetch_and_process_post(nr_process, None, None, "TEST", "TEST", content)

        if (result_json is None) or (gpt4oSubCat_result is None):
            failed += 1
            print("ERROR PROCESSING BELOW CONTENT")
            print(content)

        else:
            cat_result = result_json.get("Category", "N.A.")

            def save_review_files(result_tag):
                base_filename = filename.replace(".txt", "")
                original_path = os.path.join(nrReviewFolder_path, f"{base_filename}_{result_tag}_ORIGINAL.txt")
                new_path = os.path.join(nrReviewFolder_path, f"{base_filename}_{result_tag}_{gpt4oSubCat_result.replace('.', '').replace('_', '').upper().strip()}_NEW.txt")

                shutil.copy(file_path, original_path)
                with open(new_path, "w", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(result_json, indent=2))

            if category == "CAT2" and cat_result == "CAT_2":
                passed += 1

            elif category == "CAT2" and cat_result != "CAT_2":
                failed += 1
                save_review_files("FAILED")

            elif category == "CAT1" and cat_result == "CAT_1":

                types_result = result_json.get("Type", [])

                if sentiment == "MANUAL" or asset == "MANUAL":
                    manual_review += 1
                    save_review_files("MANUAL")

                elif len(types_result) > 1: # Save for review for those that has more than 1
                    manual_review += 1
                    save_review_files("MANUAL")

                elif len(types_result) == 1:
                    
                    validationFailed = False

                    sentiment_result = types_result[0].get("Sentiment", "N.A.")
                    asset_result = types_result[0].get("Asset", "N.A.")

                    if sentiment == "BULLISH" and sentiment_result == "Bullish":
                        if asset.lower() == asset_result.replace(".", "").lower().strip():
                            passed += 1
                        else:
                            validationFailed = True
                    elif sentiment == "BEARISH" and sentiment_result == "Bearish":
                        if asset.lower() == asset_result.replace(".", "").lower().strip():
                            passed += 1
                        else:
                            validationFailed = True
                    else:
                        validationFailed = True
                    
                    if validationFailed:
                        failed += 1
                        save_review_files("FAILED")

                else:
                    failed += 1
                    save_review_files("FAILED")

            elif category == "CAT1" and cat_result != "CAT_1":
                failed += 1
                save_review_files("FAILED")
        
    summary_msg = (
        f"📊 Non-Regression Completed\n"
        f"Total Test Cases: {total}\n"
        f"✅ Passed: {passed}\n"
        f"❌ Failed: {failed}\n"
        f"📝 Manual Review: {manual_review}"
    )
    send_telegram_message_thread(summary_msg, 0, 0)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# THREAD / WORKER / MAIN LOOP HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fileCheck():
    
    for file_name in os.listdir(functionsFolder_path):

        file_path = os.path.join(functionsFolder_path, file_name)
        if (file_name == "STATUSCHECK_SocialMarket.txt"):
            os.remove(file_path)
        if (file_name == "SIMULATION_SocialMarket.txt"):
            with open(file_path, 'r', encoding='latin1') as f:
                file_content = f.read()
            if file_content.startswith('DEBUG:'):
                file_content = file_content[len('DEBUG:'):].lstrip()
                simulate_process = {"user" : "SIMULATE", "display" : "DEBUG", "type" : "TEST", "platform" : "TEST", "scrapeFailCount" : 0}
            else:
                simulate_process = {"user" : "SIMULATE", "display" : "SIMULATE", "type" : "TEST", "platform" : "TEST", "scrapeFailCount" : 0}
            initiate_simulate(file_content, simulate_process)
            os.remove(file_path)
        if (file_name == "NON-REGRESSION_SocialMarket.txt"):
            try:
                nr_process = {"user" : "NR", "display" : "NR", "type" : "TEST", "platform" : "TEST", "scrapeFailCount" : 0}
                initiate_nr(nr_process)
            finally:
                os.remove(file_path)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fileCheck_loop():
    while True:
        try:
            fileCheck()
        except Exception as loop_error:
            loop_error = f"Loop error: {loop_error}"
            timePrint(loop_error)
            timePrint(traceback.format_exc())
        time_module.sleep(1)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
async def start_telegram_listener():
    await telegram_client.start()
    source = await telegram_client.get_entity(TELEGRAM_SOURCE)

    @telegram_client.on(events.NewMessage(chats=source))
    async def handler(event):
        msg = event.message
        if not msg or not msg.text:
            return
        
        post_obj = {
                    "idx": 0,
                    "content": msg.text.replace("(@WalterBloomberg)","").strip(),
                    "url": None,
                    "time": parse_and_adjust_datetime(msg.date.strftime("%B %d, %Y, %I:%M %p"), "%B %d, %Y, %I:%M %p", 8, "Telegram"),
                    "isOwner": True, 
                    "hasAttachments": False,
                    "valid": True
                }

        # Inject into existing pipeline
        fetch_and_process_post(
            process={"user" : "DeItaone" , "display" : "*Walter Bloomberg" , "type" : "Telegram" , "platform" : "Telegram" , "intervals" : [], "instance" : [] , "scrapeFailCount" : 0},
            user="DeItaone",
            display_name="*Walter Bloomberg",
            platform="Telegram",
            platform_name="Telegram",
            simulation=None,
            tele_obj=post_obj
        )

    await telegram_client.run_until_disconnected()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def start_telegram_thread():
    asyncio.run(start_telegram_listener())
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_scraping_job():
    while True:
        current_time = time_module.localtime()
        current_seconds = current_time.tm_sec

        for process in processes:
            if current_seconds in process.get("intervals", []):
                user = process['user']
                display_name = process['display']
                platform = process['type']
                platform_name = process['platform']
                
                scrapingEnabled = check_scrapingFlag()

                if scrapingEnabled:
                    threading.Thread(
                        target=fetch_and_process_post,
                        args=(process, user, display_name, platform, platform_name, None),
                        daemon=True  # Optional: ensures threads auto-clean up if main thread exits
                    ).start()

        time_module.sleep(1)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# STARTUP / ENTRYPOINT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Initialize and log start
initialized_info = "SocialMarket.py initialized..."
socialmarket_logger.info(initialized_info)
print(initialized_info)

# Start Telegram worker in its own thread
start_telegram_workers()
threading.Thread(target=telegram_health_monitor, daemon=True).start()

# Start stats writer in its own thread
threading.Thread(target=socialmarket_stats_worker, daemon=True).start()

# Start fileCheck in its own thread
status_thread = threading.Thread(target=fileCheck_loop, daemon=True)
status_thread.start()

# Start the main scheduled job loop (this blocks the main thread)
#threading.Thread(target=start_telegram_thread, daemon=True).start()
run_scraping_job()
