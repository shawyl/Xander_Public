"""Launch and control entrypoint for the Xander runtime.

Coordinates scheduled checks, Telegram command handling, process supervision,
and Lucian startup. Trading, AI assessment, and broker behavior are delegated
to their own modules.

AI status: Maintained with AI.
"""

import pandas_market_calendars as mcal
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import telepot
import datetime as dt
import time as pytime
import os, json
import subprocess
import threading
import pytz
import requests
import http.client
import urllib3
import socket
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import tempfile
import queue
import traceback
import sys
import logging
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent
LUCIAN_DIR = RUNTIME_ROOT / "Lucian"
if str(LUCIAN_DIR) not in sys.path:
    sys.path.insert(0, str(LUCIAN_DIR))

import lucian_balance

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# CONFIGURATION / ENVIRONMENT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Load environment variables from a .env file (if present)
load_dotenv(RUNTIME_ROOT / ".env")

def get_optional_env_int(name):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc

def get_env_csv(name):
    value = os.getenv(name, "")
    return [part.strip() for part in value.split(",") if part.strip()]

ALLOWED_USERS = get_env_csv("ALLOWED_TELEGRAM_USERS")
if not ALLOWED_USERS:
    print("Warning: ALLOWED_TELEGRAM_USERS is not configured; Telegram commands will not authorize any users.")
TELEGRAM_COMMAND_QUEUE_SIZE = int(os.getenv("XANDER_TELEGRAM_COMMAND_QUEUE_SIZE", "100"))
TELEGRAM_COMMAND_WORKERS = int(os.getenv("XANDER_TELEGRAM_COMMAND_WORKERS", "3"))
TELEGRAM_COMMAND_WARN_SECONDS = int(os.getenv("XANDER_TELEGRAM_COMMAND_WARN_SECONDS", "15"))
TELEGRAM_LISTENER_HEARTBEAT_SECONDS = int(os.getenv("XANDER_TELEGRAM_LISTENER_HEARTBEAT_SECONDS", "300"))
PYTHON_EXE = os.getenv("XANDER_PYTHON_EXE", sys.executable)
telegram_command_queue = queue.Queue(maxsize=TELEGRAM_COMMAND_QUEUE_SIZE)
telegram_worker_last_seen = {}
telegram_worker_last_seen_lock = threading.Lock()

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

# Define the API / BOT TOKEN to connect
BOT_TOKEN = require_env("XANDER_TELEGRAM_BOT_TOKEN")
CHAT_ID = require_int_env("XANDER_TELEGRAM_CHAT_ID")

BOTHUB_ROOT = require_env("BOTHUB_ROOT")

def bot_path(*parts):
    return os.path.join(BOTHUB_ROOT, *parts)

def get_env_hhmm(name):
    value = require_env(name).strip()
    try:
        datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be HH:MM, got {value!r}") from exc
    return value

def get_env_time(name):
    return datetime.strptime(get_env_hhmm(name), "%H:%M").time()

def get_env_hhmm_list(name):
    value = require_env(name)
    parts = [part.strip() for part in value.split(",") if part.strip()]
    for part in parts:
        try:
            datetime.strptime(part, "%H:%M")
        except ValueError as exc:
            raise RuntimeError(f"Environment variable {name} must be a comma-separated HH:MM list, got {value!r}") from exc
    return parts

def get_env_int_list(name):
    value = require_env(name)
    parts = [part.strip() for part in value.split(",") if part.strip()]
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a comma-separated integer list, got {value!r}") from exc

def get_env_int(name):
    value = require_env(name).strip()
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer, got {value!r}") from exc

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# PATHS / GLOBAL RUNTIME STATE
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Folder directories
fnFolder_path = bot_path(".BOT_Launch", "Functions")
posts_path = bot_path(".BOT_Launch", "SocialMarket", "POSTS")
xanderLogs_path = bot_path("_OUTPUT", "XANDER", "LOGS")
ibkrLogs_path = bot_path("_OUTPUT", "ISAIAH", "LOGS")
lucianLogs_path = os.getenv("LUCIAN_LOG_DIR", bot_path("_OUTPUT", "LUCIAN", "LOGS"))
ibkrError_path = bot_path(".BOT_Launch", "IBKR", "ERROR")
ibkrIgnored_path = bot_path(".BOT_Launch", "IBKR", "IGNORED")
ibkrClosed_path = bot_path(".BOT_Launch", "IBKR", "CLOSED")
dynamicPrompts_path = bot_path(".BOT_Launch", "SocialMarket", "PROMPTS", "TEMP_DYNAMIC")
sgd_usd_rate_cache_path = bot_path(".BOT_Launch", "SocialMarket", "FX", "sgd_usd_rate.txt")
botsService_name = ["SocialMarket.py", "IBKR.py"]
botsFolder_path = [
    bot_path(".BOT_Launch", "SocialMarket", "SocialMarket.py"),
    bot_path(".BOT_Launch", "IBKR", "IBKR.py")
]
LUCIAN_THREAD_ID = get_optional_env_int("LUCIAN_TELEGRAM_THREAD_ID")
if LUCIAN_THREAD_ID is None:
    print("Warning: LUCIAN_TELEGRAM_THREAD_ID is not configured; Lucian topic-specific routing is disabled.")
LUCIAN_TELEGRAM_SEND_TIMEOUT_SECONDS = int(os.getenv("LUCIAN_TELEGRAM_SEND_TIMEOUT_SECONDS", "10"))
LUCIAN_IGNORED_LOG_INTERVAL_SECONDS = int(os.getenv("LUCIAN_IGNORED_LOG_INTERVAL_SECONDS", "300"))
LUCIAN_WEEKLY_SUMMARY_ENABLED = os.getenv("LUCIAN_WEEKLY_SUMMARY_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
LUCIAN_WEEKLY_SUMMARY_SGT = os.getenv("LUCIAN_WEEKLY_SUMMARY_SGT", "09:00")
LUCIAN_IBKR_BALANCE_TIMEOUT_SECONDS = int(os.getenv("LUCIAN_IBKR_BALANCE_TIMEOUT_SECONDS", "20"))
LUCIAN_TRADE_REVIEW_CONTEXT_TTL_SECONDS = int(os.getenv("LUCIAN_TRADE_REVIEW_CONTEXT_TTL_SECONDS", "1800"))
LUCIAN_ALLOWED_USERS = [
    user.strip()
    for user in os.getenv("LUCIAN_ALLOWED_USERS", ",".join(ALLOWED_USERS)).split(",")
    if user.strip()
]
lucian_reviewer = None
lucian_startup_error = None
lucian_token_warning_logged = False
lucian_ignored_log_state = {}
lucian_pending_trade_reviews = {}
lucian_pending_trade_reviews_lock = threading.Lock()

GATEWAY_IBKR_CANCEL_ORDERS = "GATEWAY_CANCEL_ORDERS_IBKR.txt"
GATEWAY_IBKR_GET_ORDERS = "GATEWAY_GET_PAST_ORDERS_IBKR.txt"
GATEWAY_IBKR_GET_PNL = "GATEWAY_GET_UNREALIZED_PNL_IBKR.txt"
GATEWAY_IBKR_CREATE_SELLS = "GATEWAY_CREATE_SELLS_IBKR.txt"
GATEWAY_IBKR_REPLACE_SELLS = "GATEWAY_REPLACE_SELLS_IBKR.txt"
GATEWAY_IBKR_FETCH_SPREADS = "GATEWAY_FETCH_SPREADS_IBKR.txt"
GATEWAY_IBKR_GET_BALANCE = "GATEWAY_GET_BALANCE_IBKR.txt"
GATEWAY_IBKR_REFRESH_BALANCE = "GATEWAY_REFRESH_BALANCE_IBKR.txt"
GATEWAY_IBKR_REFRESH_AVAIL_BALANCE = "GATEWAY_REFRESH_AVAIL_BALANCE_IBKR.txt"
GATEWAY_IBKR_DISCONNECT = "GATEWAY_DISCONNECT_IBKR.txt"
GATEWAY_IBKR_STATUS = "GATEWAY_IBKR.txt"
STATUS_IBKR_AUTOMATE = "STATUSCHECK_IBKR.txt"
STATUS_SOCIALMARKET = "STATUSCHECK_SocialMarket.txt"
MAINTENANCE_IBKR = "MAINTENANCE_IBKR.txt"
XANDER_HEALTH = "XANDER_Health.txt"
XANDER_MUTED_PERIOD = "XANDER_MutedPeriod.txt"

# Simulate controls
simulation_flag = 0
simulation_user = None

# Update market context controls
updateMarketContext_flag = 0
updateMarketContext_user = None

# Update benzinga API
updateBenzinga_flag = 0
updateBenzinga_user = None

# IBKR Instant Sell
ibkrInstantSells_flag = 0
ibkrInstantSells_user = None

# Replace IBKT Limit Sell
replaceIbkrSells_flag = 0
replaceIbkrSells_user = None

# Fetch Bid Ask Price
fetchSpread_flag = 0
fetchSpread_user = None

# Mute alert controls
muteAlerts_flag = 0
muteAlerts_user = None

# Query Trades controls
queryTrades_flag = 0
queryTrades_user = None
queryTrades_option = None

# Startup control
isStartup = 1

# Global Times (No need for market hours)
tradingOpen = get_env_hhmm("XANDER_TRADING_OPEN_SGT")
tradingClose = get_env_hhmm("XANDER_TRADING_CLOSE_SGT")
marketOpen = get_env_hhmm("XANDER_MARKET_OPEN_SGT")
marketClose = get_env_hhmm("XANDER_MARKET_CLOSE_SGT")
marketHalfDay = get_env_hhmm("XANDER_MARKET_HALF_DAY_CLOSE_SGT")
checkTradingDay = get_env_hhmm("XANDER_CHECK_TRADING_DAY_SGT")
ibStart = get_env_hhmm("XANDER_IB_GATEWAY_START_SGT")
ibStartBuffer = get_env_hhmm_list("XANDER_IB_GATEWAY_START_BUFFER_SGT")
xanderRestart = get_env_hhmm("XANDER_DAILY_RESTART_SGT")
pruneFired = get_env_hhmm("XANDER_PRUNE_FIRED_SGT")
weekendMuteAfterSat = get_env_time("XANDER_WEEKEND_MUTE_AFTER_SAT_SGT")
weekendMuteUntilMon = get_env_time("XANDER_WEEKEND_MUTE_UNTIL_MON_SGT")
marketContextCheckHours = get_env_int_list("XANDER_MARKET_CONTEXT_CHECK_HOURS_SGT")
tickerRefreshStartHour = get_env_int("XANDER_TICKER_REFRESH_START_HOUR_SGT")
tickerRefreshEndHour = get_env_int("XANDER_TICKER_REFRESH_END_HOUR_SGT")
usRegularOpen = get_env_hhmm("XANDER_US_REGULAR_OPEN_ET")
usRegularClose = get_env_hhmm("XANDER_US_REGULAR_CLOSE_ET")

# Timezone
XANDER_TIMEZONE = require_env("XANDER_TIMEZONE")
XANDER_MARKET_TIMEZONE = require_env("XANDER_MARKET_TIMEZONE")
sgt = pytz.timezone(XANDER_TIMEZONE)
SGT = ZoneInfo(XANDER_TIMEZONE)

# Schedule Jobs Control
FIRED_FILE = os.path.join(fnFolder_path, "scheduler_fired.json")
_fired = set()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# LOGGING HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def timePrint(log):
    sgt_now = datetime.now(sgt)
    formatted_time = sgt_now.strftime('%d-%b-%y %I:%M:%S %p')  # e.g., 10-May-25 01:22:43 AM
    print(f"[{formatted_time}] {log}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def setup_lucian_logger():
    os.makedirs(lucianLogs_path, exist_ok=True)
    log_level_name = os.getenv("LUCIAN_LOG_LEVEL", "INFO").strip().upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger = logging.getLogger("lucian_logger")
    logger.setLevel(log_level)
    logger.propagate = False

    if not logger.handlers:
        current_log_date = datetime.now().strftime('%d-%m-%Y')
        log_filepath = os.path.join(lucianLogs_path, f"Lucian-{current_log_date}.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
lucian_logger = setup_lucian_logger()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# FILE / SCHEDULER HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def atomic_write_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=os.path.basename(path) + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def read_text_default(path, default=""):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return default
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def write_text(path, text=""):
    atomic_write_text(path, text)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def gateway_path(file_name):
    return os.path.join(fnFolder_path, file_name)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def write_gateway_file(file_name, content=""):
    write_text(gateway_path(file_name), content)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def write_flag_file(file_name, value):
    write_gateway_file(file_name, value)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def load_fired():
    global _fired
    if not os.path.exists(FIRED_FILE):
        _fired = set()
        return
    try:
        with open(FIRED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        _fired = {
            (x[0], x[1])
            for x in data
            if isinstance(x, list) and len(x) == 2
        }
    except Exception:
        _fired = set()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def save_fired():
    # Store as list of [event, date] for JSON friendliness
    data = [[k[0], k[1]] for k in sorted(_fired)]
    with open(FIRED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def prune_fired(keep_days: int = 3):
    global _fired
    dates = sorted({d for (_, d) in _fired})
    if len(dates) <= keep_days:
        return
    keep = set(dates[-keep_days:])
    _fired = {k for k in _fired if k[1] in keep}
    save_fired()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def hhmm_to_time(hhmm: str) -> time:
    h, m = map(int, hhmm.split(":"))
    return time(hour=h, minute=m)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_once_per_day_when_due(event_name: str, now: datetime, due_hhmm: str):
    # Fires exactly once per SGT calendar day when now >= due_hhmm. Persists fired state locally so restarts do NOT re-fire.
    due_t = hhmm_to_time(due_hhmm)
    if now.time() < due_t:
        return False

    key = (event_name, now.date().isoformat())
    if key in _fired:
        return False

    _fired.add(key)
    save_fired()
    return True
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_once_per_weekend_when_due(event_name: str, now: datetime, due_hhmm: str):
    # Fires once per SGT weekend when now >= due_hhmm. If Saturday is missed, Sunday can still send once.
    if now.weekday() not in (5, 6):
        return False

    due_t = hhmm_to_time(due_hhmm)
    if now.time() < due_t:
        return False

    weekend_saturday = now.date() if now.weekday() == 5 else now.date() - timedelta(days=1)
    key = (event_name, weekend_saturday.isoformat())
    if key in _fired:
        return False

    _fired.add(key)
    save_fired()
    return True
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# LUCIAN INTEGRATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def weekend_saturday_date(now: datetime):
    return now.date() if now.weekday() == 5 else now.date() - timedelta(days=1)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def initialize_lucian():
    global lucian_reviewer
    global lucian_startup_error

    try:
        import lucian_reviewer as reviewer

        lucian_reviewer = reviewer
        lucian_startup_error = None
        lucian_logger.info("Lucian initialized")
        return True
    except Exception as exc:
        lucian_reviewer = None
        lucian_startup_error = exc
        lucian_logger.exception("Lucian startup failed: %s - %s", type(exc).__name__, exc)
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_lucian_bot_token():
    token = os.getenv("LUCIAN_TELEGRAM_BOT_TOKEN", "").strip()
    if token:
        return token, "LUCIAN_TELEGRAM_BOT_TOKEN", False

    token = os.getenv("LUCIAN_BOT_TOKEN", "").strip()
    if token:
        return token, "LUCIAN_BOT_TOKEN", False

    return BOT_TOKEN, "XANDER_TELEGRAM_BOT_TOKEN", True
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def should_log_lucian_route_event(reason):
    now = pytime.monotonic()
    last_logged = lucian_ignored_log_state.get(reason, 0)
    if now - last_logged >= LUCIAN_IGNORED_LOG_INTERVAL_SECONDS:
        lucian_ignored_log_state[reason] = now
        return True
    return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def log_lucian_route_event(reason, detail):
    if should_log_lucian_route_event(reason):
        lucian_logger.debug("Lucian message ignored | reason=%s | %s", reason, detail)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def lucian_pending_key(message, username):
    chat = message.get("chat") or {}
    from_user = message.get("from") or {}
    user_key = from_user.get("id") or username
    return (chat.get("id"), message.get("message_thread_id"), user_key)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def purge_expired_lucian_trade_reviews():
    now = pytime.time()
    expired = []
    with lucian_pending_trade_reviews_lock:
        for key, payload in list(lucian_pending_trade_reviews.items()):
            if payload.get("expires_at", 0) <= now:
                expired.append(key)
                lucian_pending_trade_reviews.pop(key, None)
    for key in expired:
        lucian_logger.debug("Lucian pending trade review context expired | key=%s", key)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def store_lucian_trade_review_context(key, context):
    if not context:
        return False
    purge_expired_lucian_trade_reviews()
    payload = {
        "context": context,
        "created_at": pytime.time(),
        "expires_at": pytime.time() + LUCIAN_TRADE_REVIEW_CONTEXT_TTL_SECONDS,
    }
    with lucian_pending_trade_reviews_lock:
        lucian_pending_trade_reviews[key] = payload
    lucian_logger.debug(
        "Lucian pending trade review context stored | key=%s | trades=%s | tickers=%s",
        key,
        context.get('trade_count'),
        context.get('tickers'),
    )
    return True
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def pop_lucian_trade_review_context(key):
    purge_expired_lucian_trade_reviews()
    with lucian_pending_trade_reviews_lock:
        payload = lucian_pending_trade_reviews.pop(key, None)
    if not payload:
        return None
    lucian_logger.info("Lucian confirmation matched pending trade review context | key=%s", key)
    return payload.get("context")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def peek_lucian_trade_review_context(key):
    purge_expired_lucian_trade_reviews()
    with lucian_pending_trade_reviews_lock:
        payload = lucian_pending_trade_reviews.get(key)
    return payload.get("context") if payload else None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def lucian_trade_review_offer_text(context):
    trade_count = context.get("trade_count", 0) if context else 0
    if trade_count <= 0:
        return ""
    return "Want me to review how this trade behaved after entry and exit?" if trade_count == 1 else "Want me to review how these trades behaved after entry and exit?"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def send_lucian_message(_bot, text):
    global lucian_token_warning_logged

    if lucian_reviewer is not None and not text:
        text = lucian_reviewer.FALLBACK_UNKNOWN_RESPONSE
    elif not text:
        text = "Sorry, I don't understand that request."

    chunks = [text[i:i + 3500] for i in range(0, len(text), 3500)] or [text]

    lucian_bot_token, token_source, using_xander_token = get_lucian_bot_token()
    if using_xander_token and not lucian_token_warning_logged:
        lucian_logger.warning("Lucian Telegram bot token not configured; falling back to XANDER_TELEGRAM_BOT_TOKEN, so replies will appear from Xander.")
        lucian_token_warning_logged = True

    try:
        for chunk in chunks:
            response = requests.post(
                f"https://api.telegram.org/bot{lucian_bot_token}/sendMessage",
                json={
                    "chat_id": CHAT_ID,
                    "text": chunk,
                    "message_thread_id": LUCIAN_THREAD_ID,
                },
                timeout=LUCIAN_TELEGRAM_SEND_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response_json = response.json()
            if not response_json.get("ok"):
                raise RuntimeError(response_json.get("description", "Telegram sendMessage returned ok=false"))
    except Exception as exc:
        lucian_logger.exception("Lucian Telegram reply failed: %s - %s", type(exc).__name__, exc)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def process_lucian_request(bot, request, username, pending_key):
    try:
        response = lucian_reviewer.answer_lucian_request(request, logger=lucian_logger)
        review_context = lucian_reviewer.build_trade_review_context_for_request(request)
        if review_context and store_lucian_trade_review_context(pending_key, review_context):
            lucian_logger.info("Lucian trade-review offer created | user=%s | trades=%s", username, review_context.get('trade_count'))
            response = f"{response}\n\n{lucian_trade_review_offer_text(review_context)}"
    except Exception as exc:
        lucian_logger.exception("Lucian request failed for %s: %s - %s", username, type(exc).__name__, exc)
        response = "⚠️ Lucian hit an internal review error. The logs have the details."

    send_lucian_message(bot, response)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def process_lucian_deep_trade_review(bot, message_text, username, review_context=None):
    try:
        if review_context is None:
            review_context, context_error = lucian_reviewer.build_trade_review_context_from_text(message_text)
            if context_error:
                lucian_logger.warning("Lucian explicit trade review context issue | user=%s | detail=%s", username, context_error)
                send_lucian_message(bot, context_error)
                return
        if not review_context:
            send_lucian_message(bot, "⚠️ I can review it, but I need the trade window again. Try: \"Lucian, review yesterday's trades.\"")
            return

        lucian_logger.info(
            "Lucian explicit trade review request detected | user=%s | trades=%s | tickers=%s",
            username,
            review_context.get('trade_count'),
            review_context.get('tickers'),
        )
        send_lucian_message(bot, "Got it. I'll review the trade behavior using IBKR historical data for that window.")
        response = lucian_reviewer.answer_deep_trade_review(review_context, logger=lucian_logger)
        send_lucian_message(bot, response)
        lucian_logger.info("Lucian deep review completed | user=%s | tickers=%s", username, review_context.get('tickers'))
    except Exception as exc:
        lucian_logger.exception("Lucian deep trade review failed for %s: %s - %s", username, type(exc).__name__, exc)
        send_lucian_message(bot, "⚠️ Lucian hit an internal trade-review error. The logs have the details.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def process_unclear_lucian_request(bot, message_text, username):
    try:
        response = lucian_reviewer.answer_unclear_lucian_query(message_text, logger=lucian_logger)
        lucian_logger.info("Lucian clarification sent | user=%s", username)
    except Exception as exc:
        lucian_logger.exception("Lucian clarification failed for %s: %s - %s", username, type(exc).__name__, exc)
        response = lucian_reviewer.FALLBACK_UNKNOWN_RESPONSE

    send_lucian_message(bot, response)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def handle_lucian_message(bot, message):
    if lucian_reviewer is None:
        if lucian_startup_error is not None:
            lucian_logger.error("Lucian unavailable: %s - %s", type(lucian_startup_error).__name__, lucian_startup_error)
        return

    message_text = message.get("text") or ""
    from_user = message.get("from") or {}
    username = from_user.get("username") or from_user.get("first_name") or "Unknown"
    if LUCIAN_ALLOWED_USERS and username not in LUCIAN_ALLOWED_USERS:
        lucian_logger.warning("Unauthorized Lucian request from %s", username)
        return

    pending_key = lucian_pending_key(message, username)
    addressed_to_lucian = lucian_reviewer.extract_lucian_query(message_text) is not None

    if lucian_reviewer.is_trade_review_confirmation(message_text):
        review_context = pop_lucian_trade_review_context(pending_key)
        if review_context:
            lucian_logger.info("Lucian trade-review confirmation accepted | user=%s", username)
            threading.Thread(
                target=process_lucian_deep_trade_review,
                args=(bot, message_text, username, review_context),
                daemon=True,
            ).start()
            return
        send_lucian_message(bot, "⚠️ I can review it, but I need the trade window again. Try: \"Lucian, review yesterday's trades.\"")
        return

    if not addressed_to_lucian:
        log_lucian_route_event("not_addressed", f"thread_id={message.get('message_thread_id')}")
        return

    if lucian_reviewer.is_trade_deep_review_request(message_text):
        if lucian_reviewer.trade_review_request_needs_pending(message_text):
            review_context = pop_lucian_trade_review_context(pending_key)
            if not review_context:
                send_lucian_message(bot, "⚠️ I can review it, but I need the trade window again. Try: \"Lucian, review yesterday's trades.\"")
                return
            args = (bot, message_text, username, review_context)
        else:
            args = (bot, message_text, username)
        threading.Thread(
            target=process_lucian_deep_trade_review,
            args=args,
            daemon=True,
        ).start()
        return

    request = lucian_reviewer.parse_lucian_request(message_text)
    if request is None:
        lucian_logger.info("Lucian unclear request accepted for clarification | user=%s", username)
        threading.Thread(
            target=process_unclear_lucian_request,
            args=(bot, message_text, username),
            daemon=True,
        ).start()
        return

    lucian_logger.info("Lucian request accepted | user=%s | intents=%s", username, ",".join(request.intents))
    threading.Thread(
        target=process_lucian_request,
        args=(bot, request, username, pending_key),
        daemon=True,
    ).start()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def send_lucian_weekly_update(bot):
    if lucian_reviewer is None:
        if not initialize_lucian():
            return

    query = "Lucian, give me the weekly summary"
    request = lucian_reviewer.parse_lucian_request(query, now=datetime.now(sgt))
    if request is None:
        lucian_logger.warning("Lucian weekly update skipped: weekly summary request could not be parsed")
        return

    try:
        lucian_logger.info("Lucian weekly update started")
        response = lucian_reviewer.answer_lucian_request(request, logger=lucian_logger)
    except Exception as exc:
        lucian_logger.exception("Lucian weekly update failed: %s - %s", type(exc).__name__, exc)
        response = "⚠️ Lucian weekly review failed. The launcher logs have the details."

    balance_result = lucian_balance.capture_weekend_balance_progress(
        fnFolder_path,
        review_date=weekend_saturday_date(datetime.now(sgt)).isoformat(),
        timeout_seconds=LUCIAN_IBKR_BALANCE_TIMEOUT_SECONDS,
        logger=lucian_logger,
    )
    response = f"{response.rstrip()}\n\n{balance_result['section']}"

    send_lucian_message(bot, response)
    lucian_logger.info("Lucian weekly update sent")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TELEGRAM COMMAND QUEUE HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def describe_telegram_update(update):
    message = update.get("message") if isinstance(update, dict) else None
    if not isinstance(message, dict):
        return "non-message"

    text = str(message.get("text") or "").strip()
    if text.startswith("/"):
        return text.split()[0].split("@", 1)[0]
    if text:
        return "text-reply"
    return "message"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def telegram_command_worker(bot, worker_id):

    while True:
        try:
            update = telegram_command_queue.get(timeout=5)
        except queue.Empty:
            with telegram_worker_last_seen_lock:
                telegram_worker_last_seen[worker_id] = datetime.now(sgt)
            continue

        command_name = describe_telegram_update(update)
        start = pytime.monotonic()
        with telegram_worker_last_seen_lock:
            telegram_worker_last_seen[worker_id] = datetime.now(sgt)

        try:
            handle_message(bot, update)
            elapsed = pytime.monotonic() - start
        except Exception as exc:
            elapsed = pytime.monotonic() - start
            timePrint(
                f"Telegram command failed | worker={worker_id} | command={command_name} | "
                f"elapsed={elapsed:.1f}s | error={type(exc).__name__}: {exc}"
            )
            timePrint(traceback.format_exc())
            try:
                bot.sendMessage(CHAT_ID, f"Command failed: {command_name}. Check Xander logs for details.")
            except Exception as send_exc:
                timePrint(f"Telegram command failure notice failed: {type(send_exc).__name__} - {send_exc}")
        finally:
            with telegram_worker_last_seen_lock:
                telegram_worker_last_seen[worker_id] = datetime.now(sgt)
            telegram_command_queue.task_done()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def enqueue_telegram_update(bot, update):
    command_name = describe_telegram_update(update)
    try:
        telegram_command_queue.put_nowait(update)
        return True
    except queue.Full:
        timePrint(f"Telegram command queue full; dropping command={command_name}")
        try:
            bot.sendMessage(CHAT_ID, f"Xander command queue is full. Please retry: {command_name}")
        except Exception as exc:
            timePrint(f"Telegram queue-full notice failed: {type(exc).__name__} - {exc}")
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def start_telegram_command_workers(bot):
    for worker_id in range(TELEGRAM_COMMAND_WORKERS):
        with telegram_worker_last_seen_lock:
            telegram_worker_last_seen[worker_id] = datetime.now(sgt)
        threading.Thread(target=telegram_command_worker, args=(bot, worker_id), daemon=True).start()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def telegram_worker_watchdog():
    while True:
        pytime.sleep(TELEGRAM_LISTENER_HEARTBEAT_SECONDS)
        try:
            with telegram_worker_last_seen_lock:
                snapshot = dict(telegram_worker_last_seen)
            now = datetime.now(sgt)
            stale = {
                worker_id: int((now - last_seen).total_seconds())
                for worker_id, last_seen in snapshot.items()
                if (now - last_seen).total_seconds() > TELEGRAM_LISTENER_HEARTBEAT_SECONDS * 2
            }
            if stale:
                timePrint(f"Telegram command worker heartbeat stale | stale_workers={stale} | queue_size={telegram_command_queue.qsize()}")
        except Exception as exc:
            timePrint(f"Telegram worker watchdog failed: {type(exc).__name__} - {exc}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TELEGRAM COMMAND HANDLERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Function to handle incoming messages
def handle_message(bot, msg):

    global simulation_flag
    global simulation_user
    global updateMarketContext_flag
    global updateMarketContext_user
    global updateBenzinga_flag 
    global updateBenzinga_user 
    global ibkrInstantSells_flag
    global ibkrInstantSells_user
    global replaceIbkrSells_flag 
    global replaceIbkrSells_user 
    global fetchSpread_flag 
    global fetchSpread_user 
    global muteAlerts_flag
    global muteAlerts_user
    global queryTrades_flag
    global queryTrades_user
    global queryTrades_option

    message = msg.get("message") if isinstance(msg, dict) else None
    if not isinstance(message, dict):
        return

    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    message_thread_id = message.get("message_thread_id")
    message_text = message.get("text") or ""

    if LUCIAN_THREAD_ID is not None and chat_id == CHAT_ID and lucian_reviewer is not None and lucian_reviewer.extract_lucian_query(message_text) is not None and message_thread_id != LUCIAN_THREAD_ID:
        log_lucian_route_event("wrong_topic", f"thread_id={message_thread_id}; expected_thread_id={LUCIAN_THREAD_ID}")
        return

    if LUCIAN_THREAD_ID is not None and chat_id == CHAT_ID and message_thread_id == LUCIAN_THREAD_ID:
        handle_lucian_message(bot, message)
        return

    if chat_id == CHAT_ID and message_thread_id is None:
        message_text = message.get("text")
        from_user = message.get("from") or {}
        username = from_user.get("username")

        if not message_text or not username:
            return

        if (username in ALLOWED_USERS):

            if message_text.startswith('/help'):
                messageGuide = (
                    "===== URLS =====\n"
                    '<a href="https://status.openai.com">OPENAI Status</a>\n'
                    '<a href="https://www.dockerstatus.com">Docker Status</a>\n'
                    '<a href="https://www.interactivebrokers.com.sg/en/software/systemStatus.php">IBKR Status</a>\n'
                    '<a href="https://www.benzingastatus.com/">Benzinga Status</a>\n'
                    "\n"
                    "===== COMMANDS =====\n"
                    "List Commands: /help\n"
                    "Health Check: /health_check\n"
                    "Disable Trading: /disable_trading\n"
                    "Enable Trading: /enable_trading\n"
                    "Disable Stage 2: /disable_stage2\n"
                    "Enable Stage 2: /enable_stage2\n"
                    "Disable Scraping: /disable_scraping\n"
                    "Enable Scraping: /enable_scraping\n"
                    "Disable AI: /disable_ai\n"
                    "Enable AI: /enable_ai\n"
                    "Disable Discretionary AI: /disable_discretionary_ai\n"
                    "Enable Discretionary AI: /enable_discretionary_ai\n"
                    "Simulate: /simulate\n"
                    "Non-Regression: /initiate_nr\n"
                    "Get Market Context: /get_market_context\n"
                    "Get Market Context Prompt: /get_market_context_prompt\n"
                    "Update Market Context: /update_market_context\n"
                    "Set AM Trading: /set_am_trading\n"
                    "Set PM Trading: /set_pm_trading\n"
                    "Set Full Trading: /set_full_trading\n"
                    "Cancel All Orders: /cancel_orders\n"
                    "Sell Order(s) Now: /sell_now\n"
                    "Replace Sell(s): /replace_sells\n"
                    "Fetch Spreads: /fetch_spreads\n"
                    "Get Today's Order(s): /orders\n"
                    "Get Unrealized P&L: /pnl\n"
                    "Get Account Balance: /balance\n"
                    "Refresh All Balances: /refresh_balance\n"
                    "Refresh Available Balance: /refresh_avail_balance\n"
                    "Query Trade History: /query_trades\n"
                    "Restart Bot(s): /restart_bots\n"
                    "Start IB Gateway: /start_ib\n"
                    "Restart IB Gateway: /restart_ib\n"
                    "Kill IB Gateway: /kill_ib\n"
                    "Restart Docker: /restart_docker\n"
                    "Restart Web Scrape(s): /restart_web_scrapes\n"
                    "Update Benzinga API: /update_api\n"
                    "Reset Alert Periods: /reset_alerts\n"
                    "Mute Health Check: /mute_hc"
                )

                bot.sendMessage(CHAT_ID, messageGuide, parse_mode="HTML", disable_web_page_preview=True)

            elif message_text.startswith('/health_check'):
                check_status(True)
                
            elif message_text.startswith('/disable_trading'):
                update_tradingFlag(False)
                bot.sendMessage(CHAT_ID, "Trading has been disabled.")

            elif message_text.startswith('/enable_trading'):
                update_tradingFlag(True)
                bot.sendMessage(CHAT_ID, "Trading has been enabled.")

            elif message_text.startswith(('/disable_stage2')):
                update_stage2Flag(False)
                bot.sendMessage(CHAT_ID, "Stage 2 has been disabled.")

            elif message_text.startswith(('/enable_stage2')):
                update_stage2Flag(True)
                bot.sendMessage(CHAT_ID, "Stage 2 has been enabled.")

            elif message_text.startswith('/disable_scraping'):
                update_scrapingFlag(False)
                bot.sendMessage(CHAT_ID, "Scraping has been disabled.")

            elif message_text.startswith('/enable_scraping'):
                update_scrapingFlag(True)
                bot.sendMessage(CHAT_ID, "Scraping has been enabled.")

            elif message_text.startswith('/disable_ai'):
                update_aiFlag(False)

            elif message_text.startswith('/enable_ai'):
                update_aiFlag(True)

            elif message_text.startswith('/disable_discretionary_ai'):
                update_aiDiscretionaryFlag(False)

            elif message_text.startswith('/enable_discretionary_ai'):
                update_aiDiscretionaryFlag(True)

            elif not message_text.startswith('/simulate') and simulation_flag == 1 and simulation_user and simulation_user == username:
                start_simulation(message_text)
                simulation_flag = 0
                simulation_user = None

            elif message_text.startswith('/simulate'):
                simulation_flag = 1
                simulation_user = username
                bot.sendMessage(CHAT_ID, f"Please post your content, {username}")

            elif not message_text.startswith('/mute_hc') and muteAlerts_flag == 1 and muteAlerts_user and muteAlerts_user == username:
                mute_health_checks(message_text)
                muteAlerts_flag = 0
                muteAlerts_user = None

            elif message_text.startswith('/mute_hc'):
                muteAlerts_flag = 1
                muteAlerts_user = username
                bot.sendMessage(CHAT_ID, f"Please state your preferred mute time in hours, {username}")

            elif not message_text.startswith('/update_market_context') and updateMarketContext_flag == 1 and updateMarketContext_user and updateMarketContext_user == username:
                update_market_context(message_text)
                updateMarketContext_flag = 0
                updateMarketContext_user = None

            elif message_text.startswith('/update_market_context'):
                updateMarketContext_flag = 1
                updateMarketContext_user = username
                bot.sendMessage(CHAT_ID, f"Please provide your update, {username}")

            elif (message_text.startswith('/1') or message_text.startswith('/2') or message_text.startswith('/3') or message_text.startswith('/4')) and queryTrades_flag == 1 and queryTrades_user and queryTrades_user == username:
                queryTrades_option = int(message_text.split("@", 1)[0][1:])
                bot.sendMessage(CHAT_ID, f"Please specify X, {username}")

            elif queryTrades_flag == 1 and queryTrades_user and queryTrades_user == username:
                queryTradeHistory(queryTrades_option, message_text)
                queryTrades_flag = 0
                queryTrades_user = None
                queryTrades_option = None

            elif message_text.startswith('/query_trades'):
                queryTrades_flag = 1
                queryTrades_user = username
                bot.sendMessage(CHAT_ID,
                    (
                        "SELECT * FROM TradeHistory WHERE..\n"
                        "(/1) Last X trading days\n"
                        "(/2) Amount > X\n"
                        "(/3) Amount < X\n"
                        "(/4) Ticker = X"
                    ))
            
            elif not message_text.startswith('/update_api') and updateBenzinga_flag == 1 and updateBenzinga_user and updateBenzinga_user == username:
                update_api(message_text)
                updateBenzinga_flag = 0
                updateBenzinga_user = None

            elif message_text.startswith('/update_api'):
                updateBenzinga_flag = 1
                updateBenzinga_user = username
                bot.sendMessage(CHAT_ID, f"Please paste the API Key, {username}")

            elif message_text.startswith('/initiate_nr'):
                start_nr()

            elif message_text.startswith('/get_market_context_prompt'):
                fetch_market_context_prompt()

            elif message_text.startswith('/get_market_context'):
                show_market_context()

            elif message_text.startswith('/set_am_trading'):
                adjust_trading_hrs("AM")

            elif message_text.startswith('/set_pm_trading'):
                adjust_trading_hrs("PM")

            elif message_text.startswith('/set_full_trading'):
                adjust_trading_hrs("FULL")

            elif message_text.startswith('/cancel_orders'):
                cancel_open_orders()

            elif message_text.startswith('/orders'):
                get_last_market_orders()

            elif message_text.startswith('/pnl'):
                get_unrealized_profitloss()
            
            elif not message_text.startswith('/sell_now') and ibkrInstantSells_flag == 1 and ibkrInstantSells_user and ibkrInstantSells_user == username:
                create_sells(message_text.replace("$","").upper())
                ibkrInstantSells_flag = 0
                ibkrInstantSells_user = None

            elif message_text.startswith('/sell_now'):
                ibkrInstantSells_flag = 1
                ibkrInstantSells_user = username
                bot.sendMessage(CHAT_ID, f"{username}, please specify the intended ticker(s), separated by commas, all in CAPS. Do not include the '$' symbol.")
            
            elif not message_text.startswith('/replace_sells') and replaceIbkrSells_flag == 1 and replaceIbkrSells_user and replaceIbkrSells_user == username:
                replace_sells(message_text.replace("$","").upper())
                replaceIbkrSells_flag = 0
                replaceIbkrSells_user = None

            elif message_text.startswith('/replace_sells'):
                replaceIbkrSells_flag = 1
                replaceIbkrSells_user = username
                bot.sendMessage(CHAT_ID, f"{username}, please specify the intended ticker(s), separated by commas, all in CAPS. Do not include the '$' symbol.")
            
            elif not message_text.startswith('/fetch_spread') and fetchSpread_flag == 1 and fetchSpread_user and fetchSpread_user == username:
                fetch_spreads(message_text.replace("$","").upper())
                fetchSpread_flag = 0
                fetchSpread_user = None

            elif message_text.startswith('/fetch_spread'):
                fetchSpread_flag = 1
                fetchSpread_user = username
                bot.sendMessage(CHAT_ID, f"{username}, please specify the intended ticker(s), separated by commas, all in CAPS. Do not include the '$' symbol.")

            elif message_text.startswith('/balance'):
                get_account_balance()

            elif message_text.startswith('/refresh_balance'):
                refresh_account_balance()

            elif message_text.startswith('/refresh_avail_balance'):
                refresh_avail_account_balance()
            
            elif message_text.startswith('/restart_bots'):
                restart_app(None, True, "BOT(s)")
            
            elif message_text.startswith('/start_ib'):
                threading.Thread(target=start_ib_gateway, args=(True,), daemon=True).start()
            
            elif message_text.startswith('/restart_ib'):
                threading.Thread(target=restart_ib_gateway, args=(True,), daemon=True).start()
            
            elif message_text.startswith('/kill_ib'):
                kill_ib_gateway()
                bot.sendMessage(CHAT_ID, "IB Gateway has been killed.")
            
            elif message_text.startswith('/restart_docker'):
                restart_docker(True)
            
            elif message_text.startswith('/restart_web_scrapes'):
                restart_nitter(True)

            elif message_text.startswith('/reset_alerts'):
                reset_alert_periods()
                bot.sendMessage(CHAT_ID, "Alert periods have been reset.")

        else:
            bot.sendMessage(CHAT_ID, f"Hello {username}, you are not authorized to create requests here.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# COMMAND ACTION HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fetch_market_context_prompt():
    
    file_path = os.path.join(fnFolder_path, 'MANUAL_SNAPSHOT_PROMPT.txt')

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    bot.sendMessage(CHAT_ID, content)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def show_market_context():
    
    file_path = os.path.join(fnFolder_path, 'XANDER_MarketContext_Snapshot.txt')

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    bot.sendMessage(CHAT_ID, content)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_api(apiKey):

    file_path = os.path.join(fnFolder_path, 'XANDER_Benzinga_API.txt')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(apiKey)

    bot.sendMessage(CHAT_ID, "Benzinga API Key has been updated.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_market_context(market_context):

    file_path = os.path.join(fnFolder_path, 'XANDER_MarketContext_Snapshot.txt')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(market_context)

    bot.sendMessage(CHAT_ID, "Market Context has been updated.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def adjust_trading_hrs(session_type):
    file_name_TradingHoursSet = "XANDER_TradingHours_Set.txt"
    file_path_TradingHoursSet = os.path.join(fnFolder_path, file_name_TradingHoursSet)
    if (session_type in ["AM", "PM"]):
        write_text(file_path_TradingHoursSet, session_type)
    else:
        write_text(file_path_TradingHoursSet, "")
    
    trdOpenTime = datetime.strptime(tradingOpen, "%H:%M")
    trdCloseTime = datetime.strptime(tradingClose, "%H:%M")
    mktOpenTime = datetime.strptime(marketOpen, "%H:%M")
    mktCloseTime = datetime.strptime(marketClose, "%H:%M")
    mktHalfDayTime = datetime.strptime(marketHalfDay, "%H:%M")

    if (session_type == "AM"):
        bot.sendMessage(
            CHAT_ID,
            (
                "📢 TRADING HOURS UPDATED (TODAY)\n"
                f"• Trading Starts : {trdOpenTime.strftime('%I:%M%p').lstrip('0')} SGT\n"
                f"• Market Opens : {mktOpenTime.strftime('%I:%M%p').lstrip('0')} SGT\n"
                f"• Trading Ends   : {mktHalfDayTime.strftime('%I:%M%p').lstrip('0')} SGT\n"
                f"• Market Closes : {mktHalfDayTime.strftime('%I:%M%p').lstrip('0')} SGT"
            )
        )
    elif (session_type == "PM"):
        bot.sendMessage(
            CHAT_ID,
            (
                "📢 TRADING HOURS UPDATED (TODAY)\n"
                f"• Trading Starts : {trdOpenTime.strftime('%I:%M%p').lstrip('0')} SGT\n"
                f"• Market Opens : {mktHalfDayTime.strftime('%I:%M%p').lstrip('0')} SGT\n"
                f"• Trading Ends   : {trdCloseTime.strftime('%I:%M%p').lstrip('0')} SGT\n"
                f"• Market Closes : {mktCloseTime.strftime('%I:%M%p').lstrip('0')} SGT"
            )
        )
    elif (session_type == "FULL"):
        bot.sendMessage(
            CHAT_ID,
            (
                "📢 TRADING HOURS UPDATED (TODAY)\n"
                f"• Trading Starts : {trdOpenTime.strftime('%I:%M%p').lstrip('0')} SGT\n"
                f"• Market Opens : {mktOpenTime.strftime('%I:%M%p').lstrip('0')} SGT\n"
                f"• Trading Ends   : {trdCloseTime.strftime('%I:%M%p').lstrip('0')} SGT\n"
                f"• Market Closes : {mktCloseTime.strftime('%I:%M%p').lstrip('0')} SGT"
            )
        )
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_adjusted_trading_hrs():
    file_name_TradingHoursSet = "XANDER_TradingHours_Set.txt"
    file_path_TradingHoursSet = os.path.join(fnFolder_path, file_name_TradingHoursSet)
    val = read_text_default(file_path_TradingHoursSet, "")
    if (val in ["AM","PM"]):
        return val
    else:
        return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def cancel_open_orders():
    write_gateway_file(GATEWAY_IBKR_CANCEL_ORDERS)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_last_market_orders():
    write_gateway_file(GATEWAY_IBKR_GET_ORDERS)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_unrealized_profitloss():
    write_gateway_file(GATEWAY_IBKR_GET_PNL)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def create_sells(tickers_listed):
    write_gateway_file(GATEWAY_IBKR_CREATE_SELLS, tickers_listed)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def replace_sells(tickers_listed):
    write_gateway_file(GATEWAY_IBKR_REPLACE_SELLS, tickers_listed)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fetch_spreads(ticker_listed):
    write_gateway_file(GATEWAY_IBKR_FETCH_SPREADS, ticker_listed)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_account_balance():
    write_gateway_file(GATEWAY_IBKR_GET_BALANCE)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def refresh_account_balance():
    write_gateway_file(GATEWAY_IBKR_REFRESH_BALANCE)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def refresh_avail_account_balance():
    write_gateway_file(GATEWAY_IBKR_REFRESH_AVAIL_BALANCE)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def reset_alert_periods():
    file_path = bot_path("_OUTPUT", "XANDER", "VALIDATE", "TELE", "messages.txt")

    if os.path.exists(file_path):
        open(file_path, 'w').close()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def start_simulation(content):
    write_gateway_file("SIMULATION_SocialMarket.txt", content)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_tradingFlag(boolFlag):
    write_flag_file("XANDER_TradingFlag.txt", "0" if boolFlag == False else "1")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_stage2Flag(boolFlag):
    write_flag_file("XANDER_Stage2Flag.txt", "0" if boolFlag == False else "1")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_scrapingFlag(boolFlag):
    write_flag_file("XANDER_ScrapingFlag.txt", "0" if boolFlag == False else "1")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_aiFlag(boolFlag):
    if (boolFlag == False):
        write_flag_file("XANDER_AiFlag.txt", "0")
        bot.sendMessage(CHAT_ID, "AI has been disabled.")
    else:
        write_flag_file("XANDER_AiFlag.txt", "1")
        bot.sendMessage(CHAT_ID, "AI has been enabled.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_aiDiscretionaryFlag(boolFlag):
    if (boolFlag == False):
        write_flag_file("XANDER_DiscretionaryAiFlag.txt", "0")
        bot.sendMessage(CHAT_ID, "Discretionary AI has been disabled.")
    else:
        write_flag_file("XANDER_DiscretionaryAiFlag.txt", "1")
        bot.sendMessage(CHAT_ID, "Discretionary AI has been enabled.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def reset_trades():
    write_gateway_file("SOCIALMARKET_Trades.txt")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def start_nr():
    write_gateway_file("NON-REGRESSION_SocialMarket.txt", "NON-REGRESSION_SocialMarket")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def disconnect_ib_gateway():
    write_gateway_file(GATEWAY_IBKR_DISCONNECT)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# PROCESS / SERVICE HEALTH HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def is_docker_engine_responsive():
    try:
        output = subprocess.check_output("docker info", stderr=subprocess.STDOUT, shell=True, timeout=10).decode()
        return "server version" in output.lower()
    except subprocess.CalledProcessError:
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def is_ib_gateway_process_running():
    try:
        query = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -eq 'java.exe' -and $_.CommandLine -like '*ibgateway*' } | "
            "Select-Object -ExpandProperty ProcessId"
        )
        result = subprocess.check_output(["powershell", "-NoProfile", "-Command", query], text=True, stderr=subprocess.DEVNULL, timeout=10)
        return any(line.strip().isdigit() for line in result.splitlines())
    except Exception as e:
        timePrint(f"Failed to check IB Gateway process: {e}")
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def are_nitter_web_instances_running(expected_count=3):
    try:
        # List names and images of all running containers
        output = subprocess.check_output(
            'docker ps --format "{{.Names}} {{.Image}}"',
            shell=True,
            timeout=10
        ).decode().strip().lower()

        # Match only containers using the zedeus/nitter image (excluding redis)
        nitter_containers = [
            line for line in output.splitlines()
            if "zedeus/nitter" in line
        ]

        return len(nitter_containers) >= expected_count, len(nitter_containers)

    except subprocess.CalledProcessError as e:
        print(f"Error checking Docker containers: {e}")
        return False, 0
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def getMutedPeriod():
    mute_file_path = gateway_path(XANDER_MUTED_PERIOD)
    unmute_time_str = read_text_default(mute_file_path, "")

    try:
        return datetime.strptime(unmute_time_str, "%d-%b-%Y %I:%M%p")
    except ValueError:
        return datetime.min
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def mute_health_checks(string_hrs):
    try:
        hours = int(string_hrs)
        if hours < 0:
            bot.sendMessage(CHAT_ID, "Invalid time range detected. The process will now exit.")
            return
    except:
        bot.sendMessage(CHAT_ID, "Invalid time range detected. The process will now exit.")
        return
    
    # Calculate the unmute time
    now = datetime.now()
    unmute_time = now + timedelta(hours=hours)

    # Format unmute time as string (e.g., '06-Jun-2025 08:00PM')
    unmute_time_str = unmute_time.strftime("%d-%b-%Y %I:%M%p")

    # Define file path
    write_gateway_file(XANDER_MUTED_PERIOD, unmute_time_str)

    bot.sendMessage(CHAT_ID, f"Alerts for health checks are muted until {unmute_time_str}.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def parse_dt_sgt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=SGT)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def compute_cutoff_always_previous_open(now: datetime, tradingOpen: str, x: int) -> datetime:
    hh, mm = map(int, tradingOpen.split(":"))
    open_t = time(hh, mm)

    now = now.astimezone(SGT)
    today_open = datetime.combine(now.date(), open_t, tzinfo=SGT)

    # x=1 => yesterday open
    # x=2 => 2 days ago open ...
    cutoff = today_open - timedelta(days=x)
    return cutoff
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TRADE HISTORY QUERY HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def queryTradeHistory(optionType, inputVal):

    # ---------------- Validate input ----------------
    if optionType in (1, 2, 3):
        raw = str(inputVal).strip()
        if not raw.lstrip("+-").isdigit():
            bot.sendMessage(CHAT_ID, "Invalid value specified. Please input only digits ('-' is allowed).")
            return
        x = int(raw)
        if optionType == 1 and x <= 0:
            bot.sendMessage(CHAT_ID, "X must be > 0.")
            return

    elif optionType == 4:
        ticker = str(inputVal).upper().strip()
        if not ticker:
            bot.sendMessage(CHAT_ID, "Invalid value specified.")
            return
    else:
        bot.sendMessage(CHAT_ID, "Invalid option.")
        return

    # ---------------- Load file ----------------
    file_path = os.path.join(fnFolder_path, "GATEWAY_IBKR_TradeHistory.txt")
    if not os.path.exists(file_path):
        bot.sendMessage(CHAT_ID, "Trade history file not found.")
        return

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    if not records:
        bot.sendMessage(CHAT_ID, "No trade history records found.")
        return

    # ---------------- Helpers ----------------
    def parse_dt(s):
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

    # ---------------- Filter ----------------
    results = []
    now = datetime.now()

    if optionType == 1:
        now = datetime.now(SGT)
        cutoff = compute_cutoff_always_previous_open(now, tradingOpen, x)

        for r in records:
            try:
                td = parse_dt_sgt(r["TriggeredDate"])
                if td >= cutoff:
                    results.append(r)
            except Exception:
                pass

    elif optionType == 2:
        threshold = float(x)
        for r in records:
            try:
                if float(r["PNL"]) > threshold:
                    results.append(r)
            except Exception:
                pass

    elif optionType == 3:
        threshold = float(x)
        for r in records:
            try:
                if float(r["PNL"]) < threshold:
                    results.append(r)
            except Exception:
                pass

    elif optionType == 4:
        for r in records:
            if r.get("Ticker", "").upper() == ticker:
                results.append(r)

    if not results:
        bot.sendMessage(CHAT_ID, "No matching records found.")
        return

    results.sort(
        key=lambda r: parse_dt(r.get("TriggeredDate", "1970-01-01 00:00:00")),
        reverse=True
    )

    # ---------------- Output ----------------
    def fmt(r):
        c = r.get("Content", "")
        pnl = r.get("PNL")
        try:
            pnl_val = float(pnl)
            pnl_emoji = "🟢" if pnl_val >= 0 else "🔴"
            pnl_str = f"{pnl_emoji} PNL: {pnl_val:.2f}"
        except (TypeError, ValueError):
            pnl_str = "PNL: N.A."

        return (
            f"{r.get('Ticker')} | {pnl_str}\n"
            f"FGI: {r.get('FGI')}\n"
            f"Triggered: {r.get('TriggeredDate')} SGT\n"
            f"Sold: {r.get('SoldDate')} SGT\n"
            f"Entry / Exit: {r.get('EntryPrice')} ➜ {r.get('ExitPrice')}\n"
            f"📝 {c}"
        )

    batch = 10
    total = len(results)

    header = f"TradeHistory Results ({total})\n"
    if optionType == 1:
        header += f"Filter: Last {x} trading days\n\n"
    elif optionType == 2:
        header += f"Filter: PNL > {x}\n\n"
    elif optionType == 3:
        header += f"Filter: PNL < {x}\n\n"
    elif optionType == 4:
        header += f"Filter: Ticker = {ticker}\n\n"

    for i in range(0, total, batch):
        chunk = results[i:i + batch]
        body = "\n\n---\n\n".join(
            f"#{i+j+1}\n{fmt(r)}" for j, r in enumerate(chunk)
        )
        if i == 0:
            msg = header + body
        else:
            msg = f"(continued) {i+1}-{min(i+batch, total)} of {total}\n\n" + body

        bot.sendMessage(CHAT_ID, msg)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# HEALTH STATUS HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_status(isRequest):

    global isStartup
    cross_mark = '\u274C'
    tradingEnabled = 0
    stage2Enabled = 0
    scrapingEnabled = 0
    aiEnabled = 0
    discretionaryAiEnabled = 0
    botsOkay = True
    nrOngoing = False
    responseFile_Content = "TEST"
    unmute_time = getMutedPeriod()

    #------------------------------------------------------------------------------------#
    # Check Traing control flag 
    xander_TradingFlag_fileName = "XANDER_TradingFlag.txt"
    file_path_xander_TradingFlag = os.path.join(fnFolder_path, xander_TradingFlag_fileName)
    if not os.path.isfile(file_path_xander_TradingFlag):
        write_text(file_path_xander_TradingFlag, "0")
    else:
        content = read_text_default(file_path_xander_TradingFlag, "0")
        if (content == "1"):
            tradingEnabled = 1

    if (tradingEnabled == 1):
        messageStatus = f'Trading Enabled: True\n'
    else:
        messageStatus = f'Trading Enabled: False\n'
    #------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------#
    # Check Stage 2 control flag
    xander_Stage2Flag_fileName = "XANDER_Stage2Flag.txt"
    file_path_xander_Stage2Flag = os.path.join(fnFolder_path, xander_Stage2Flag_fileName)
    if not os.path.isfile(file_path_xander_Stage2Flag):
        write_text(file_path_xander_Stage2Flag, "0")
    else:
        content = read_text_default(file_path_xander_Stage2Flag, "0")
        if (content == "1"):
            stage2Enabled = 1

    if (stage2Enabled == 1):
        messageStatus += f'Stage 2 Enabled: True\n'
    else:
        messageStatus += f'Stage 2 Enabled: False\n'
    #------------------------------------------------------------------------------------#
    
    #------------------------------------------------------------------------------------#
    # Check Scraping control flag 
    xander_ScrapingFlag_fileName = "XANDER_ScrapingFlag.txt"
    file_path_xander_ScrapingFlag = os.path.join(fnFolder_path, xander_ScrapingFlag_fileName)
    if not os.path.isfile(file_path_xander_ScrapingFlag):
        write_text(file_path_xander_ScrapingFlag, "0")
    else:
        content = read_text_default(file_path_xander_ScrapingFlag, "0")
        if (content == "1"):
            scrapingEnabled = 1

    if (scrapingEnabled == 1):
        messageStatus += f'Scraping Enabled: True\n'
    else:
        messageStatus += f'Scraping Enabled: False\n'
    #------------------------------------------------------------------------------------#
    
    #------------------------------------------------------------------------------------#
    # Check AI control flag 
    xander_AiFlag_fileName = "XANDER_AiFlag.txt"
    file_path_xander_AiFlag = os.path.join(fnFolder_path, xander_AiFlag_fileName)
    if not os.path.isfile(file_path_xander_AiFlag):
        write_text(file_path_xander_AiFlag, "0")
    else:
        content = read_text_default(file_path_xander_AiFlag, "0")
        if (content == "1"):
            aiEnabled = 1

    if (aiEnabled == 1):
        messageStatus += f'AI Enabled: True\n'
    else:
        messageStatus += f'AI Enabled: False\n'
    #------------------------------------------------------------------------------------#
    
    #------------------------------------------------------------------------------------#
    # Check Discretionary AI control flag 
    xander_AiDiscretionaryFlag_fileName = "XANDER_DiscretionaryAiFlag.txt"
    file_path_xander_AiDiscretionaryFlag = os.path.join(fnFolder_path, xander_AiDiscretionaryFlag_fileName)
    if not os.path.isfile(file_path_xander_AiDiscretionaryFlag):
        write_text(file_path_xander_AiDiscretionaryFlag, "0")
    else:
        content = read_text_default(file_path_xander_AiDiscretionaryFlag, "0")
        if (content == "1"):
            discretionaryAiEnabled = 1

    if (discretionaryAiEnabled == 1):
        messageStatus += f'D-AI Enabled: True\n'
    else:
        messageStatus += f'D-AI Enabled: False\n'
    #------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------#
    # If it can respond it will always be OK
    messageStatus += f'- Xander: OK\n'
    #------------------------------------------------------------------------------------#
    
    #------------------------------------------------------------------------------------#
    # Write file to IBKR Trader Workstation (IBKR)
    file_path_IBKR_Workstation = gateway_path(GATEWAY_IBKR_STATUS)
    write_gateway_file(GATEWAY_IBKR_STATUS, responseFile_Content)

    # Write file to BOT (IBKR)
    file_path_IBKR = gateway_path(STATUS_IBKR_AUTOMATE)
    write_gateway_file(STATUS_IBKR_AUTOMATE, responseFile_Content)

    # Write file to BOT (SocialMarket)
    file_path_SocialMarket = gateway_path(STATUS_SOCIALMARKET)
    write_gateway_file(STATUS_SOCIALMARKET, responseFile_Content)
    #------------------------------------------------------------------------------------#
    
    #------------------------------------------------------------------------------------#
    # Check Docker
    docker_engine_status = is_docker_engine_responsive()
    if docker_engine_status:
        messageStatus += f'- Docker: OK\n'
    else:
        messageStatus += f'- Docker: NOT OK {cross_mark}\n'
    #------------------------------------------------------------------------------------#
    
    #------------------------------------------------------------------------------------#
    # Check local instances of Nitter
    nitter_ok, nitter_count = are_nitter_web_instances_running()
    if nitter_ok:
        messageStatus += f'- Local Nitter(s): OK ({nitter_count}/3)\n'
    else:
        messageStatus += f'- Local Nitter(s): NOT OK ({nitter_count}/3) {cross_mark}\n'
    #------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------#
    # Set timer to give time for bots to respond
    if isStartup == 1:
        pytime.sleep(20)
        isStartup = 0
    else:
        pytime.sleep(10)
    #------------------------------------------------------------------------------------#

    # Check response for IBKR Trader Workstation (IBKR)
    if os.path.exists(file_path_IBKR_Workstation):
        file_path_IBKR_Maintenance = gateway_path(MAINTENANCE_IBKR)
        if os.path.exists(file_path_IBKR_Maintenance):
            messageStatus += f'- IBKR GW: SCHEDULED MAINTENANCE\n'
            os.remove(file_path_IBKR_Maintenance)
        else:
            messageStatus += f'- IBKR GW: NOT OK {cross_mark}\n'

    else:
        file_path_IBKR_Maintenance = gateway_path(MAINTENANCE_IBKR)
        if os.path.exists(file_path_IBKR_Maintenance):
            messageStatus += f'- IBKR GW: SCHEDULED MAINTENANCE\n'
            os.remove(file_path_IBKR_Maintenance)
        else:
            messageStatus += f'- IBKR GW: OK\n'

    # Check response for BOT (IBKR)
    if os.path.exists(file_path_IBKR):
        botsOkay = False

    # Check response for BOT (SocialMarket)
    if os.path.exists(file_path_SocialMarket):
        nr_file_name_SocialMarket = "NON-REGRESSION_SocialMarket.txt"
        nr_file_path_SocialMarket = os.path.join(fnFolder_path, nr_file_name_SocialMarket)
        if os.path.exists(nr_file_path_SocialMarket):
            nrOngoing = True
            os.remove(nr_file_path_SocialMarket)
        else:
            botsOkay = False
    #------------------------------------------------------------------------------------#

    if (nrOngoing):
        messageStatus += f'- BOT(s): NR ONGOING\n'
    elif (botsOkay):
        messageStatus += f'- BOT(s): OK\n'
    else:
        messageStatus += f'- BOT(s): NOT OK {cross_mark}\n'

    # Finalize
    file_path_xander_Health = gateway_path(XANDER_HEALTH)
    healthFlag = read_text_default(file_path_xander_Health, "1")
    if healthFlag not in {"0", "1"} or not os.path.exists(file_path_xander_Health):
        healthFlag = "1"
        write_gateway_file(XANDER_HEALTH, healthFlag)

    isMuted = datetime.now() < unmute_time
    isWeekendMute = False
    
    now_sgt = datetime.now(sgt)  # keep as full datetime object
    current_time = now_sgt.time()
    current_weekday = now_sgt.weekday()  # Monday = 0, Tuesday = 1, Wednesday = 2, Thursday = 3, Friday = 4, Saturday = 5, Sunday = 6

    isWeekendMute = (messageStatus.count(': OK') >= 2) and ("IBKR GW: NOT OK" in messageStatus) and ((current_weekday == 5 and current_time > weekendMuteAfterSat) or (current_weekday == 6) or (current_weekday == 0 and current_time < weekendMuteUntilMon))
    
    if (isRequest):
        bot.sendMessage(CHAT_ID, messageStatus)
        if ("NOT OK" in messageStatus):
            if int(healthFlag) == 1:
                write_text(file_path_xander_Health, "0")
        else:
            if int(healthFlag) == 0:
                write_text(file_path_xander_Health, "1")
    else:
        if ("NOT OK" in messageStatus):
            now = datetime.now(sgt)
            if int(healthFlag) == 1:
                if not isWeekendMute:
                    bot.sendMessage(CHAT_ID, messageStatus)
                if ("BOT(s): NOT OK ❌" in messageStatus):
                    forceRestart_IBKR_file_path = bot_path(".BOT_Launch", "Functions", "IBKR_FORCE_RESTART.txt")
                    if os.path.exists(forceRestart_IBKR_file_path):
                        os.remove(forceRestart_IBKR_file_path)
                        threading.Thread(target=restart_ib_gateway, args=(False,), daemon=True).start()
                    
                    forceRestart_SocialMarket_file_path = bot_path(".BOT_Launch", "Functions", "SOCIALMARKET_FORCE_RESTART.txt")
                    if os.path.exists(forceRestart_SocialMarket_file_path):
                        os.remove(forceRestart_SocialMarket_file_path)
                        restart_app(None, True, "SocialMarket.py")
                    else:
                        restart_app(None, True, "BOT(s)")
                else:
                    write_text(file_path_xander_Health, "0")
            elif int(healthFlag) == 0 and now.minute == 0 and not isMuted and not isWeekendMute:
                bot.sendMessage(CHAT_ID, messageStatus)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# RESTART / SHUTDOWN HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def kill_process_by_script(script_name):
    try:
        query = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*" + script_name + "*' } | "
            "Select-Object -ExpandProperty ProcessId"
        )
        result = subprocess.check_output(["powershell", "-NoProfile", "-Command", query], text=True, stderr=subprocess.DEVNULL, timeout=10)
        for line in result.splitlines():
            line = line.strip()
            if line.isdigit():
                subprocess.run(["taskkill", "/PID", line, "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except Exception as e:
        print(f"Failed to kill {script_name}: {e}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def kill_ib_gateway():
    try:
        disconnect_ib_gateway()
        pytime.sleep(2.5)
        query = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -eq 'java.exe' -and $_.CommandLine -like '*ibgateway*' } | "
            "Select-Object -ExpandProperty ProcessId"
        )
        result = subprocess.check_output(["powershell", "-NoProfile", "-Command", query], text=True, stderr=subprocess.DEVNULL, timeout=10)
        for line in result.splitlines():
            line = line.strip()
            if line.isdigit():
                subprocess.run(["taskkill", "/PID", line, "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except Exception as e:
        print(f"Failed to kill IB Gateway: {e}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def start_ib_gateway(isRequest):

    if not is_ib_gateway_process_running():
        try:
            disconnect_ib_gateway()
            script_path = r"C:\IBC\StartGateway.bat"

            subprocess.Popen(
                ["cmd.exe", "/c", script_path],
                cwd=os.path.dirname(script_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            print("IB Gateway initialized...")
            
            if isRequest:
                bot.sendMessage(CHAT_ID, f"Initiating IB Gateway start. Health check will be conducted after 60s.")
                pytime.sleep(60)
                check_status(True)

        except Exception as e:
            error_msg = f"Failed to start IB Gateway: {e}"
            print(error_msg)
            if isRequest:
                bot.sendMessage(CHAT_ID, error_msg)

    else:
        active_msg = "IB Gateway is already active."
        print(active_msg)
        if isRequest:
            bot.sendMessage(CHAT_ID, active_msg)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def restart_ib_gateway(isRequest):

    if is_ib_gateway_process_running():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect(("127.0.0.1", 7465))  # Port must match config.ini
                s.sendall(b"RESTART\n")

            disconnect_ib_gateway()

            if isRequest:
                bot.sendMessage(CHAT_ID, f"Initiating IB Gateway restart (approx. 90 seconds). Health check will be performed upon completion.")
                pytime.sleep(90)
                check_status(True)

        except Exception as e:
            error_msg = f"Error restarting IB Gateway: {e}"
            print(error_msg)
            if isRequest:
                bot.sendMessage(CHAT_ID, error_msg)

    else:
        if isRequest:
            start_ib_gateway(isRequest)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def restart_app(processes, isReboot, name):
    # BOT (SocialMarket), BOT (IBKR) & IBKR TWS (Trader Workstation from IBKR, TO BE IMPLEMENTED)
    process_name = []
    path = []

    if not (isReboot):
        process_name = botsService_name
        path = botsFolder_path

    else:
        if (name == "BOT(s)"):
            process_name = botsService_name
            path = botsFolder_path
        elif (name == "SocialMarket.py"):
            process_name = [botsService_name[0]]
            path = [botsFolder_path[0]]
        elif (name == "IBKR.py"):
            process_name = [botsService_name[1]]
            path = [botsFolder_path[1]]
    
    for i in range(len(process_name)):
        try:
            kill_process_by_script(process_name[i])
        except Exception as e:
            print(f"Error stopping process: {e}")

        try:
            child_env = os.environ.copy()
            child_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
            subprocess.Popen([PYTHON_EXE, path[i]], cwd=os.path.dirname(path[i]), env=child_env)
        except Exception as e:
            print(f"Error starting process with {PYTHON_EXE}: {e}")

        pytime.sleep(2)

    if isReboot:
        print(f"{name} restarted.")
        bot.sendMessage(CHAT_ID, f"{name} restarted.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def restart_docker(isRequest):
    try:
        timer = 0
        docker_processes = [
                "Docker Desktop.exe",
                "com.docker.backend.exe",
                "com.docker.dev-envs.exe",
                "com.docker.build.exe",
                "vmmem"
            ]

        for proc in docker_processes:
            timer += 1
            subprocess.run(["taskkill", "/f", "/im", proc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)

        pytime.sleep(timer)

        # Restart Docker Desktop
        docker_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
        subprocess.Popen([docker_path])
        if isRequest:
            pytime.sleep(5)
            bot.sendMessage(CHAT_ID, f"Docker restarted.")
        else:
            pytime.sleep(10)
        return True
    except Exception as e:
        error_msg = f"Failed to restart Docker: {e}"
        timePrint(error_msg)
        if isRequest:
            bot.sendMessage(CHAT_ID, error_msg)
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def restart_nitter(isRequest):
    nitter_path = r"C:\nitter"
    try:
        subprocess.run("docker-compose up -d", cwd=nitter_path, shell=True, check=True, timeout=30)
        pytime.sleep(2)
        if isRequest:
            bot.sendMessage(CHAT_ID, f"Local Nitter(s) restarted.")
        return True
    except Exception as e:
        error_msg = f"Failed to restart Local Nitter(s): {e}"
        timePrint(error_msg)
        if isRequest:
            bot.sendMessage(CHAT_ID, error_msg)
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# DATA REFRESH HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_market_context_freshness():
    snapshot_file_path = bot_path(".BOT_Launch", "Functions", "XANDER_MarketContext_Snapshot.txt")

    # Use SGT timezone
    now = datetime.now(sgt)

    # Run only on Sundays & Mondays from 5pm to 11:59pm every 3 hours
    if now.weekday() in [6, 0] and now.hour in marketContextCheckHours and now.minute == 0:  # Sunday is 6, Monday is 0
        if not os.path.exists(snapshot_file_path):
            bot.sendMessage(CHAT_ID, "Market context snapshot file was not found.")
            return

        last_modified = dt.datetime.fromtimestamp(os.path.getmtime(snapshot_file_path)).astimezone(sgt)
        days_since_modification = (now - last_modified).days

        if days_since_modification >= 5:
            bot.sendMessage(CHAT_ID, "Market context has not been updated.\nPlease ensure it is refreshed before the start of the new week to maintain accurate context for Monday's market open.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def download_csv(url):
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_tickers_list():
    csv_directory = bot_path(".BOT_Launch", "SocialMarket", "TICKERS")
    nasdaq_url = "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv"
    nyse_url = "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv"

    nasdaq_df = download_csv(nasdaq_url)  # Contains: 'Symbol', 'Security Name'
    nyse_df = download_csv(nyse_url)      # Contains: 'ACT Symbol', 'Company Name'

    # Rename NYSE to match Nasdaq format
    nyse_df = nyse_df.rename(columns={
        "ACT Symbol": "Symbol",
        "Company Name": "Security Name"
    })

    # Combine the two datasets
    combined_df = pd.concat([
        nasdaq_df[["Symbol", "Security Name"]],
        nyse_df[["Symbol", "Security Name"]]
    ])

    # Drop duplicates
    combined_df = combined_df.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    # Save to file
    os.makedirs(csv_directory, exist_ok=True)
    output_path = os.path.join(csv_directory, "combined_unique_ticker_list.csv")
    combined_df.to_csv(output_path, index=False)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_sgd_usd_rate_cache():
    url = "https://api.frankfurter.app/latest"
    params = { "amount": 1, "from": "SGD", "to": "USD" }

    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        rate = data["rates"]["USD"]

        if not isinstance(rate, (int, float)) or rate <= 0:
            raise ValueError(f"Invalid SGD/USD rate from Frankfurter: {rate!r}")

        os.makedirs(os.path.dirname(sgd_usd_rate_cache_path), exist_ok=True)
        atomic_write_text(sgd_usd_rate_cache_path, str(rate))
        timePrint(f"[FX] SGD/USD rate cache updated: {rate}")
        return True
    except Exception as e:
        timePrint(f"[FX][WARN] Failed to update SGD/USD rate cache: {e}")
        return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def automate_tickers_list():
    csv_directory = bot_path(".BOT_Launch", "SocialMarket", "TICKERS")
    ticker_file_dir = bot_path(".BOT_Launch", "SocialMarket", "TICKERS", "combined_unique_ticker_list.csv")

    # Use SGT timezone
    now = datetime.now(sgt)

    # Run only on Saturdays, Sundays & Mondays from 6pm to 11:59pm
    if now.weekday() in [5, 6, 0] and tickerRefreshStartHour <= now.hour < tickerRefreshEndHour:  # Saturday is 5, Sunday is 6, Monday is 0
        if not os.path.exists(ticker_file_dir):
            update_tickers_list()
            update_sgd_usd_rate_cache()
            return

        last_modified = dt.datetime.fromtimestamp(os.path.getmtime(ticker_file_dir)).astimezone(sgt)
        days_since_modification = (now - last_modified).days

        if days_since_modification >= 5:
            update_tickers_list()
            update_sgd_usd_rate_cache()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# HOUSEKEEPING / SCHEDULED CHECKS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def purge(target_path, hours):
    
    # Use SGT timezone
    now = datetime.now(sgt)

    if not os.path.isdir(target_path):
        return

    for file_name in os.listdir(target_path):

        file_path = os.path.join(target_path, file_name)
        if not os.path.isfile(file_path):
            continue

        last_modified = dt.datetime.fromtimestamp(os.path.getmtime(file_path)).astimezone(sgt)
        hrs_since_modification = (now - last_modified).total_seconds() / 3600

        if hrs_since_modification > hours:
            os.remove(file_path)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def check_us_market_status():
    nytz = pytz.timezone(XANDER_MARKET_TIMEZONE)

    now_sgt = datetime.now(sgt)
    us_date = now_sgt.astimezone(nytz).date()

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=str(us_date), end_date=str(us_date))

    if schedule.empty:
        msg = (
            f"🚫 US Market Closed Today ({us_date.strftime('%b %d')})\n"
            f"There will be no trading session tonight (SGT)"
        )
        bot.sendMessage(CHAT_ID, msg)
        update_tradingFlag(False)
        bot.sendMessage(CHAT_ID, "Trading has been disabled.")
        update_aiDiscretionaryFlag(False)
        update_aiFlag(False)
        return

    open_time = schedule.iloc[0]['market_open']
    close_time = schedule.iloc[0]['market_close']

    # Expected regular hours
    expected_open = nytz.localize(datetime.combine(us_date, datetime.strptime(usRegularOpen, "%H:%M").time()))
    expected_close = nytz.localize(datetime.combine(us_date, datetime.strptime(usRegularClose, "%H:%M").time()))

    # Flags
    late_open = open_time > expected_open
    early_close = close_time < expected_close

    # Optional: check total session length (regular is 6.5 hours)
    session_hours = (close_time - open_time).total_seconds() / 3600
    short_session = session_hours < 6.5  # tweak if you want

    update_tradingFlag(True)

    if early_close or short_session or late_open:
        close_sgt = close_time.astimezone(sgt).strftime("%I:%M%p")
        open_sgt = open_time.astimezone(sgt).strftime("%I:%M%p")
        msg = (
            f"⚠️ US Market Half Day Today ({us_date.strftime('%b %d')})\n"
            f"Open: {open_sgt} (SGT)\n"
            f"Close: {close_sgt} (SGT)"
        )
        bot.sendMessage(CHAT_ID, msg)
        if early_close:
            adjust_trading_hrs("AM")
        else:
            adjust_trading_hrs("PM")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def regular_check_iteration():
    now_sgt = datetime.now(sgt)
    current_weekday = now_sgt.weekday()
    isTradingHrsToBeAdjusted = check_adjusted_trading_hrs()

    if run_once_per_day_when_due("PRUNE_FIRED", now_sgt, pruneFired):
        prune_fired(keep_days=3)

    if LUCIAN_WEEKLY_SUMMARY_ENABLED and run_once_per_weekend_when_due("LUCIAN_WEEKLY_SUMMARY", now_sgt, LUCIAN_WEEKLY_SUMMARY_SGT):
        threading.Thread(target=send_lucian_weekly_update, args=(bot,), daemon=True).start()

    if current_weekday in range(0, 5) and run_once_per_day_when_due("IB_GATEWAY_START", now_sgt, ibStart):
        if is_ib_gateway_process_running():
            threading.Thread(target=restart_ib_gateway, args=(False,), daemon=True).start()
        else:
            threading.Thread(target=start_ib_gateway, args=(False,), daemon=True).start()

    if current_weekday in range(0, 5) and run_once_per_day_when_due("DAILY_RESTART", now_sgt, xanderRestart):
        restart_app(None, True, "BOT(s)")

    if current_weekday in range(0, 5) and run_once_per_day_when_due("CHECK_US_MARKET_STATUS", now_sgt, checkTradingDay):
        threading.Thread(target=check_us_market_status, args=(), daemon=True).start()

    if current_weekday in range(1, 6):
        if isTradingHrsToBeAdjusted == "AM":
            if run_once_per_day_when_due("AI_DISABLE_HALF_DAY", now_sgt, marketHalfDay):
                update_aiFlag(False)
                update_aiDiscretionaryFlag(False)

            if run_once_per_day_when_due("RESET_TRADING_HRS", now_sgt, tradingClose):
                adjust_trading_hrs("")
        else:
            if run_once_per_day_when_due("AI_DISABLE_FULL_DAY", now_sgt, tradingClose):
                update_aiFlag(False)
                update_aiDiscretionaryFlag(False)

            if isTradingHrsToBeAdjusted == "PM" and run_once_per_day_when_due("RESET_TRADING_HRS", now_sgt, tradingClose):
                adjust_trading_hrs("")

    if current_weekday in range(0, 5) and run_once_per_day_when_due("AI_ENABLE", now_sgt, tradingOpen):
        reset_trades()
        update_aiFlag(True)
        update_aiDiscretionaryFlag(True)
        check_us_market_status()

    current_time_str = now_sgt.strftime('%H:%M')
    if current_time_str not in ibStartBuffer:
        check_status(False)
    check_market_context_freshness()
    automate_tickers_list()
    if not os.path.exists(sgd_usd_rate_cache_path):
        update_sgd_usd_rate_cache()
    purge(posts_path, 12)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def regular_check_loop():
    load_fired()

    while True:
        try:
            regular_check_iteration()
        except Exception as e:
            timePrint(f"regular_check_loop error: {type(e).__name__} - {e}")

        now2 = datetime.now(sgt)
        next_run = now2.replace(second=0, microsecond=0) + timedelta(minutes=1)
        pytime.sleep(max(0, (next_run - now2).total_seconds()))
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TELEGRAM POLLING LOOP
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def safe_polling_loop(bot, handler):
    offset = None
    last_heartbeat = pytime.monotonic()

    while True:
        try:
            updates = bot.getUpdates(offset=offset, timeout=20)

            now_monotonic = pytime.monotonic()
            if now_monotonic - last_heartbeat >= TELEGRAM_LISTENER_HEARTBEAT_SECONDS:
                last_heartbeat = now_monotonic

            for update in updates:
                offset = update["update_id"] + 1

                if "my_chat_member" in update:
                    info = update["my_chat_member"]
                    user = info["from"].get(
                        "username",
                        info["from"].get("first_name", "Unknown")
                    )
                    old_status = info["old_chat_member"]["status"]
                    new_status = info["new_chat_member"]["status"]
                    timePrint(f"Ignored my_chat_member event: {user} {old_status} → {new_status}")
                    continue

                if "message" not in update:
                    continue

                enqueue_telegram_update(bot, update)

        except (
            requests.exceptions.RequestException,
            http.client.RemoteDisconnected,
            urllib3.exceptions.ProtocolError,
            urllib3.exceptions.MaxRetryError,
            urllib3.exceptions.ConnectTimeoutError,
        ) as e:
            timePrint(f"Telegram polling error: {type(e).__name__} — {e}")
            pytime.sleep(5)

        except Exception as e:
            timePrint(f"Unexpected polling error: {type(e).__name__} — {e}")
            pytime.sleep(2)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ENTRYPOINT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':

    timePrint("Booting up..")

    xander_bot = telepot.Bot(BOT_TOKEN)
    bot = xander_bot
    initialize_lucian()
    start_telegram_command_workers(xander_bot)
    threading.Thread(target=safe_polling_loop, args=(xander_bot, handle_message), daemon=True).start()
    threading.Thread(target=telegram_worker_watchdog, daemon=True).start()

    # Initialize IB Gateway
    #start_ib_gateway(False)

    # Initialize Docker Desktop
    if is_docker_engine_responsive():
        print("Docker Desktop is already active.")
    else:
        if restart_docker(False):
            print("Docker Desktop Initialized...")

    # Initialize Local Nitter(s)
    nitter_ok, _ = are_nitter_web_instances_running()
    if nitter_ok:
        print("Local Nitter(s) are already active.")
    else:
        if restart_nitter(False):
            print("Local Nitter(s) Initialized...")
    
    restart_app(None, False, None)

    print("Xander is listening for messages...")
    print("Lucian is listening for messages...")

    # Do a purge on startup
    purge(xanderLogs_path, 336) # 2 weeks
    purge(lucianLogs_path, 336) # 2 weeks
    purge(ibkrLogs_path, 336) # 2 weeks
    purge(ibkrClosed_path, 336) # 2 weeks
    purge(ibkrError_path, 336) # 2 weeks
    purge(ibkrIgnored_path, 336) # 2 weeks
    purge(dynamicPrompts_path, 336) # 2 weeks

    # Start check_status scheduler in a separate thread
    threading.Thread(target=regular_check_loop, daemon=True).start()
    
    # Keep the script running indefinitely
    while True:
        pytime.sleep(1)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
