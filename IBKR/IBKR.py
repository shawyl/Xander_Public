"""Broker automation workflow for Xander.

Handles broker-side validation, order lifecycle handling, market-data checks,
and related operational safeguards. Sensitive execution and risk details remain
inside the implementation and should not be exposed in public documentation.

AI status: Maintained with AI.
"""

import sys
import asyncio
if sys.version_info >= (3, 10):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

import requests
from ib_insync import *
from datetime import datetime, time, timedelta
import os
import time as time_module
import shutil
import logging
import pytz
import json
import subprocess
import numpy as np
import warnings
import contextlib

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
import faiss
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import queue
import threading
import platform
import re
import math
import yfinance as yf
import tempfile
from dotenv import load_dotenv
from decimal import Decimal, InvalidOperation

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

# Load environment variables from a .env file (if present)
load_dotenv()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# CONFIGURATION / ENVIRONMENT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# IBKR API connection
ib = IB()
IB_CLIENT_ID = 1
TWS_HOST = "127.0.0.1"
TWS_PORT = 4001

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

def bot_path(*parts):
    return os.path.join(BOTHUB_ROOT, *parts)

# Define the API / BOT TOKEN to connect
XANDER_BOT_TOKEN = require_env("XANDER_TELEGRAM_BOT_TOKEN")
XANDER_CHAT_ID = require_int_env("XANDER_TELEGRAM_CHAT_ID")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# MODEL INITIALIZATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Initialize MODEL

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# CONNECTION STATUS THROTTLING HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Initialize Connect Fail Count
state = {"connect_fail_count": 0}
CONNECT_FAILURE_LOG_INTERVAL_SECONDS = 180

def should_log_connect_failure(isFrequentCall):
    now = time_module.time()
    last_logged = state.get("last_connect_failure_log_time", 0)

    if state["connect_fail_count"] == 1 or now - last_logged >= CONNECT_FAILURE_LOG_INTERVAL_SECONDS:
        state["last_connect_failure_log_time"] = now
        return True

    return False

def should_log_repeated_status(key):
    now = time_module.time()
    last_logged = state.get(key, 0)

    if last_logged == 0 or now - last_logged >= CONNECT_FAILURE_LOG_INTERVAL_SECONDS:
        state[key] = now
        return True

    return False

def emit_connect_status(key, message, isWeekendMute, include_status_link=False, log_file=False):
    if not should_log_repeated_status(key) or isWeekendMute:
        return

    if log_file:
        ibkrautomate_logger.info(message)
    print(message)

    telegram_message = message
    if include_status_link:
        telegram_message += "\nIB Gateway system availability: https://www.interactivebrokers.com.sg/en/software/systemStatus.php"

    if not isWeekendMute:
        send_telegram_message_thread(telegram_message, "XANDER", 12, 0)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# PATHS / GLOBAL RUNTIME STATE
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Global queue where threads will send valid orders
order_queue = queue.Queue()

# Threading locks
ibkr_lock = Lock()
lock = threading.Lock()

# Global Times
BOTHUB_ROOT = require_env("BOTHUB_ROOT")
XANDER_TIMEZONE = require_env("XANDER_TIMEZONE")
XANDER_IBKR_MAINTENANCE_TIMEZONE = require_env("XANDER_IBKR_MAINTENANCE_TIMEZONE")
tradingOpen = get_env_time("XANDER_MARKET_OPEN_SGT")
tradingClose = get_env_time("XANDER_MARKET_CLOSE_SGT")
tradingHalfDay = get_env_time("XANDER_MARKET_HALF_DAY_CLOSE_SGT")
updateBalWindowOpen = get_env_time("XANDER_BALANCE_UPDATE_OPEN_SGT")
updateBalWindowClose = get_env_time("XANDER_BALANCE_UPDATE_CLOSE_SGT")
ibkrWeekendMuteAfterSat = get_env_time("XANDER_WEEKEND_MUTE_AFTER_SAT_SGT")
ibkrWeekendMuteUntilMon = get_env_time("XANDER_IBKR_WEEKEND_MUTE_UNTIL_MON_SGT")
ibkrMaintenanceStart = get_env_time("XANDER_IBKR_MAINTENANCE_START_ET")
ibkrMaintenanceEnd = get_env_time("XANDER_IBKR_MAINTENANCE_END_ET")
ibkrSatMaintenanceStart = get_env_time("XANDER_IBKR_SAT_MAINTENANCE_START_SGT")
ibkrSatMaintenanceEnd = get_env_time("XANDER_IBKR_SAT_MAINTENANCE_END_SGT")

# Folder directories
functionsFolder_path = bot_path(".BOT_Launch", "Functions")
validationFolder_path = bot_path("_OUTPUT", "XANDER", "VALIDATE")
ibkrFolder_path = bot_path(".BOT_Launch", "IBKR")
incoming_directory = bot_path(".BOT_Launch", "IBKR", "INCOMING")
open_directory = bot_path(".BOT_Launch", "IBKR", "OPEN")
closed_directory = bot_path(".BOT_Launch", "IBKR", "CLOSED")
error_directory = bot_path(".BOT_Launch", "IBKR", "ERROR")
ignored_directory = bot_path(".BOT_Launch", "IBKR", "IGNORED")
stale_validating_directory = bot_path(".BOT_Launch", "IBKR", "FAILED_STALE_VALIDATING")
logsFolder_path = bot_path("_OUTPUT", "IBKR", "LOGS")
posts_path = bot_path(".BOT_Launch", "SocialMarket", "POSTS")
sgd_usd_rate_cache_path = bot_path(".BOT_Launch", "SocialMarket", "FX", "sgd_usd_rate.txt")

STALE_VALIDATING_SECONDS = 10 * 60
LIMIT_BREACH_PRICE_QUANT = Decimal("0.0001")
LIMIT_BREACH_MAX_SPREAD_PCT = float(os.getenv("IBKR_LIMIT_BREACH_MAX_SPREAD_PCT", "20"))
LIMIT_BREACH_PENDING_COOLDOWN_SECONDS = int(os.getenv("IBKR_LIMIT_BREACH_PENDING_COOLDOWN_SECONDS", "120"))
limit_breach_sell_pending = {}
LUCIAN_BALANCE_SNAPSHOT_REQUEST_FILE = "GATEWAY_LUCIAN_BALANCE_SNAPSHOT_IBKR.txt"
LUCIAN_BALANCE_SNAPSHOT_RESPONSE_FILE = "GATEWAY_LUCIAN_BALANCE_SNAPSHOT_RESPONSE.json"

# Establish logging (ibkrautomate_logger.info(f''))
os.makedirs(logsFolder_path, exist_ok=True)
current_date = datetime.now().strftime('%d-%m-%Y')
log_filename = f"IBKRAutomate-{current_date}.log"
log_filepath = os.path.join(logsFolder_path, log_filename)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ibkrautomate_logger = logging.getLogger('ibkrautomate_logger')
ibkrautomate_logger.setLevel(logging.INFO)
ibkrautomate_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
ibkrautomate_handler.setLevel(logging.INFO)
ibkrautomate_handler.setFormatter(formatter)
ibkrautomate_logger.addHandler(ibkrautomate_handler)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# LOGGING HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def timePrint(log):
    sgt_now = datetime.now(pytz.timezone(XANDER_TIMEZONE))
    formatted_time = sgt_now.strftime('%d-%b-%y %I:%M:%S %p')  # e.g., 10-May-25 01:22:43 AM
    print(f"[{formatted_time}] {log}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def log_and_print(msg):
    ibkrautomate_logger.info(msg)
    timePrint(msg)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def log_warning_and_print(msg):
    ibkrautomate_logger.warning(msg)
    timePrint(msg)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# GENERAL FILE / PARSING HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def parse_exec_time(exec_time_str):
    # Format is usually: '20240524  10:34:22'
    return datetime.strptime(exec_time_str, "%Y%m%d  %H:%M:%S")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_contract_min_tick(contract):
    previous_timeout = getattr(ib, "RequestTimeout", 0)
    try:
        ib.RequestTimeout = 5
        details = ib.reqContractDetails(contract)
        if details and details[0].minTick:
            return float(details[0].minTick)
    except Exception as e:
        log_and_print(f"[IBKR][WARN] Unable to fetch minTick for {contract.symbol}; using 0.01 fallback: {e}")
    finally:
        try:
            ib.RequestTimeout = previous_timeout
        except Exception:
            pass
    return 0.01
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def round_price_to_tick(price, min_tick, direction="nearest"):
    if not min_tick or min_tick <= 0:
        min_tick = 0.01

    scaled = price / min_tick
    if direction == "up":
        rounded = math.ceil(scaled - 1e-12) * min_tick
    elif direction == "down":
        rounded = math.floor(scaled + 1e-12) * min_tick
    else:
        rounded = round(scaled) * min_tick

    return round(rounded, 6)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def ibkr_connection_heartbeat():
    if not ib.isConnected():
        raise ConnectionError("IBKR is not connected")
    return True
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fetch_account_summary_snapshot():
    with ibkr_lock:
        account_values = ib.accountValues()
        if account_values:
            return account_values

        accounts = ib.managedAccounts()
        if accounts:
            ib.reqAccountUpdates(accounts[0])
            account_values = ib.accountValues(accounts[0])
            if account_values:
                return account_values

        raise RuntimeError("IBKR account values unavailable after account update request")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def atomic_write_text(path: str, text: str) -> None:
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=os.path.basename(path) + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
            tmpf.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def atomic_write_lines(path: str, lines: list[str]) -> None:
    atomic_write_text(path, "".join(lines))
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def safe_append_line(filePath: str, line: str) -> None:
    needs_newline = False
    try:
        with open(filePath, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > 0:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b"\n"
    except FileNotFoundError:
        needs_newline = False

    with open(filePath, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(line.rstrip("\r\n") + "\n")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def ensure_lines_end_with_newline(lines: list[str]) -> None:
    if lines and not str(lines[-1]).endswith("\n"):
        lines[-1] = str(lines[-1]) + "\n"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def append_marker_line(lines: list[str], line: str) -> None:
    ensure_lines_end_with_newline(lines)
    lines.append(line.rstrip("\r\n") + "\n")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def atomic_write_json(path: str, value) -> None:
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=os.path.basename(path) + ".tmp.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
            json.dump(value, tmpf, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def safe_load_json(path: str, default=None):
    if default is None:
        default = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def safe_read_text(path: str, encodings=("utf-8", "ISO-8859-1")) -> str:
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding=encodings[0], errors="replace") as f:
        return f.read()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def parse_signal_filename(filename: str) -> dict:
    base = filename
    for suffix in (".processing", ".validating", ".txt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    parts = base.split("_")

    if len(parts) == 5:
        return {
            "bot_name": parts[0],
            "action": parts[1],
            "lot": None,
            "ticker": parts[2],
            "order_id": None,
            "unique_id": parts[3],
            "reprocess": parts[4],
        }
    if len(parts) == 6:
        return {
            "bot_name": parts[0],
            "action": parts[1],
            "lot": int(parts[2]) if parts[2].isdigit() else None,
            "ticker": parts[3],
            "order_id": parts[4],
            "unique_id": parts[4],
            "reprocess": parts[5],
        }
    raise ValueError(f"Invalid signal filename format: {filename}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def set_marker_line(lines: list[str], prefix: str, value: str) -> None:
    marker = f"{prefix}{value}**"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(prefix) and stripped.endswith("**"):
            lines[index] = marker + "\n"
            return
    append_marker_line(lines, marker)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# TELEGRAM HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def send_telegram_message(text, bot, lockHrs, retryCount):

    bot_token =  XANDER_BOT_TOKEN
    chat_id = XANDER_CHAT_ID

    toProceed = True

    if lockHrs > 0:
        with lock:
            base_text = text.strip()
            tele_folder = os.path.join(validationFolder_path, "TELE")
            messages_file_path = os.path.join(tele_folder, "messages.txt")
            os.makedirs(tele_folder, exist_ok=True)

            message_records = safe_load_json(messages_file_path, [])
            
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
            if (lockHrs < 1000):
                alert_msg = f"* Alert above will be paused for {lockHrs}HR(S) to prevent spam"
                alert_pattern = r"\* Alert above will be paused for \d+HR\(S\) to prevent spam"

                if not re.search(alert_pattern, text):
                    text += f"\n{alert_msg}"

            atomic_write_json(messages_file_path, message_records)

    if not toProceed:
        return

    if retryCount > 0:
        delay_time = 5 * retryCount
        delay_msg = f"* Delayed for ~{delay_time}s due to rate limiting"
        if re.search(r"\* Delayed for ~\d+\.?\d*s due to rate limiting", text):
            text = re.sub(r"\* Delayed for ~\d+\.?\d*s due to rate limiting", delay_msg, text)
        else:
            text += f"\n{delay_msg}"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }

    try:
        response = requests.post(url, json=payload, timeout=(5, 15))
    except requests.exceptions.RequestException as exc:
        error_msg = f"Failed to send message [{retryCount}]: {exc}"
        time_module.sleep(5)

        if retryCount < 11: # retry maximum of 10
            send_telegram_message(text, bot, lockHrs, retryCount + 1)

        timePrint(error_msg)
        return

    if response.status_code != 200:
        error_msg = f"Failed to send message [{retryCount}]. HTTP Status Code: {response.status_code}"
        time_module.sleep(5)

        if retryCount < 11: # retry maximum of 10
            send_telegram_message(text, bot, lockHrs, retryCount + 1)

        timePrint(error_msg)
        return

    try:
        response_json = response.json()
    except ValueError as exc:
        error_msg = f"Failed to parse Telegram response [{retryCount}]: {exc}"
        time_module.sleep(5)

        if retryCount < 11: # retry maximum of 10
            send_telegram_message(text, bot, lockHrs, retryCount + 1)

        timePrint(error_msg)
        return

    if not response_json.get("ok"):
        error_msg = f"Error sending message [{retryCount}]: {response_json.get('description')}"
        time_module.sleep(5)

        if retryCount < 6: # retry maximum of 5
            send_telegram_message(text, bot, lockHrs, retryCount + 1)

        timePrint(error_msg)
        return
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def send_telegram_message_thread(text, bot, lockHrs, retryCount):
    threading.Thread(target=send_telegram_message, args=(text, bot, lockHrs, retryCount), daemon=True).start()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# IBKR CONNECTION HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def ibkrErrorHandler(reqId, errorCode, errorString, advancedOrderRejectJson=""):
    
    ignore_codes = [2104, 2106, 1100, 1102, 2103, 2107, 2108, 2119, 2157, 2158]
    if errorCode in ignore_codes:
        return  # skip noisy info

    noisy_timeout_fragments = (
        "positions request timed out",
        "open orders request timed out",
        "completed orders request timed out",
        "account updates",
    )
    error_text = str(errorString or "").lower()
    if any(fragment in error_text for fragment in noisy_timeout_fragments):
        ibkrautomate_logger.debug(f"[IBKR Handler][suppressed] Error {errorCode}: {errorString}")
        return
    
    error_msg = f"[IBKR Handler] Error {errorCode}: {errorString}"
    if should_log_repeated_status(f"ibkr_error_handler_{errorCode}_{errorString}"):
        send_telegram_message_thread(error_msg, "XANDER", 12, 0)
        ibkrautomate_logger.info(error_msg)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def is_within_ibkr_maintenance():
    # US focus
    now_utc = datetime.now(pytz.utc)

    # Timezones
    ET = pytz.timezone(XANDER_IBKR_MAINTENANCE_TIMEZONE)
    SGT = pytz.timezone(XANDER_TIMEZONE)

    now_et = now_utc.astimezone(ET)
    now_sgt = now_utc.astimezone(SGT)

    hour_min_et = now_et.time()
    hour_min_sgt = now_sgt.time()

    # === U.S. Daily Reset: 11:45 PM – 12:45 AM ET
    if ibkrMaintenanceStart <= hour_min_et or hour_min_et <= ibkrMaintenanceEnd:
        return True

    # === Friday Extended Reset: Sat 11:00 AM – 3:00 PM SGT
    if now_sgt.weekday() == 5 and ibkrSatMaintenanceStart <= hour_min_sgt <= ibkrSatMaintenanceEnd:
        return True

    return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def connect_ibkr(isFrequentCall, isRequest):

    sgt = pytz.timezone(XANDER_TIMEZONE)
    now_sgt = datetime.now(sgt)  # keep as full datetime object
    current_time = now_sgt.time()
    current_weekday = now_sgt.weekday()  # Monday = 0, Tuesday = 1, Wednesday = 2, Thursday = 3, Friday = 4, Saturday = 5, Sunday = 6

    isWeekendMute = (current_weekday == 5 and current_time > ibkrWeekendMuteAfterSat) or (current_weekday == 6) or (current_weekday == 0 and current_time < ibkrWeekendMuteUntilMon)

    try:
    
        # Check if it is undergoing scheduled maintenance
        if not is_within_ibkr_maintenance():

            # Check if IBKR Gateway window is opened locally
            output = subprocess.check_output('tasklist', shell=True).decode().lower()
            if 'java.exe' not in output:
                connectFailed_msg = "IB Gateway connection failed — Gateway application is currently not running on the local machine."
                emit_connect_status("gateway_application_not_running", connectFailed_msg, isWeekendMute)
                if isRequest:
                    return False

            else:
                # Proceed to check connectivity
                if not ib.isConnected():
                    if not state.get("error_handler_registered"):
                        ib.errorEvent += ibkrErrorHandler
                        state["error_handler_registered"] = True
                    ib.connect(TWS_HOST, TWS_PORT, clientId=IB_CLIENT_ID)
                    state["connect_fail_count"] = 0 # Reset fail count

                if ib.isConnected():
                    try:
                        ibkr_connection_heartbeat()
                        if isRequest:
                            return True
                        return
                    except Exception as e:
                        state["connect_fail_count"] += 1
                        connectFailed_msg = "IBKR Server connection failed — Session most likely expired."
                        emit_connect_status("ibkr_session_expired", connectFailed_msg, isWeekendMute)
                        if isRequest:
                            return False

                else:
                    state["connect_fail_count"] += 1
                    connectFailed_msg = "IB Gateway connection refused — Gateway may be logged out, under unscheduled maintenance, or unable to reach IBKR servers."
                    emit_connect_status("ibkr_gateway_refused", connectFailed_msg, isWeekendMute, include_status_link=True)
                    if isRequest:
                        return False

    except Exception as e:
        state["connect_fail_count"] += 1
        connectFailed_msg = "IB Gateway connection failed — Gateway may be logged out, under unscheduled maintenance, or unable to reach IBKR servers."
        should_emit_exception = should_log_connect_failure(isFrequentCall)
        if should_emit_exception:
            ibkrautomate_logger.info(connectFailed_msg)
            ibkrautomate_logger.info(e)
            timePrint(connectFailed_msg)
            print(e)
            connectFailed_msg += "\nIB Gateway system availability: https://www.interactivebrokers.com.sg/en/software/systemStatus.php"
            if not isWeekendMute:
                send_telegram_message_thread(connectFailed_msg, "XANDER", 12, 0)
        if isRequest:
            return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def safe_ibkr_connect_check(isFrequent):
    with ibkr_lock:
        if not ib.isConnected():
            return connect_ibkr(isFrequent, True)
        try:
            ibkr_connection_heartbeat()
            return True
        except:
            # Connection is stale or broken
            return connect_ibkr(isFrequent, True)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def safe_ibkr_connect(isFrequent):
    with ibkr_lock:
        if not ib.isConnected():
            connect_ibkr(isFrequent, False)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ORDER VALIDATION / DISPATCH HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def handle_signals_concurrently():
    with ThreadPoolExecutor(max_workers=10) as executor:
        for filename in os.listdir(incoming_directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(incoming_directory, filename)
                executor.submit(preprocess_and_queue_signal, file_path)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def unique_destination_path(directory, filename):
    destination = os.path.join(directory, filename)
    if not os.path.exists(destination):
        return destination

    name, extension = os.path.splitext(filename)
    counter = 1
    while True:
        candidate = os.path.join(directory, f"{name}_{counter}{extension}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def move_to_ignored(file_path, reason_code, detail=None, exc=None, original_filename=None, destination_filename=None):
    current_filename = os.path.basename(file_path)
    original_filename = original_filename or destination_filename or current_filename
    destination_filename = destination_filename or original_filename
    destination_path = unique_destination_path(ignored_directory, destination_filename)

    detail_text = str(detail).strip() if detail is not None else ""
    if exc is not None:
        exc_text = f"{type(exc).__name__}: {exc}"
        detail_text = f"{detail_text}; exception={exc_text}" if detail_text else f"exception={exc_text}"
    if not detail_text:
        detail_text = "N.A."

    log_and_print(
        "Moving to IGNORED | "
        f"original_file={original_filename} | "
        f"file={current_filename} | "
        f"destination_folder={ignored_directory} | "
        f"destination_file={os.path.basename(destination_path)} | "
        f"reason={reason_code} | "
        f"detail={detail_text}"
    )

    try:
        shutil.move(file_path, destination_path)
    except Exception as move_exc:
        log_and_print(
            "Failed moving to IGNORED | "
            f"original_file={original_filename} | "
            f"file={current_filename} | "
            f"destination_folder={ignored_directory} | "
            f"reason={reason_code} | "
            f"exception={type(move_exc).__name__}: {move_exc}"
        )
        raise

    return destination_path
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def recover_stale_validating_files(max_age_seconds=STALE_VALIDATING_SECONDS):
    os.makedirs(stale_validating_directory, exist_ok=True)

    try:
        filenames = os.listdir(incoming_directory)
    except FileNotFoundError:
        return

    now = time_module.time()
    for filename in filenames:
        if not filename.endswith(".validating"):
            continue

        source_path = os.path.join(incoming_directory, filename)
        if not os.path.isfile(source_path):
            continue

        try:
            age_seconds = now - os.path.getmtime(source_path)
        except OSError as e:
            log_and_print(f"[STALE_VALIDATING][WARN] Could not inspect {filename}: {e}")
            continue

        if age_seconds < max_age_seconds:
            continue

        destination_path = unique_destination_path(stale_validating_directory, filename)
        try:
            shutil.move(source_path, destination_path)
            log_and_print(
                f"[STALE_VALIDATING] {filename} is older than {max_age_seconds}s and was moved to "
                f"{stale_validating_directory} for manual review. It was not requeued."
            )
        except OSError as e:
            log_and_print(f"[STALE_VALIDATING][ERROR] Failed to move {filename}: {e}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def has_open_order(current_filename):
    current_name_part = os.path.splitext(current_filename)[0]  # e.g., "XANDER_BUY_TSLA_12345"
    current_parts = current_name_part.split('_')

    current_pending_order = '_'.join(current_parts[0:3])  # e.g., "XANDER_BUY_TSLA"

    for file in os.listdir(open_directory):

        file_name_part = os.path.splitext(file)[0]
        file_parts = file_name_part.split('_')

        existing_order = '_'.join(file_parts[0:3])  # e.g., "XANDER_BUY_TSLA"

        if existing_order == current_pending_order:
            return True

    return False
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

MARKET_CAP_LINE_RE = re.compile(r"^\*\*MARKETCAP AT\s+(\d+)\*\*$")
RISK_PROFILE_START = "**RISK PROFILE**"
RISK_PROFILE_END = "**END RISK PROFILE**"

def _format_percent_for_profile(value):
    value = float(value)
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value:.4f}".rstrip("0").rstrip(".") + "%"

def _parse_percent_field(value):
    text = value.strip()
    if not text.endswith("%"):
        raise ValueError(f"percent field missing % suffix: {value!r}")
    return float(text[:-1].strip())

def _parse_bool_field(value):
    text = value.strip()
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"invalid boolean field: {value!r}")

def parse_market_cap_from_lines(lines, filePath=None, log_warnings=False):
    raw_market_cap_line = None
    malformed_market_cap_line = None

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("**MARKETCAP AT"):
            continue

        raw_market_cap_line = stripped
        match = MARKET_CAP_LINE_RE.fullmatch(stripped)
        if match:
            return int(match.group(1)), False, raw_market_cap_line, malformed_market_cap_line

        malformed_market_cap_line = stripped
        if log_warnings:
            log_warning_and_print(
                f"[RISK_PROFILE][WARN] Malformed MARKETCAP line ignored | file={filePath} | line={malformed_market_cap_line}"
            )
        return None, True, raw_market_cap_line, malformed_market_cap_line

    return None, True, raw_market_cap_line, malformed_market_cap_line

def build_risk_profile(market_cap, failed_lookup, isEarnings):
    if failed_lookup or market_cap is None:
        risk_branch = "FALLBACK"
        allocation_percent = 35
        trailing_percent = 1
        dynamic_trail_percent = 0
        limit_tp_percent = 4
        limit_sl_percent = 2

    elif market_cap >= 50_000_000_000 and isEarnings:   # >= $50B
        risk_branch = "LARGE_CAP_EARNINGS"
        allocation_percent = 35
        trailing_percent = 3
        dynamic_trail_percent = 1
        limit_tp_percent = 4
        limit_sl_percent = 2

    elif market_cap >= 100_000_000_000 and not isEarnings:   # >= $100B
        risk_branch = "MEGA_CAP_NON_EARNINGS"
        allocation_percent = 50
        trailing_percent = 1
        dynamic_trail_percent = 0
        limit_tp_percent = 2
        limit_sl_percent = 1

    elif market_cap >= 1_000_000_000 and not isEarnings:   # $1B to $100B
        risk_branch = "MID_CAP_NON_EARNINGS"
        allocation_percent = 35
        trailing_percent = 3
        dynamic_trail_percent = 1
        limit_tp_percent = 6
        limit_sl_percent = 3

    elif not isEarnings:                               # 0 to $1B
        risk_branch = "MICRO_CAP_NON_EARNINGS"
        allocation_percent = 15
        trailing_percent = 10
        dynamic_trail_percent = 5
        limit_tp_percent = 10
        limit_sl_percent = 5

    else:
        risk_branch = "INSUFFICIENT_EARNINGS_MARKET_CAP"
        allocation_percent = 0
        trailing_percent = 0
        dynamic_trail_percent = 0
        limit_tp_percent = 0
        limit_sl_percent = 0

    return {
        "market_cap": market_cap,
        "failed_lookup": failed_lookup,
        "is_earnings": isEarnings,
        "risk_branch": risk_branch,
        "allocation_percent": allocation_percent,
        "trailing_percent": trailing_percent,
        "dynamic_trail_percent": dynamic_trail_percent,
        "limit_tp_percent": limit_tp_percent,
        "limit_sl_percent": limit_sl_percent,
    }

def parse_risk_profile_from_lines(lines, filePath=None):
    start_idx = None
    end_idx = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == RISK_PROFILE_START:
            start_idx = idx
        elif stripped == RISK_PROFILE_END and start_idx is not None:
            end_idx = idx
            break

    if start_idx is None:
        return None, False, None

    if end_idx is None:
        return None, True, "RISK PROFILE block missing end marker"

    values = {}
    for line in lines[start_idx + 1:end_idx]:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            return None, True, f"malformed RISK PROFILE line: {stripped!r}"
        key, value = stripped.split(":", 1)
        values[key.strip()] = value.strip()

    required = [
        "Market cap",
        "Failed lookup",
        "Earnings",
        "Risk branch",
        "Allocation",
        "Trailing",
        "Dynamic trailing",
        "TP",
        "SL",
    ]
    missing = [key for key in required if key not in values]
    if missing:
        return None, True, f"RISK PROFILE missing fields: {', '.join(missing)}"

    try:
        market_cap_text = values["Market cap"]
        market_cap = None if market_cap_text == "None" else int(market_cap_text)
        profile = {
            "market_cap": market_cap,
            "failed_lookup": _parse_bool_field(values["Failed lookup"]),
            "is_earnings": _parse_bool_field(values["Earnings"]),
            "risk_branch": values["Risk branch"],
            "allocation_percent": _parse_percent_field(values["Allocation"]),
            "trailing_percent": _parse_percent_field(values["Trailing"]),
            "dynamic_trail_percent": _parse_percent_field(values["Dynamic trailing"]),
            "limit_tp_percent": _parse_percent_field(values["TP"]),
            "limit_sl_percent": _parse_percent_field(values["SL"]),
        }
    except ValueError as exc:
        return None, True, f"RISK PROFILE parse failed: {exc}"

    return profile, True, None

def render_risk_profile_lines(profile):
    market_cap = profile.get("market_cap")
    return [
        RISK_PROFILE_START + "\n",
        f"Market cap: {market_cap if market_cap is not None else 'None'}\n",
        f"Failed lookup: {bool(profile.get('failed_lookup'))}\n",
        f"Earnings: {bool(profile.get('is_earnings'))}\n",
        f"Risk branch: {profile.get('risk_branch')}\n",
        f"Allocation: {_format_percent_for_profile(profile.get('allocation_percent', 0))}\n",
        f"Trailing: {_format_percent_for_profile(profile.get('trailing_percent', 0))}\n",
        f"Dynamic trailing: {_format_percent_for_profile(profile.get('dynamic_trail_percent', 0))}\n",
        f"TP: {_format_percent_for_profile(profile.get('limit_tp_percent', 0))}\n",
        f"SL: {_format_percent_for_profile(profile.get('limit_sl_percent', 0))}\n",
        RISK_PROFILE_END + "\n",
    ]

def write_risk_profile(filePath, profile):
    with open(filePath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    skip_profile = False
    for line in lines:
        stripped = line.strip()
        if stripped == RISK_PROFILE_START:
            skip_profile = True
            continue
        if skip_profile:
            if stripped == RISK_PROFILE_END:
                skip_profile = False
            continue
        new_lines.append(line)

    ensure_lines_end_with_newline(new_lines)
    new_lines.extend(render_risk_profile_lines(profile))
    atomic_write_lines(filePath, new_lines)

def get_trade_risk_profile(filePath, prefer_persisted=True, warn_on_fallback=False):
    with open(filePath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    context = {
        "risk_profile_found": False,
        "risk_profile_error": None,
        "fallback_used": False,
        "raw_market_cap_line": None,
        "malformed_market_cap_line": None,
        "source": None,
    }

    if prefer_persisted:
        persisted_profile, found, parse_error = parse_risk_profile_from_lines(lines, filePath)
        context["risk_profile_found"] = found
        context["risk_profile_error"] = parse_error

        if persisted_profile is not None:
            context["source"] = "RISK_PROFILE"
            return persisted_profile, context

        if found and parse_error:
            log_warning_and_print(
                f"[RISK_PROFILE][WARN] Persisted RISK PROFILE unusable; falling back to MARKETCAP parsing | file={filePath} | error={parse_error}"
            )

    isEarnings = any(line.strip() == "**EARNINGS REQUEST**" for line in lines)
    market_cap, failedLookup, raw_market_cap_line, malformed_market_cap_line = parse_market_cap_from_lines(
        lines,
        filePath=filePath,
        log_warnings=warn_on_fallback,
    )

    context.update({
        "fallback_used": True,
        "raw_market_cap_line": raw_market_cap_line,
        "malformed_market_cap_line": malformed_market_cap_line,
        "source": "MARKETCAP_FALLBACK",
    })

    if warn_on_fallback:
        log_warning_and_print(
            f"[RISK_PROFILE][WARN] Using MARKETCAP fallback risk calculation | file={filePath} | "
            f"risk_profile_found={context['risk_profile_found']} | risk_profile_error={context['risk_profile_error']} | "
            f"raw_market_cap_line={raw_market_cap_line} | malformed_market_cap_line={malformed_market_cap_line}"
        )

    return build_risk_profile(market_cap, failedLookup, isEarnings), context

def ensure_risk_profile(filePath):
    profile, context = get_trade_risk_profile(filePath, prefer_persisted=True, warn_on_fallback=False)
    if context["source"] != "RISK_PROFILE":
        write_risk_profile(filePath, profile)
        log_and_print(
            f"[RISK_PROFILE] Persisted entry risk profile | file={filePath} | "
            f"market_cap={profile['market_cap']} | failed_lookup={profile['failed_lookup']} | "
            f"earnings={profile['is_earnings']} | risk_branch={profile['risk_branch']} | "
            f"allocation={profile['allocation_percent']}% | trailing={profile['trailing_percent']}% | "
            f"dynamic_trailing={profile['dynamic_trail_percent']}% | tp={profile['limit_tp_percent']}% | sl={profile['limit_sl_percent']}%"
        )
        if profile["risk_branch"] == "FALLBACK":
            log_warning_and_print(
                f"[RISK_PROFILE][WARN] Entry risk profile is using FALLBACK | file={filePath} | failed_lookup={profile['failed_lookup']} | market_cap={profile['market_cap']}"
            )
    return profile

def return_val_from_marketCap(filePath, typeVal, warn_on_fallback=False):
    profile, _ = get_trade_risk_profile(filePath, prefer_persisted=True, warn_on_fallback=warn_on_fallback)

    if typeVal == "ALLOCATION":
        return profile["allocation_percent"] / 100
    elif typeVal == "TRAILING":
        return profile["trailing_percent"]
    elif typeVal == "DYNAMIC_TRAILING":
        return profile["dynamic_trail_percent"] / 100
    elif typeVal == "LIMIT":
        return profile["limit_tp_percent"], profile["limit_sl_percent"]
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

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
def is_market_open_sgt(closing_up):
    sgt = pytz.timezone(XANDER_TIMEZONE)
    now = datetime.now(sgt)

    adjustedHours = check_adjusted_trading_hrs()
    
    if (adjustedHours is not None and adjustedHours == "AM"):
        market_open_time = tradingOpen
        market_close_time = tradingHalfDay
    elif (adjustedHours is not None and adjustedHours == "PM"):
        market_open_time = tradingHalfDay
        market_close_time = tradingClose
    else:
        market_open_time = tradingOpen
        market_close_time = tradingClose

    if not closing_up:
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
def load_recent_embeddings(directory, hours=12):
    now = datetime.now()
    vecs = []
    texts = []
    ids = []
    timestamps = []
    file_count = 0  # ← counter for successful files

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                ts = datetime.fromisoformat(data['timestamp'])
                if (now - ts).total_seconds() <= hours * 3600:
                    vecs.append(np.array(data['embedding'], dtype='float32'))
                    texts.append(data['text'])
                    ids.append(data.get('id', ''))  # ← use .get() to avoid KeyError
                    timestamps.append(ts)
                    file_count += 1
            except Exception:
                continue  # skip bad files

    log_and_print(f"Found {file_count} valid file(s) in the last {hours} hour(s)")
    return np.array(vecs), texts, ids, timestamps
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def is_similar_post(new_post_text, directory=posts_path, threshold=0.75, self_match_window_sec=30):
    new_vec = MODEL.encode([new_post_text], normalize_embeddings=True)
    now = datetime.now()

    past_vecs, past_texts, past_ids, past_timestamps = load_recent_embeddings(directory)

    if len(past_vecs) == 0:
        return False, None, None, None

    # Collect all candidates (do not skip any based on text/timestamp here)
    filtered_vecs = np.array(past_vecs)
    filtered_texts = past_texts
    filtered_timestamps = past_timestamps

    # Build FAISS index and search for most similar post
    index = faiss.IndexFlatIP(new_vec.shape[1])
    index.add(filtered_vecs)

    D, I = index.search(new_vec, k=3)
    for i in range(len(D[0])):
        score = float(D[0][i])
        idx = int(I[0][i])
        match_text = filtered_texts[idx]
        match_time = filtered_timestamps[idx]
        time_diff = abs((now - match_time).total_seconds())

        if score >= threshold and time_diff < self_match_window_sec:
            log_and_print(f"Similarity candidate excluded within {self_match_window_sec}s self-match window (score={score:.2f}, time_diff={time_diff:.1f}s)")
            continue
        if score >= threshold:
            return True, score, match_text, match_time

    return False, None, None, None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def insert_tradeHistory(closed_directory, ticker, entry_price, exit_price, pnl):
    tradeHistory_path = os.path.join(functionsFolder_path, "GATEWAY_IBKR_TradeHistory.txt")

    # --- Read the closed post file ---
    try:
        with open(closed_directory, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        send_telegram_message_thread(
            f"❌ insert_tradeHistory: Failed to open file for {ticker}: {e}",
            "XANDER", 0, 0
        )
        return False

    # --- Look for Content, URL & TriggerDate of the Post ---
    has_content = False
    has_url = False
    has_fgi = False
    has_triggered_date = False

    # Extract Content
    content_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip first line (e.g. [@Benzinga])
        if i == 0:
            continue

        # Stop when dashed separator is reached
        if stripped.startswith("---"):
            break

        if stripped:
            content_lines.append(stripped)

    content = " ".join(content_lines).strip()
    if content:
        has_content = True

    # Extract URL & TriggeredDate
    url = ""
    fgi_value = ""
    triggered_date = ""

    for line in lines:
        stripped = line.strip()

        # **URL: https://...**
        if stripped.startswith("**URL: ") and stripped.endswith("**"):
            url = stripped[len("**URL: "):-2].strip()
            if url:
                has_url = True

        # **FGI: 100**
        elif stripped.startswith("**FGI: ") and stripped.endswith("**"):
            fgi_value = stripped[len("**FGI: "):-2].strip()
            if fgi_value:
                has_fgi = True

        # **TRIGGERED AT 2025-12-06 22:15:00**
        elif stripped.startswith("**TRIGGERED AT ") and stripped.endswith("**"):
            triggered_date = stripped[len("**TRIGGERED AT "):-2].strip()
            if triggered_date:
                has_triggered_date = True

        # If we have all required fields, no need to continue scanning
        if has_url and has_fgi and has_triggered_date:
            break

    # --- Validation ---
    if not (has_content and has_url and has_fgi and has_triggered_date):
        msg_parts = []
        if not has_content:
            msg_parts.append("Content")
        if not has_url:
            msg_parts.append("URL")
        if not has_fgi:
            msg_parts.append("FGI")
        if not has_triggered_date:
            msg_parts.append("TriggeredDate")

        missing_str = ", ".join(msg_parts) if msg_parts else "Unknown fields"
        send_telegram_message_thread(
            f"❌ insert_tradeHistory: Missing {missing_str} for {ticker}. Please check.",
            "XANDER", 0, 0
        )
        return False

    # --- Build JSONL row ---
    sold_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "Ticker": ticker,
        "TriggeredDate": triggered_date,   # from the file
        "SoldDate": sold_date,             # now
        "EntryPrice": float(entry_price),
        "ExitPrice": float(exit_price),
        "PNL": float(pnl),
        "URL": url,
        "FGI": fgi_value,
        "Content": content,
    }

    # Ensure directory exists in case functionsFolder_path is nested
    os.makedirs(os.path.dirname(tradeHistory_path), exist_ok=True)

    # --- Append to JSONL file ---
    try:
        with open(tradeHistory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        send_telegram_message_thread(
            f"❌ insert_tradeHistory: Failed to append to history for {ticker}: {e}",
            "XANDER", 0, 0
        )
        return False

    return True
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def place_limit_sell(ib, ticker, file_directory):
    try:
        risk_profile, risk_context = get_trade_risk_profile(file_directory, prefer_persisted=True, warn_on_fallback=True)
        limit_tp_p = risk_profile["limit_tp_percent"]
        limit_sl_p = risk_profile["limit_sl_percent"]
        tp_pct = limit_tp_p / 100
        sl_pct = limit_sl_p / 100

        sym = ticker.strip().upper()

        # Find current long position size and avg cost
        pos = next((p for p in ib.positions() if p.contract.symbol.strip().upper() == sym), None)
        if not pos or pos.position <= 0:
            return

        qty = int(round(pos.position))
        avg = float(pos.avgCost)

        # Check if TP/SL already exist
        open_statuses = {'PreSubmitted', 'Submitted', 'PendingSubmit'}

        active_trades = [
            tr for tr in ib.trades()
            if getattr(tr.contract, 'symbol', '').strip().upper() == sym
            and tr.order.action == 'SELL'
            and tr.order.orderType in ('LMT', 'STP', 'STP LMT')
            and tr.orderStatus.status in open_statuses
        ]

        existing_tp = any(tr.order.orderType == 'LMT' for tr in active_trades)
        existing_sl = any(tr.order.orderType in ('STP', 'STP LMT') for tr in active_trades)

        if existing_tp:
            send_telegram_message_thread(f"⚠️ Skipping TP for {sym}: already exists.", "XANDER", 0, 0)
        if existing_sl:
            send_telegram_message_thread(f"⚠️ Skipping SL for {sym}: already exists.", "XANDER", 0, 0)

        # Compute prices
        raw_tp_price = avg * (1 + tp_pct)
        raw_sl_price = avg * (1 - sl_pct)

        contract = Stock(sym, 'SMART', 'USD')
        try:
            minTick = ib.reqContractDetails(contract)[0].minTick or 0.01
        except Exception:
            minTick = 0.01

        tp_price = round(raw_tp_price / minTick) * minTick
        sl_price = round(raw_sl_price / minTick) * minTick

        oca = f"OCA_{sym}_{int(time_module.time())}"

        t_tp = t_sl = None

        # Place TP only if not existing
        if not existing_tp:
            takeProfit = LimitOrder(
                'SELL', qty, tp_price,
                tif='DAY', outsideRth=True,
                ocaGroup=oca, ocaType=1
            )
            t_tp = ib.placeOrder(contract, takeProfit)

        # Place SL only if not existing
        if not existing_sl:
            stopLoss = StopLimitOrder(
                'SELL', qty, sl_price, sl_price,
                tif='DAY', outsideRth=True,
                ocaGroup=oca, ocaType=1
            )
            t_sl = ib.placeOrder(contract, stopLoss)

        # Define valid statuses
        valid_statuses = {'Submitted', 'PreSubmitted'}

        # Poll for status updates (up to 15 seconds)
        for _ in range(15):
            tp_status = t_tp.orderStatus.status if t_tp else "Skipped"
            sl_status = t_sl.orderStatus.status if t_sl else "Skipped"
            if (
                (tp_status in valid_statuses or tp_status == "Skipped")
                and (sl_status in valid_statuses or sl_status == "Skipped")
            ):
                break
            ib.sleep(1.0)

        tp_order_id = t_tp.order.orderId if t_tp else "Skipped"
        sl_order_id = t_sl.order.orderId if t_sl else "Skipped"
        log_and_print(
            f"[RISK_PROFILE][TP_SL] file={file_directory} | "
            f"raw_market_cap_line={risk_context.get('raw_market_cap_line')} | "
            f"risk_profile_found={risk_context.get('risk_profile_found')} | "
            f"fallback_used={risk_context.get('fallback_used')} | "
            f"parsed_market_cap={risk_profile.get('market_cap')} | "
            f"failed_lookup={risk_profile.get('failed_lookup')} | "
            f"earnings={risk_profile.get('is_earnings')} | "
            f"risk_branch={risk_profile.get('risk_branch')} | "
            f"avg_cost={avg:.6f} | minTick={minTick} | "
            f"raw_tp_price={raw_tp_price:.6f} | rounded_tp_price={tp_price:.6f} | "
            f"raw_sl_price={raw_sl_price:.6f} | rounded_sl_price={sl_price:.6f} | "
            f"tp_order_id={tp_order_id} | sl_order_id={sl_order_id} | "
            f"tp_status={tp_status} | sl_status={sl_status}"
        )

        # Final checks
        if t_tp:
            tp_status = t_tp.orderStatus.status
            if tp_status in valid_statuses:
                send_telegram_message_thread(f"📤 TP ORDER for {qty} of {sym} placed successfully ({limit_tp_p}%)", "XANDER", 0, 0)
            else:
                send_telegram_message_thread(f"🚨 Request on TP ORDER for {qty} of {sym} was not accepted by IBKR ({tp_status})", "XANDER", 0, 0)
                if tp_status != "PendingSubmit":
                    ib.cancelOrder(t_tp.order)

        if t_sl:
            sl_status = t_sl.orderStatus.status
            if sl_status in valid_statuses:
                send_telegram_message_thread(f"📤 SL ORDER for {qty} of {sym} placed successfully ({limit_sl_p}%)", "XANDER", 0, 0)
            else:
                send_telegram_message_thread(f"🚨 Request on SL ORDER for {qty} of {sym} was not accepted by IBKR ({sl_status})", "XANDER", 0, 0)
                if sl_status != "PendingSubmit":
                    ib.cancelOrder(t_sl.order)

        if (t_tp and tp_status == "PendingSubmit") or (t_sl and sl_status == "PendingSubmit"):
            get_last_market_execution()

    except Exception as e:
        ibkrautomate_logger.info(e)
        raise ValueError("Error occurred placing LIMIT sell.")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def preprocess_and_queue_signal(file_path):
    start_time = time_module.time()  # Start timer
    filename = os.path.basename(file_path)
    filename_log = filename
    temp_path = None  # Define early to avoid UnboundLocalError
    post_content = None  # Define early to avoid scope issue
    toIgnore = False
    ignore_reason_code = None
    ignore_detail = None

    try:
        # Early validation: must be .txt, have 4 parts and 3 "_"
        if not filename.endswith('.txt'):
            raise Exception("File is not a .txt")

        name_part = os.path.splitext(filename)[0]  # removes '.txt'
        parts = name_part.split('_')

        if len(parts) != 5:
            raise Exception("Filename does not contain 5 parts")

        '''if len(parts) != 5 :
            raise Exception("Filename does not contain 5 parts")
        if filename.count('_') != 4:
            raise Exception("Filename does not contain 4x '_'")'''

        # Move to temp to avoid double processing
        temp_path = file_path + ".validating"
        try:
            shutil.move(file_path, temp_path)
            rename_msg = f"{filename} renamed to {filename}.validating"
            log_and_print(rename_msg)
            filename_log = filename + ".validating"
        except FileNotFoundError:
            return  # Already picked up by another thread

        parsed = parse_signal_filename(filename)
        bot_name = parsed["bot_name"]
        action = parsed["action"].upper()
        ticker = parsed["ticker"].upper()
        unique_id = parsed["unique_id"]
        reprocess = parsed["reprocess"]

        if action not in ['BUY']:
            toIgnore = True
            ignore_reason_code = "NON_BUY_ACTION"
            ignore_detail = f"action={action}"

        # For EARNINGS only, check if market cap suffice
        #if is_market_open_sgt(False):

        # Check if ticker trade already taken
        filePath_SocialMarket_Trades = os.path.join(functionsFolder_path, "SOCIALMARKET_Trades.txt")
        with open(filePath_SocialMarket_Trades, 'r') as f:
            tradesTaken = f.read().strip()

        if tradesTaken != "":
            tradeList = [t.strip() for t in tradesTaken.split(",") if t.strip() != ""]
            if ticker in tradeList:
                send_telegram_message_thread("⚠️ Skipping: Ticker has already been processed for today's session.", "XANDER", 0, 0)
                toIgnore = True
                ignore_reason_code = "TICKER_ALREADY_PROCESSED"
                ignore_detail = f"ticker={ticker} already exists in SOCIALMARKET_Trades.txt"

        # Check for duplicate context
        if not toIgnore and int(reprocess) == 0:
            post_content = safe_read_text(temp_path).strip()
            ibkrautomate_logger.info(f"Conducting similarity check for: {filename_log}")
            is_old_context, similarity_score, matched_post, matched_time = is_similar_post(post_content)
            ibkrautomate_logger.info(f"Similarity Found: {is_old_context}")
            
            if is_old_context:
                duration = round(time_module.time() - start_time, 3)
                old_context_msg = (
                    f"⚠️ Similar Post Detected\n"
                    f"⏱️ Speed: {duration}s\n"
                    f"🧐 Score: {similarity_score:.4f}\n"
                    f"🕰️ Timestamp: {matched_time.strftime('%Y-%m-%d %H:%M')}\n"
                    f"Matched Post:\n{matched_post}"
                )
                send_telegram_message_thread(old_context_msg, "XANDER", 0, 0)
                toIgnore = True
                ignore_reason_code = "SIMILAR_POST"
                ignore_detail = f"score={similarity_score:.4f}; matched_time={matched_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        if has_open_order(filename):
            toIgnore = True
            if ignore_reason_code is None:
                ignore_reason_code = "OPEN_ORDER_EXISTS"
                ignore_detail = f"open order file already exists for ticker={ticker}"

        # Valid signal — send to queue
        if not toIgnore:
            ibkrautomate_logger.info("Sending to queue to dispatch IBKR order..")
            # dispatch_ibkr_signal will carry on
            order_queue.put({
                "temp_path": temp_path,
                "bot_name": bot_name,
                "action": action,
                "ticker": ticker,
                "post_content": post_content,
                "original_filename": filename
            })
            if ticker != "SPXL":
                with open(filePath_SocialMarket_Trades, 'a') as f:
                    f.write(f"{ticker},")

        # Invalid signal — move to ignored
        else:
            move_to_ignored(
                temp_path,
                ignore_reason_code or "VALIDATION_REJECTED",
                detail=ignore_detail,
                original_filename=filename,
                destination_filename=filename,
            )

    except Exception as e:
        err_msg = f"Error processing {filename}: {e}"
        log_and_print(err_msg)
        send_telegram_message_thread(err_msg, "XANDER", 0, 0)
        if temp_path and os.path.exists(temp_path):
            try:
                shutil.move(temp_path, unique_destination_path(error_directory, filename))
                if (filename_log.endswith(".validating")):
                    log_and_print(f"{filename_log} renamed to {filename}")
                log_and_print(f"{filename} sent to {error_directory}")
            except OSError as move_error:
                log_and_print(f"[VALIDATING][ERROR] Failed to move {filename_log} to {error_directory}: {move_error}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def dispatch_ibkr_signal():

    asyncio.set_event_loop(asyncio.new_event_loop())

    while True:

        order_data = order_queue.get()

        try:

            required_keys = ["ticker", "action", "bot_name", "temp_path", "original_filename"]
            for key in required_keys:
                if key not in order_data:
                    raise KeyError(f"Missing key: '{key}' not found in queue provided.")

            safe_ibkr_connect(False)
    
            # Check if there is duplicate tickers in incoming directory (.validating)
            duplicate_tickers = [
                f for f in os.listdir(incoming_directory)
                if f.endswith(".validating") and order_data["ticker"].upper() in f.upper() and order_data["original_filename"] not in f
            ]
                
            # Check if it is EARNINGS
            with open(order_data["temp_path"], "r", encoding="utf-8") as f:
                lines = f.readlines()

            isEarnings = False

            # Look for existing **EARNINGS REQUEST** line
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped == "**EARNINGS REQUEST**":
                    isEarnings = True
                    break

            # Check again if there is existing trade for same ticker
            positions = ib.positions()
            live_position_symbols = {
                p.contract.symbol.strip().upper(): int(round(p.position)) for p in positions
            }

            position_size = live_position_symbols.get(order_data['ticker'], 0)

            if position_size == 0 and not duplicate_tickers:

                etfs = ["TQQQ", "SOXL", "ERX", "FAS", "CURE", "WANT", "DFEN", "DUSL", "DRN"]

                lot = 0
                latest_price = 0
                funds_to_use = 0

                isWithinMarketHrs = is_market_open_sgt(False)

                contract = Stock(order_data["ticker"], "SMART", "USD")

                try:
                    if isWithinMarketHrs:
                        raise ValueError()
                    
                    result = subprocess.run(
                        ["python", "IBKR_MD.py", order_data["ticker"]],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        cwd=ibkrFolder_path
                    )
                    
                    if result.returncode == 0:
                        t_live = json.loads(result.stdout)
                        ask_price = t_live['ask']
                        bid_price = t_live['bid']
                        log_and_print(t_live)

                        if (
                            ask_price is None or bid_price is None
                            or not isinstance(ask_price, (int, float))
                            or not isinstance(bid_price, (int, float))
                            or math.isnan(ask_price) or math.isnan(bid_price)
                        ):
                            raise ValueError("Missing or invalid ask/bid prices.")

                        # --- FIX #1: 2% SPREAD CHECK ---
                        mid_price = (ask_price + bid_price) / 2
                        spread_pct = abs(ask_price - bid_price) / mid_price

                        if spread_pct > 0.02:
                            raise ValueError(
                                f"⚠️ Request on BUY for {order_data['ticker']} SKIPPED: "
                                f"Spread {spread_pct*100:.2f}% exceeds 2%."
                            )

                        min_tick = get_contract_min_tick(contract)
                        latest_price = round_price_to_tick(ask_price, min_tick, "up")
                        log_and_print(
                            f"[IBKR][BUY_QUOTE] {order_data['ticker']} bid={bid_price} ask={ask_price} "
                            f"mid={round(mid_price, 6)} spread={spread_pct*100:.2f}% minTick={min_tick} "
                            f"submittedLimit={latest_price}"
                        )

                    else:
                        helper_stdout = (result.stdout or "").strip()
                        helper_error = helper_stdout or f"market data helper exited with returncode {result.returncode}"
                        log_and_print(
                            f"[IBKR][BUY_QUOTE_ERROR] {order_data['ticker']} "
                            f"returncode={result.returncode} output={helper_error!r}"
                        )
                        raise ValueError(f"IBKR market data helper failed: {helper_error}")

                except Exception as e:
                    # keep custom message
                    custom_msg = str(e)

                    if isWithinMarketHrs:
                        # Yahoo fallback
                        try:
                            yticker = yf.Ticker(order_data['ticker'])
                            price_data = yticker.history(period="1d")
                            if not price_data.empty:
                                latest_price = price_data["Close"].iloc[-1]
                            else:
                                lot = 10
                        except Exception:
                            lot = 10
                    else:
                        raise ValueError(custom_msg)

                # Get available funds
                base_balance = fetch_base_balance() 
                avail_funds = fetch_avail_balance()

                if order_data['ticker'] == "SPXL" or order_data['ticker'] == "TQQQ":
                    allocation = 0.5 # 50%
                    funds_to_use = base_balance * allocation
                elif order_data['ticker'] in etfs:
                    allocation = 0.35 # 35%
                    funds_to_use = base_balance * allocation
                else:
                    risk_profile = ensure_risk_profile(order_data["temp_path"])
                    allocation = risk_profile["allocation_percent"] / 100
                    if allocation == 0:
                        raise ValueError(f"⚠️ Request on BUY for {order_data['ticker']} SKIPPED: Insufficient market cap requirement to place order.")
                    funds_to_use = base_balance * allocation

                # Calculate lot size if price is fetched
                if latest_price > 0:
                    lot = math.floor(funds_to_use / latest_price)

                if lot < 1:
                    raise ValueError(f"⚠️ Request on BUY for {order_data['ticker']} SKIPPED: Allocated amount below minimum lot size.")

                # Revise amount to be used with fees considered
                approx_fund_amount = lot * latest_price

                if 100 <= approx_fund_amount <= 500:
                    fee_percent = 0.01  # 1%
                elif 500 < approx_fund_amount <= 2000:
                    fee_percent = 0.003  # 0.3%
                elif 2000 < approx_fund_amount <= 10000:
                    fee_percent = 0.0015  # 0.15%
                else:  
                    fee_percent = 0.0005  # 0.05%

                if approx_fund_amount < 100:
                    funds_to_use_with_fees = round(approx_fund_amount + 1.00, 2)
                else:
                    funds_to_use_with_fees = round(approx_fund_amount * (1 + fee_percent), 2)
                

                if (funds_to_use_with_fees * 1.05) < avail_funds: # Extra 5% leeway

                    # Step 1: Create BUY order
                    if isWithinMarketHrs:
                        buy_order = Order(
                            action='BUY',
                            orderType='MKT',
                            totalQuantity=lot,
                            transmit=True,
                            outsideRth=True
                        )

                    # Step 1: Create LIMIT order
                    else:
                        buy_order = LimitOrder(
                            action='BUY',
                            totalQuantity=lot,
                            lmtPrice=latest_price,
                            tif='DAY',
                            outsideRth=True
                        )

                    # Place BUY order and get Trade object
                    buy_trade = ib.placeOrder(contract, buy_order)
                    ib.sleep(1)

                    status = buy_trade.orderStatus.status
                    
                    duration = round(time_module.time() - os.path.getmtime(order_data["temp_path"]), 3)

                    if status.lower() not in {"inactive", "cancelled", "apicancelled", "pendingcancel"}:

                        # Step 2: Notify of order submission
                        if isWithinMarketHrs:
                            log_and_print(f"Order Placed --> ID: {buy_order.orderId} | {order_data['action']} {lot} {order_data['ticker']} | Allocation: {allocation * 100}% | Status: {status}")
                        else:
                            log_and_print(f"Order Placed --> ID: {buy_order.orderId} | {order_data['action']} {lot} {order_data['ticker']} | Limit: {latest_price} | Allocation: {allocation * 100}% | Status: {status}")

                        if isWithinMarketHrs:
                            order_placed_msg = (
                                f"🚨 ORDER PLACED\n"
                                f"ID: {buy_order.orderId}\nAction: {order_data['action']}\nTicker: {order_data['ticker']}\nLot: {lot}\nAllocation: {allocation * 100}%\nStatus: {status}\n"
                                f"⏱️ Speed: {duration}s"
                            )
                        else:
                            order_placed_msg = (
                                f"🚨 ORDER PLACED\n"
                                f"ID: {buy_order.orderId}\nAction: {order_data['action']}\nTicker: {order_data['ticker']}\nLot: {lot}\nPrice: {latest_price}\nAllocation: {allocation * 100}%\nStatus: {status}\n"
                                f"⏱️ Speed: {duration}s"
                            )
                            

                        send_telegram_message_thread(order_placed_msg, "XANDER", 0, 0)

                        # Rename filename with Lot size and Order id
                        base_name = os.path.basename(order_data["temp_path"])  # "BOT_BUY_NVDA_JCW7GL86_0.txt.validating"

                        name_without_ext, ext1 = os.path.splitext(base_name)  # -> "BOT_BUY_NVDA_JCW7GL86_0.txt", ".validating"

                        name_core, ext2 = os.path.splitext(name_without_ext)  # -> "BOT_BUY_NVDA_JCW7GL86_0", ".txt"

                        parts = name_core.split('_')  # ['BOT', 'BUY', 'NVDA', 'JCW7GL86', '0]
                        parts.insert(2, str(lot))  # insert after BUY: ['BOT', 'BUY', '100', 'NVDA', 'JCW7GL86', '0']
                        parts[-2] = str(buy_order.orderId)  # replace unique id with order id: ['BOT', 'BUY', '100', 'NVDA', '12345', '0']

                        new_name = '_'.join(parts) + ext2 + ".processing"  # -> "BOT_BUY_100_NVDA_12345_0.txt.processing"

                        log_and_print(f'{base_name} renamed to {new_name}')
                        
                        # monitor_order_fill will carry on
                        new_temp_path = os.path.join(incoming_directory, new_name)
                        shutil.move(order_data["temp_path"], new_temp_path)
                        os.utime(new_temp_path, (time_module.time(), time_module.time()))

                    else:
                        if buy_trade.orderStatus.status.lower() == "inactive":
                            ib.cancelOrder(buy_order)
                        raise ValueError(f"🚨 Request on BUY for {lot} of {contract.symbol} was not accepted by IBKR ({status})")

                else:
                    raise ValueError(f"⚠️ Request on BUY for {order_data['ticker']} SKIPPED: Insufficient balance relative to allocated trade amount.")
        
            else:

                order_exist_msg = f"{order_data['action']} order for {order_data['ticker']} already exists — duplicate submission skipped."
                log_and_print(order_exist_msg)
                send_telegram_message_thread(order_exist_msg, "XANDER", 0, 0)
                
                move_to_ignored(
                    order_data["temp_path"],
                    "DUPLICATE_OR_POSITION_EXISTS",
                    detail=f"position_size={position_size}; duplicate_validating_files={duplicate_tickers}",
                    original_filename=order_data["original_filename"],
                    destination_filename=order_data["original_filename"],
                )

        except Exception as e: 
            original_filename = order_data.get("original_filename", "UNKNOWN")
            temp_path = order_data.get("temp_path")
            err_msg = f"ORDER Creation Failed for {original_filename}: {e}"
            log_and_print(err_msg)
            send_telegram_message_thread(err_msg, "XANDER", 0, 0)
            if temp_path and os.path.exists(temp_path):
                try:
                    err_text = str(e) or ""
                    is_skip = isinstance(e, ValueError) and "SKIPPED:" in err_text
                    destination_dir = ignored_directory if is_skip else error_directory
                    dest_path = unique_destination_path(destination_dir, original_filename)
                    if is_skip:
                        move_to_ignored(
                            temp_path,
                            "ORDER_SKIPPED",
                            detail=err_text,
                            exc=e,
                            original_filename=original_filename,
                            destination_filename=original_filename,
                        )
                    else:
                        shutil.move(temp_path, dest_path)
                        log_and_print(f'{os.path.basename(temp_path)} renamed to {original_filename}')
                        log_and_print(f'{os.path.basename(original_filename)} sent to {destination_dir}')
                except OSError as move_error:
                    log_and_print(f"[VALIDATING][ERROR] Failed to move {os.path.basename(temp_path)} to {destination_dir}: {move_error}")
        finally:
            order_queue.task_done()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ORDER MONITORING / AUDIT HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def audit():
    
    safe_ibkr_connect(True)

    # AUDIT 1: Check open positions (Audit IBKR open orders vs local tracking)
    portfolio = ib.portfolio()
    max_age_seconds = 12 * 60 * 60  # 12 hours

    if portfolio:
        for p in portfolio:

            ticker_opened = (p.contract.symbol or "").strip()
            if not ticker_opened:
                continue

            bought_price = float(getattr(p, "marketPrice", 0.0) or 0.0)
            if bought_price <= 0:
                continue

            ticker_upper = ticker_opened.upper()

            def is_recent(filepath: str) -> bool:
                try:
                    mtime = os.path.getmtime(filepath)
                    return (time_module.time() - mtime) <= max_age_seconds
                except OSError:
                    return False

            def is_valid_signal_filename(filename: str) -> bool:
                # Example: XANDER_BUY_2_SMX_3055_0
                parts = filename.split("_")
                if len(parts) < 5:
                    return False
                return parts[2].isdigit() and parts[4].isdigit()

            def find_matching_files(directory: str) -> list[str]:
                matches = []
                try:
                    for name in os.listdir(directory):
                        full = os.path.join(directory, name)
                        if not os.path.isfile(full):
                            continue
                        if ticker_upper not in name.upper():
                            continue
                        if not is_valid_signal_filename(name):
                            continue
                        if not is_recent(full):
                            continue
                        matches.append(full)
                except FileNotFoundError:
                    pass
                return matches

            def append_bought_at_if_missing(filepath: str) -> None:
                marker = "**BOUGHT AT"
                line_to_add = f"**BOUGHT AT {bought_price:.2f}**"

                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    if marker in content:
                        return

                    safe_append_line(filepath, line_to_add)

                    log_and_print(f"[AUDIT] Appended BOUGHT AT price for {ticker_upper}: {bought_price:.2f}")

                except OSError:
                    log_and_print(f"[AUDIT][WARN] Failed to append BOUGHT AT for {ticker_upper}")
                    return

            # (1) If file exists in OPEN, do nothing
            open_matches = find_matching_files(open_directory)
            if open_matches:
                continue

            # (2) Otherwise, search IGNORED
            ignored_matches = find_matching_files(ignored_directory)
            if not ignored_matches:
                continue

            # Pick most recently modified file if multiple
            ignored_matches.sort(key=lambda fp: os.path.getmtime(fp), reverse=True)
            src = ignored_matches[0]

            log_and_print(f"[AUDIT] Restoring {ticker_upper} signal from IGNORED → OPEN")

            # (3) Append bought line if missing, then move it back to OPEN
            append_bought_at_if_missing(src)

            dst = os.path.join(open_directory, os.path.basename(src))
            try:
                shutil.move(src, dst)
                log_and_print(f"[AUDIT] Moved signal file back to OPEN for {ticker_upper}")
            except OSError:
                log_and_print(f"[AUDIT][WARN] Failed to move signal file for {ticker_upper}")
                continue

    # AUDIT 2: Verify open/pending sell orders are not missing
    isWithinMarketHrs = is_market_open_sgt(False)

    open_trades = [t for t in ib.trades() if t.orderStatus.status not in ("Filled", "Cancelled")]

    ticker_to_types = {}
    ticker_to_valid_trail = {}

    for t in open_trades:
        try:
            sym = (t.contract.symbol or "").strip().upper()
            o = t.order
            order_type = (o.orderType or "").upper().strip()

            if not sym or not order_type:
                continue

            ticker_to_types.setdefault(sym, set()).add(order_type)

            if order_type == "TRAIL":
                valid_trail = (
                    (o.auxPrice is not None and o.auxPrice > 0) or
                    (o.trailStopPrice is not None and o.trailStopPrice > 0)
                )
                ticker_to_valid_trail[sym] = valid_trail

        except Exception:
            continue

    tickers_missing = []

    if portfolio:
        for p in portfolio:
            sym = (p.contract.symbol or "").strip().upper()
            if not sym:
                continue

            types_present = ticker_to_types.get(sym, set())

            if isWithinMarketHrs:
                has_valid_trail = ticker_to_valid_trail.get(sym, False)
                if not has_valid_trail:
                    tickers_missing.append(sym)
            else:
                if not (("LMT" in types_present) and ("STP LMT" in types_present)):
                    tickers_missing.append(sym)

    # Deduplicate
    seen = set()
    tickers_missing = [x for x in tickers_missing if not (x in seen or seen.add(x))]

    if tickers_missing:
        tickers_listed = ", ".join(tickers_missing)

        file_name_IBKR_ReplaceSells = "GATEWAY_REPLACE_SELLS_IBKR.txt"
        file_path_IBKR_ReplaceSells = os.path.join(functionsFolder_path, file_name_IBKR_ReplaceSells)

        try:
            with open(file_path_IBKR_ReplaceSells, "w", encoding="utf-8") as file:
                file.write(tickers_listed)

            log_and_print(
                f"[AUDIT][ACTION] Missing sell protection detected → "
                f"queued replacement for: {tickers_listed}"
            )
        except OSError:
            log_and_print("[AUDIT][ERROR] Failed to write GATEWAY_REPLACE_SELLS file")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def monitor_order_fill():

    if not any(f.endswith(".processing") for f in os.listdir(incoming_directory)):
        return  # No files to process

    safe_ibkr_connect(True)
    
    isWithinMarketHrs = is_market_open_sgt(False)
    
    ib.reqPositions() 
    ib.sleep(0.5)

    # Build a set of live order IDs
    positions = ib.positions()
    live_position_symbols = {
        p.contract.symbol.strip().upper(): int(round(p.position)) for p in positions
    }

    looping_filename = None

    for filename in os.listdir(incoming_directory):

        if filename.endswith(".processing"):

            looping_filename = filename

            try:

                parsed = parse_signal_filename(filename)
                order_id = parsed["order_id"]
                ticker = parsed["ticker"]
                lotSize = parsed["lot"]
                action = parsed["action"]
                botName = parsed["bot_name"]

                if order_id is None or lotSize is None:
                    raise ValueError(f"Invalid filename format: Expected format BOT_ACTION_LOTSIZE_TICKER_ORDERID_REPROCESS.txt")
                    
                # Define timezone
                sgt = pytz.timezone(XANDER_TIMEZONE)

                position_size = live_position_symbols.get(ticker, 0)

                trade = next((t for t in ib.trades() if str(t.order.orderId) == order_id), None)

                # Process Order FILLED
                if ((int(position_size) == int(lotSize)) or (trade and trade.orderStatus.status == "Filled")) and isWithinMarketHrs:

                    log_and_print(f"Order Filled --> ID: {order_id} | {action} {lotSize} {ticker}")

                    # The IBKR data feed is not updating reliably, executing a restart to restore accuracy
                    if (int(position_size) != int(lotSize) and trade.orderStatus.status == "Filled"):
                        forceStopMsg = "⚠️ The IBKR data feed is not updating reliably, executing a restart to restore accuracy."
                        send_telegram_message_thread(forceStopMsg, "XANDER", 0, 0)
                        file_name_IBKR_ForceRestart = "IBKR_FORCE_RESTART.txt"
                        target_path = os.path.join(functionsFolder_path, file_name_IBKR_ForceRestart)
                        atomic_write_text(target_path, "")
                        raise SystemExit(forceStopMsg) 

                    msg = f"📦 ORDER UPDATED ({order_id})\n"

                    executions = ib.reqExecutions()
                    ib.sleep(0.5)

                    # Filter manually by order ID
                    order_id_int = int(order_id)
                    filtered_executions = [f for f in executions if f.execution.orderId == order_id_int]

                    if not filtered_executions:
                        raise ConnectionError(f"Order filled but not able to retrieve AVG Fill Price.")

                    fill = filtered_executions[0]
                    avg_fill_price = fill.execution.avgPrice
                    contract = fill.contract

                    etfs = ["TQQQ", "SOXL", "ERX", "FAS", "CURE", "WANT", "DFEN", "DUSL", "DRN"]

                    # Define trailing stop parameter
                    if ticker == "SPXL" or ticker == "TQQQ":
                        trailing_percent = 0.35  # 0.35% TRAIL STOP LOSS

                    elif ticker in etfs:
                        trailing_percent = 0.5  # 0.5% TRAIL STOP LOSS

                    else:
                        trailing_percent = return_val_from_marketCap(os.path.join(incoming_directory, looping_filename), "TRAILING", warn_on_fallback=True)

                    # Create and place trailing stop SELL order
                    contract = contract
                    contract.exchange = "SMART"
                    trailing_stop = Order(
                        action="SELL",
                        orderType="TRAIL",
                        totalQuantity=lotSize,
                        trailingPercent=trailing_percent,
                        outsideRth=True,
                        transmit=True
                    )

                    trail_trade = ib.placeOrder(contract, trailing_stop)
                    ib.sleep(0.5) 

                    status = trail_trade.orderStatus.status

                    # Convert IBKR fill time (UTC) → SGT
                    fill_time = fill.execution.time.astimezone(sgt)
                    formatted_fill_time = fill_time.strftime('%Y-%m-%d %H:%M:%S')

                    # Get modified time of the file and localize to SGT
                    processing_file_path = os.path.join(incoming_directory, looping_filename)
                    file_mod_time = datetime.fromtimestamp(os.path.getmtime(processing_file_path))
                    file_mod_time = sgt.localize(file_mod_time)

                    # Compute duration
                    time_to_fill = fill_time - file_mod_time
                    seconds_to_fill = max(0, int(time_to_fill.total_seconds()))

                    # Add to message
                    msg += "✅ Order filled successfully.\n"
                    msg += f"💰 Avg Fill Price: ${avg_fill_price:.2f}\n"
                    msg += f"📅 Fill Time: {formatted_fill_time}\n"
                    msg += f"⏱️ Duration: {seconds_to_fill}s"

                    if status.lower() in {"presubmitted", "submitted"}:
                        msg += f"\n📌 Trailing ({trailing_percent}%) stop order placed successfully"
                    else:
                        msg += f"\n⚠️ Trailing stop placement issue ({status})"
                        log_and_print(f"Trailing ({trailing_percent}%) stop placement issue for {ticker} — Status: {status}")

                    send_telegram_message_thread(msg, "XANDER", 0, 0)

                    # Add average fill price into file content
                    fill_line = f"**BOUGHT AT {avg_fill_price:.2f}**"
                    file_to_modify = os.path.join(incoming_directory, looping_filename)

                    with open(file_to_modify, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    set_marker_line(lines, "**BOUGHT AT ", f"{avg_fill_price:.2f}")

                    if isWithinMarketHrs:
                        append_marker_line(lines, "**TRADE ADJUSTED TO MARKET**")

                    atomic_write_lines(file_to_modify, [str(line) for line in lines])

                    # Update available balance
                    total_cost = avg_fill_price * int(lotSize)
                    respawn_avail_balance(total_cost, "BUY")
                        
                    filename_adjusted = looping_filename.replace(".processing","")
                    new_path = os.path.join(open_directory, filename_adjusted)
                    shutil.move(os.path.join(incoming_directory, looping_filename), new_path)

                    log_and_print(f"{looping_filename} renamed to {filename_adjusted}")
                    log_and_print(f"{filename_adjusted} sent to {open_directory}")
                        
                else:

                    # Get current time and file's modified time
                    now = time_module.time()
                    last_modified = os.path.getmtime(os.path.join(incoming_directory, filename))

                    # Check if file was last modified more than 10 seconds ago, very UNLIKELY to occur
                    if isWithinMarketHrs and (now - last_modified) > 10:
                    
                        issue_msg = f"🚨 Market Buy Delay\n📉 {ticker} | 🆔 {order_id}\n⏱️ Unfilled after 10s — Initiating Cancel.."
                        send_telegram_message_thread(issue_msg, "XANDER", 1000, 0)
                        cancel_all_orders([ticker])

                    elif not isWithinMarketHrs and (((now - last_modified) >= 35) or (int(position_size) == int(lotSize))):

                        sym = ticker.strip().upper()

                        # 1) Cancel any still-active parent orders (unfilled or partially filled)
                        for t in ib.trades():
                            if getattr(t.contract, 'symbol', '').strip().upper() != sym:
                                continue
                            if getattr(t.order, 'parentId', 0):  # skip TP/SL children
                                continue

                            st = (t.orderStatus.status or '').strip()
                            rem = t.orderStatus.remaining or 0

                            # Cancel if not already filled/cancelled/inactive
                            if st not in ('Filled', 'Cancelled', 'Inactive'):
                                if order_id is None or t.order.orderId == order_id:  # optional single-target filter
                                    ib.cancelOrder(t.order)

                        # Ensure IBKR processes cancel requests
                        ib.sleep(1)

                        # 2) Compute aggregate fill metrics for this ticker (parents only)
                        parents = [tr for tr in ib.trades()
                                if getattr(tr.contract, 'symbol', '').strip().upper() == sym
                                and not getattr(tr.order, 'parentId', 0)]

                        total_qty = sum((tr.order.totalQuantity or 0) for tr in parents) or 0
                        filled_qty = sum((tr.orderStatus.filled or 0) for tr in parents) or 0
                        remaining_qty = sum((tr.orderStatus.remaining or 0) for tr in parents) or 0
                        filled_pct = round(100 * filled_qty / total_qty, 2) if total_qty else 0.0

                        # Weighted average fill price + latest fill time across executions
                        exec_shares = 0
                        exec_value = 0.0
                        latest_exec_dt = None
                        for tr in parents:
                            for f in tr.fills:
                                sh = getattr(f.execution, 'shares', 0) or 0
                                px = getattr(f.execution, 'price', None)
                                if px is not None and sh:
                                    exec_shares += sh
                                    exec_value += sh * px
                                et = getattr(f.execution, 'time', None)  # tz-aware (UTC) in ib_insync
                                if et and (latest_exec_dt is None or et > latest_exec_dt):
                                    latest_exec_dt = et

                        avg_fill_price = (exec_value / exec_shares) if exec_shares else None
                        formatted_fill_time = 'N.A.'
                        if latest_exec_dt:
                            formatted_fill_time = latest_exec_dt.astimezone(sgt).strftime('%Y-%m-%d %H:%M:%S')

                        # 3) Duration since the file started processing (SGT)
                        processing_file_path = os.path.join(incoming_directory, looping_filename)
                        file_mod_time = datetime.fromtimestamp(os.path.getmtime(processing_file_path), tz=sgt)
                        end_time = latest_exec_dt.astimezone(sgt) if latest_exec_dt else datetime.now(tz=sgt)
                        seconds_to_fill = max(0, int((end_time - file_mod_time).total_seconds()))

                        # 4) Telegram message
                        if filled_qty == 0:
                            status_line = "❌ Unfilled after 30s — Order canceled."
                        elif (filled_qty < int(lotSize)):
                            status_line = f"🟡 Partially filled after 30s — remaining {int(remaining_qty)} canceled."
                        else:
                            status_line = "✅ Fully filled."

                        if filled_qty == 0 or (filled_qty < int(lotSize)):

                            open_orders = ib.openOrders()

                            if open_orders:
                                # get the order with the highest orderId (usually the latest)
                                latest_order = max(open_orders, key=lambda o: o.orderId)
                                ib.cancelOrder(latest_order)
                                send_telegram_message_thread(f"🔚 Cancelled existing LIMIT BUY for {ticker}\n", "XANDER", 0, 0)
                            else:
                                send_telegram_message_thread(f"⚠️ Unable to cancel existing LIMIT BUY for {ticker}\n", "XANDER", 0, 0)


                        msg = []
                        msg.append(f"📦 ORDER STATUS ({order_id})")
                        msg.append(status_line)
                        msg.append(f"🧮 Filled: {int(filled_qty)}/{int(total_qty)} ({filled_pct}%)")
                        if avg_fill_price is not None:
                            msg.append(f"💰 Avg Fill Price: ${avg_fill_price:.2f}")
                        if formatted_fill_time != 'N.A.':
                            msg.append(f"📅 Fill Time: {formatted_fill_time}")
                            msg.append(f"⏱️ Duration: {seconds_to_fill}s")

                        send_telegram_message_thread("\n".join(msg), "XANDER", 0, 0)
                        log_and_print(f"Premarket Order Status for {sym} --> {status_line[1:].strip()}")

                        if "Unfilled" not in status_line:
                            # Add average fill price into file content
                            fill_line = f"**BOUGHT AT {avg_fill_price:.2f}**"
                            file_to_modify = os.path.join(incoming_directory, looping_filename)

                            with open(file_to_modify, "r", encoding="utf-8") as f:
                                lines = f.readlines()

                            set_marker_line(lines, "**BOUGHT AT ", f"{avg_fill_price:.2f}")
                            lines = [str(line) for line in lines]

                            atomic_write_lines(file_to_modify, lines)

                            # Update available balance
                            total_cost = avg_fill_price * int(lotSize)
                            respawn_avail_balance(total_cost, "BUY")

                            place_limit_sell(ib, sym, os.path.join(incoming_directory, looping_filename))
                            
                        filename_adjusted = looping_filename.replace(".processing","")

                        if "Unfilled" in status_line:
                            move_to_ignored(
                                os.path.join(incoming_directory, looping_filename),
                                "ORDER_UNFILLED_CANCELLED",
                                detail=status_line,
                                original_filename=filename_adjusted,
                                destination_filename=filename_adjusted,
                            )
                            continue
                        else:
                            new_path = os.path.join(open_directory, filename_adjusted)

                        shutil.move(os.path.join(incoming_directory, looping_filename), new_path)

                        log_and_print(f"{looping_filename} renamed to {filename_adjusted}")
                        log_and_print(f"{filename_adjusted} sent to {open_directory}")

            except FileNotFoundError:
                log_and_print(f"[SKIP] File removed for re-processing")
                continue

            except Exception as e: 
                err_msg = f"MONITORING Process Failed for {looping_filename}: {e}"
                log_and_print(err_msg)
                send_telegram_message_thread(err_msg, "XANDER", 0, 0)
                reverted_filename = looping_filename.replace(".processing", "")
                shutil.move(os.path.join(incoming_directory, looping_filename), os.path.join(error_directory, reverted_filename))
                log_and_print(f"{looping_filename} renamed to {reverted_filename}")
                log_and_print(f"{reverted_filename} sent to {error_directory}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def sync_closed_orders_to_folder():

    if not any(f.endswith(".txt") for f in os.listdir(open_directory)):
        return  # No files to process
    
    safe_ibkr_connect(True)
    
    ib.reqPositions() 
    ib.sleep(1)

    # Build a set of live order IDs
    positions = ib.positions()

    live_position_symbols = {
        p.contract.symbol.strip().upper(): int(round(p.position)) for p in positions
    }

    for filename in os.listdir(open_directory):

        name_part = os.path.splitext(filename)[0]
        parts = name_part.split('_')

        try:
            order_id = parts[4]
            ticker = parts[3]
            lotSize = parts[2]
            action = parts[1]
            botName = parts[0]

        except ValueError:
            raise ValueError(f"Invalid filename format: Expected format BOT_ACTION_LOTSIZE_TICKER_ORDERID_REPROCESS.txt")
        
        isWithinMarketHrs = is_market_open_sgt(False)
        isBeforeAfterMakretHrs = not is_market_open_sgt(True)
        position_size = live_position_symbols.get(ticker, 0)

        # If the ticker is no longer available in the positions list in IBKR, move the file to CLOSED & notify in Telegram the result
        if position_size == 0:

            src = os.path.join(open_directory, filename)
            dest = os.path.join(closed_directory, filename)
            shutil.move(src, dest)
            
            log_and_print(f"Order ID {order_id} CLOSED.")
            log_and_print(f"{filename} sent to {closed_directory}")

            # Calculate P&L
            try:
                executions = ib.reqExecutions()
                ib.sleep(0.5)

                # Filter manually
                matching_execs = [e for e in executions if e.contract.symbol.upper() == ticker.upper()]
                matching_execs.sort(
                    key=lambda e: e.execution.time if isinstance(e.execution.time, datetime) else parse_exec_time(e.execution.time),
                    reverse=True
                )

                buy_fill = next((e.execution for e in matching_execs if e.execution.side == "BOT"), None)
                sell_fill = next((e.execution for e in matching_execs if e.execution.side == "SLD"), None)

                # When buy_fill cannot be retrieved due to IB Session reset, resort to finding within file content
                if not buy_fill:
                    
                    try:
                        with open(dest, "r") as f:
                            for line in f:
                                stripped = line.strip()
                                if stripped.startswith("**BOUGHT AT") and stripped.endswith("**"):
                                    parts = stripped.split("**")
                                    if len(parts) >= 2:
                                        raw_price = parts[1].replace("BOUGHT AT", "").strip().replace("$", "")
                                        recovered_price = float(raw_price)

                                        class FallbackFill:
                                            avgPrice = recovered_price
                                            shares = int(lotSize)
                                        buy_fill = FallbackFill()

                                        break

                    except Exception as e:
                        log_and_print(f"Failed to recover entry price from file: {e}")

                if buy_fill and sell_fill:

                    entry_price = buy_fill.avgPrice
                    exit_price = sell_fill.avgPrice
                    shares = min(buy_fill.shares, sell_fill.shares)  # protect against partials

                    pnl = (exit_price - entry_price) * shares

                    pl_msg = (
                        f"📦 ORDER CLOSED ({order_id})\n — {action.upper()} {ticker.upper()}\n"
                        f"🔹 Entry: {shares} @ ${entry_price:.2f}\n"
                        f"🔹 Exit: {shares} @ ${exit_price:.2f}\n"
                        f"➡️ Realized P&L: ${pnl:.2f}"
                    )

                    send_telegram_message_thread(pl_msg, "XANDER", 0, 0)

                    # Update availble balnace
                    total_return = exit_price * shares
                    respawn_avail_balance(total_return, "SELL")
                    insert_tradeHistory(dest, ticker, entry_price, exit_price, pnl)
                    
                else:
                    # Raise error for missing fill data
                    raise ValueError(f"Error: Missing fill data — Cannot compute P&L.")

            except Exception as e:
                log_and_print(f"Error computing P&L for {filename}: {e}")
                error_msg = f"⚠️ Error computing P&L for {filename}: {e}"
                send_telegram_message_thread(error_msg, "XANDER", 0, 0)

        # Check current price of stock to adjust the trailing dynamically or adjust earnings sell to TRAIL
        elif isWithinMarketHrs:
            
            try:
                
                targetted_file = os.path.join(open_directory, filename)
                current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(targetted_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Check if it has been adjusted to TRAIL
                isTradeAdjusted = False

                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped == "**TRADE ADJUSTED TO MARKET**":
                        isTradeAdjusted = True
                
                # Verify if limit order still contains TP & SL order, cancel them & convert to TRAIL
                if not isTradeAdjusted:

                    trades = ib.trades()

                    adjustmentLine = "**TRADE ADJUSTED TO MARKET**"
                    tradeAdjustMsg = f"📦 TRADE ORDER UPDATED\n"

                    for trade in trades:
                        contract = trade.contract
                        order = trade.order
                        status = (trade.orderStatus.status or "").lower()

                        if contract.symbol.strip().upper() != ticker:
                            continue

                        if order.action.upper() != 'SELL':
                            continue

                        if status in {"filled", "cancelled", "inactive"}:
                            continue

                        if order.orderType.upper() in {"STP", "LMT"}:
                            try:
                                ib.cancelOrder(order)
                                tradeAdjustMsg += f"🔚 Cancelled existing {order.orderType.upper()} for {ticker}\n"
                            except Exception as e:
                                tradeAdjustMsg += f"⚠️ Failed to cancel {order.orderType.upper()} for {ticker}\n"
                                errMsg = f"Failed to cancel {order.orderType.upper()} for {ticker}: {e}"
                                log_and_print(errMsg)

                    trailing_percent = return_val_from_marketCap(targetted_file, "TRAILING", warn_on_fallback=True)

                    ib.sleep(2)

                    if contract.symbol.strip().upper() != ticker:
                        from ib_insync import Contract
                        contract = Contract()
                        contract.symbol = ticker
                        contract.secType = "STK"
                        contract.currency = "USD"
                        contract.exchange = "SMART"
                    else:
                        contract.exchange = "SMART"
                    
                    # Create and place trailing stop SELL order
                    trailing_stop = Order(
                        action="SELL",
                        orderType="TRAIL",
                        totalQuantity=lotSize,
                        trailingPercent=trailing_percent,
                        outsideRth=True,
                        transmit=True
                    )

                    trail_trade = ib.placeOrder(contract, trailing_stop)
                    ib.sleep(0.5) 

                    status = trail_trade.orderStatus.status

                    if status.lower() in {"presubmitted", "submitted"}:
                        tradeAdjustMsg += f"📌 Trailing ({trailing_percent}%) stop order for {ticker} placed successfully"
                    else:
                        tradeAdjustMsg += f"⚠️ Trailing stop placement issue ({status})"

                    send_telegram_message_thread(tradeAdjustMsg, "XANDER", 0, 0)

                    # Update file content for earnings trail completed
                    append_marker_line(lines, adjustmentLine)
                    atomic_write_lines(targetted_file, [str(line) for line in lines])

                # Add / check last checked price into file content
                fill_line = f"**LAST CHECKED AT {current_timestamp}**"

                found = False
                last_checked_timestamp = None

                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith("**LAST CHECKED AT") and stripped.endswith("**"):
                        last_checked_timestamp = stripped[len("**LAST CHECKED AT "):-2].strip()
                        found = True
                        break

                if not found:
                    set_marker_line(lines, "**LAST CHECKED AT ", current_timestamp)
                    atomic_write_lines(targetted_file, [str(line) for line in lines])
                    return
                
                if (last_checked_timestamp != "XXX"):

                    last_dt = datetime.strptime(last_checked_timestamp, "%Y-%m-%d %H:%M:%S")
                    now_dt = datetime.now()

                    # time apart is equal or longer than 1 minute
                    if now_dt >= last_dt + timedelta(minutes=1):

                        # Update the timestamp in file again
                        set_marker_line(lines, "**LAST CHECKED AT ", current_timestamp)
                        atomic_write_lines(targetted_file, [str(line) for line in lines])

                        # Use yahoo finance to check highest price last 1 min
                        yticker = yf.Ticker(ticker)
                        price_data = yticker.history(interval="1m", period="1d")

                        if not price_data.empty:
                            latest_row = price_data.iloc[-1]
                            high_price = latest_row["High"]
                            bought_price = None
                            trail_compare = return_val_from_marketCap(os.path.join(open_directory, filename), "DYNAMIC_TRAILING", warn_on_fallback=True)

                            if trail_compare > 0:

                                with open(os.path.join(open_directory, filename), "r") as f:
                                    for line in f:
                                        stripped = line.strip()
                                        if stripped.startswith("**BOUGHT AT") and stripped.endswith("**"):
                                            parts = stripped.split("**")
                                            if len(parts) >= 2:
                                                raw = parts[1].replace("BOUGHT AT", "").strip().replace("$", "")
                                                bought_price = float(raw)
                                                break
                                
                                if bought_price is not None:

                                    if high_price >= bought_price * (1 + trail_compare):

                                        # Modify existing TRAIL SELL
                                        # Step 1: Get open trades for the ticker
                                        open_trades = ib.openTrades()

                                        # Step 2: Find your trailing stop order for this ticker
                                        for order_obj in open_trades:
                                            contract = order_obj.contract
                                            order = order_obj.order

                                            if contract.symbol.upper() == ticker.upper() and order.orderType == "TRAIL":

                                                # Step 3: Cancel the original order
                                                ib.cancelOrder(order)

                                                # Wait for the cancellation to be processed
                                                ib.sleep(1)  # Optional, or use an event listener

                                                send_telegram_message_thread(f"🔚 Cancelled existing {order.orderType.upper()} for {ticker}\n", "XANDER", 0, 0)

                                                # Step 4: Create a new order with modified trailing percent
                                                new_trailing_order = Order(
                                                    action=order.action,
                                                    orderType='TRAIL',
                                                    totalQuantity=order.totalQuantity,
                                                    trailingPercent=trail_compare * 100,
                                                    tif=order.tif
                                                )

                                                # Reuse the same contract object
                                                trail_trade = ib.placeOrder(contract, new_trailing_order)
                                                ib.sleep(0.5) 

                                                status = trail_trade.orderStatus.status
                                                trailAdjustMsg = ""
                                                if status.lower() in {"presubmitted", "submitted"}:
                                                    trailAdjustMsg += f"🔄 TRAIL SELL for {ticker} has been updated to {trail_compare * 100:.2f}%"
                                                else:
                                                    trailAdjustMsg += f"⚠️ Trailing stop placement issue ({status})"

                                                send_telegram_message_thread(trailAdjustMsg, "XANDER", 0, 0)

                                                break

                                        # Disable re-checking
                                        set_marker_line(lines, "**LAST CHECKED AT ", "XXX")
                                        atomic_write_lines(targetted_file, [str(line) for line in lines])

                                else:
                                    raise Exception("Unable to fetch average bought price")

                            else:

                                # Disable re-checking
                                set_marker_line(lines, "**LAST CHECKED AT ", "XXX")
                                atomic_write_lines(targetted_file, [str(line) for line in lines])

                        else:
                            raise Exception("Unable to fetch price via Yahoo Finance")

            except Exception as e:
                err_msg = f"TRAILIFY Process Failed for {filename}: {e}"
                log_and_print(err_msg)

        # To monitor price of orders to ensure it does not exceed / skip the limits too much
        elif isBeforeAfterMakretHrs:
                
            targetted_file = os.path.join(open_directory, filename)
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(targetted_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            # Add / check last checked price into file content
            fill_line = f"**LAST CHECKED AT {current_timestamp}**"

            found = False
            last_checked_timestamp = None

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("**LAST CHECKED AT") and stripped.endswith("**"):
                    last_checked_timestamp = stripped[len("**LAST CHECKED AT "):-2].strip()
                    found = True
                    break

            if not found:
                set_marker_line(lines, "**LAST CHECKED AT ", current_timestamp)
                atomic_write_lines(targetted_file, [str(line) for line in lines])
                return
            
            if (last_checked_timestamp is None or last_checked_timestamp != "XXX"):

                last_dt = datetime.strptime(last_checked_timestamp, "%Y-%m-%d %H:%M:%S")
                now_dt = datetime.now()

                # time apart is equal or longer than 12s
                if now_dt >= last_dt + timedelta(seconds=12):

                    set_marker_line(lines, "**LAST CHECKED AT ", current_timestamp)
                    atomic_write_lines(targetted_file, [str(line) for line in lines])

                    market_snapshot = fetch_limit_breach_market_data(ticker)
                    if market_snapshot is None:
                        continue

                    current_tp_price = None
                    current_sl_price = None

                    trades = ib.trades()

                    for trade in trades:
                        contract = trade.contract
                        order = trade.order
                        status = (trade.orderStatus.status or "").lower()

                        if contract.symbol.strip().upper() != ticker:
                            continue

                        if order.action.upper() != 'SELL':
                            continue

                        if status in {"filled", "cancelled", "inactive"}:
                            continue

                        if order.orderType.upper() == "LMT":
                            current_tp_price = order.lmtPrice

                        elif order.orderType.upper() == "STP LMT":
                            current_sl_price = order.auxPrice

                    if current_tp_price is None and current_sl_price is None:
                        log_limit_breach_skip(ticker, "LIMIT_PRICE_MISSING", "No active TP/SL limit prices found")
                        continue

                    breach = evaluate_limit_breach(ticker, current_tp_price, current_sl_price, market_snapshot)
                    if breach is None:
                        continue

                    if int(position_size) <= 0:
                        log_limit_breach_skip(ticker, "NO_POSITION", f"position_qty={position_size}")
                        continue

                    if not reserve_limit_breach_sell(ticker):
                        continue

                    try:
                        target = None

                        for trade in trades:
                            contract = trade.contract
                            order = trade.order
                            status = (trade.orderStatus.status or "").lower()

                            if contract.symbol.strip().upper() != ticker:
                                continue
                            else:
                                target = trade

                            if order.action.upper() != 'SELL':
                                continue

                            if status in {"filled", "cancelled", "inactive"}:
                                continue

                            if order.orderType.upper() in {"LMT", "STP LMT"}:
                                try:
                                    ib.cancelOrder(order)
                                except Exception as e:
                                    errMsg = f"Failed to cancel {order.orderType.upper()} for {ticker}: {e}"
                                    log_warning_and_print(errMsg)

                        if target is not None:
                            msg = format_limit_breach_message(target.contract.symbol, breach, market_snapshot, int(position_size))
                            log_warning_and_print(msg)
                            send_telegram_message_thread(msg, "XANDER", 0, 0)

                            cancel_all_orders([target.contract.symbol])
                        else:
                            log_limit_breach_skip(ticker, "NO_TARGET_TRADE", "Breach detected but no matching trade found")

                    except Exception as e:
                        err_msg = f"LIMIT SELL Process Failed for {filename}: {e}"
                        log_warning_and_print(err_msg)

                    continue
                    """

                                        msg = f"⛔ {target.contract.symbol} has breached limit price. Initiating sell execution.."
                                        log_and_print(msg)
                                        send_telegram_message_thread(msg, "XANDER", 0, 0)

                                    cancel_all_orders([target.contract.symbol])

                            except Exception as e:
                                err_msg = f"LIMIT SELL Process Failed for {filename}: {e}"
                                log_and_print(err_msg)
                                
                    else:
                        log_and_print("Error occurred getting Ask price from IBKR.")
                    """

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# GATEWAY COMMAND HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def reply_maintenance():
    file_name_IBKR_Maintenance = "MAINTENANCE_IBKR.txt"
    file_path_IBKR_Maintenance = os.path.join(functionsFolder_path, file_name_IBKR_Maintenance)
    with open(file_path_IBKR_Maintenance, 'w') as file:
        file.write("")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def is_ibkr_connected():
    if (is_within_ibkr_maintenance()):
        reply_maintenance()
    else:
        return safe_ibkr_connect_check(True)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def convert_sgd_to_usd(amount):
    url = "https://api.frankfurter.app/latest"
    params = { "amount": amount, "from": "SGD", "to": "USD" }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        usd_amount = data["rates"]["USD"]
        if not isinstance(usd_amount, (int, float)) or usd_amount <= 0:
            raise ValueError(f"Invalid live SGD/USD conversion value: {usd_amount!r}")
        return usd_amount
    except Exception as live_error:
        log_and_print(f"[FX][WARN] Live SGD to USD conversion failed: {live_error}")

        try:
            cached_text = safe_read_text(sgd_usd_rate_cache_path).strip()
            if not cached_text:
                raise ValueError("Cached SGD/USD rate file is empty.")

            cached_rate = float(cached_text)
            if cached_rate <= 0:
                raise ValueError(f"Cached SGD/USD rate must be positive, got {cached_rate}.")

            fallback_amount = amount * cached_rate
            log_and_print(
                f"[FX][WARN] Using cached SGD/USD rate {cached_rate} from {sgd_usd_rate_cache_path}"
            )
            return fallback_amount
        except Exception as cache_error:
            err_msg = (
                f"[FX][ERROR] Failed to convert SGD to USD via live API and cached rate. "
                f"live_error={live_error}; cache_error={cache_error}"
            )
            log_and_print(err_msg)
            raise RuntimeError(err_msg) from cache_error
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# SELL / CANCEL HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def cancel_all_orders(tickers_list=None):

    logged_time = time_module.time()
    safe_ibkr_connect(True)

    # === Ensure trade and order state is fresh ===
    ib.reqOpenOrders()
    ib.reqExecutions()
    ib.sleep(1.0)
    open_orders = ib.openOrders()
    open_trades = ib.openTrades()

    # === Cancel all pending orders ===
    if not open_orders:
        msg = "No pending orders to cancel."
        log_and_print(msg)
        send_telegram_message_thread(msg, "XANDER", 0, 0)
    else:

        sell_count = 0

        for order in open_orders:
            
            match = True

            if (tickers_list is not None):
                trade = next((t for t in open_trades if t.order == order), None)
                ticker = trade.contract.symbol if trade else None
                if ticker and ticker.upper() not in [t for t in tickers_list]:
                    match = False
            
            if match:
                log_and_print(f"Cancelling Order --> ID: {order.orderId} | {order.action} {order.totalQuantity} {order.orderType}")
                ib.cancelOrder(order)
                sell_count +=1

        final_msg = f"Total Pending Orders Cancelled: {sell_count}"
        log_and_print(final_msg)
        send_telegram_message_thread(final_msg, "XANDER", 0, 0)

    # === Move all files in INCOMING to CLOSED ===
    for file_name in os.listdir(incoming_directory):
        match = True

        if (tickers_list is not None):
            match = next((tick for tick in tickers_list if f"_{tick}_" in file_name), False)

        if match:
            file_path = os.path.join(incoming_directory, file_name)

            if os.path.isfile(file_path):
                last_modified = os.path.getmtime(file_path)

                # If the file was modified before the recorded time, move it
                if last_modified < logged_time:
                    move_to_ignored(
                        file_path,
                        "CANCEL_ALL_ORDERS_CLEANUP",
                        detail=f"cancel_all_orders moved stale INCOMING file; tickers_filter={tickers_list}; last_modified={datetime.fromtimestamp(last_modified).isoformat(timespec='seconds')}",
                        original_filename=file_name,
                        destination_filename=file_name,
                    )

    # === Sell all open positions ===
    positions = ib.positions()
    open_positions = {
        p.contract.symbol: p for p in positions if p.position > 0
    }

    if not open_positions:
        msg = "No open positions found to sell."
        log_and_print(msg)
        send_telegram_message_thread(msg, "XANDER", 0, 0)
        return

    if is_market_open_sgt(True):

        # === Submit new MARKET SELLs if none exist ===
        sell_count = 0

        for symbol, position in open_positions.items():

            if (tickers_list is None) or (tickers_list is not None and symbol.upper() in tickers_list):

                contract = position.contract
                contract.exchange = "SMART"
                qty_to_sell = int(round(position.position))

                sell_order = Order(
                    action='SELL',
                    orderType='MKT',
                    totalQuantity=qty_to_sell,
                    transmit=True
                )

                sell_order = ib.placeOrder(contract, sell_order)
                ib.sleep(0.5)

                status = sell_order.orderStatus.status

                if status.lower() not in {"inactive", "cancelled", "apicancelled", "pendingcancel"}:
                    msg = f"Submitted MARKET SELL for {qty_to_sell} of {contract.symbol}"
                    log_and_print(msg)
                    send_telegram_message_thread(msg, "XANDER", 0, 0)
                    sell_count += 1
                    
                else:
                    msg = f"🚨 Request on MARKET SELL for {qty_to_sell} of {contract.symbol} was not accepted by IBKR ({status})"
                    log_and_print(msg)
                    send_telegram_message_thread(msg, "XANDER", 0, 0)
                    sell_count += 1

        final_msg = f"Total Market Sell Orders Submitted: {sell_count}"
        log_and_print(final_msg)
        send_telegram_message_thread(final_msg, "XANDER", 0, 0)

    else:

        try:
            # === Submit new LIMIT SELLs if none exist ===
            sell_count = 0

            for symbol, position in open_positions.items():

                if (tickers_list is None) or (tickers_list is not None and symbol.upper() in tickers_list):

                    contract = position.contract
                    contract.exchange = "SMART"
                    qty_to_sell = int(round(position.position))

                    result = subprocess.run(
                        ["python", "IBKR_MD.py", symbol],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        cwd=ibkrFolder_path
                    )

                    t_live = json.loads(result.stdout)
                    bid_price = t_live['bid']

                    sell_order = LimitOrder(
                        action='SELL',
                        totalQuantity=qty_to_sell,
                        lmtPrice=bid_price,
                        tif='DAY',
                        outsideRth=True
                    )

                    sell_order = ib.placeOrder(contract, sell_order)
                    ib.sleep(0.5)

                    status = sell_order.orderStatus.status

                    if status.lower() not in {"inactive", "cancelled", "apicancelled", "pendingcancel", "pendingsubmit"}:
                        msg = f"Submitted LIMIT SELL for {qty_to_sell} of {contract.symbol}"
                        log_and_print(msg)
                        send_telegram_message_thread(msg, "XANDER", 0, 0)
                        sell_count += 1
                        
                    else:
                        msg = f"🚨 Request on LIMIT SELL for {qty_to_sell} of {contract.symbol} was not accepted by IBKR ({status})"
                        log_and_print(msg)
                        send_telegram_message_thread(msg, "XANDER", 0, 0)
                        sell_count += 1

            final_msg = f"Total LIMIT Sell Orders Submitted: {sell_count}"
            log_and_print(final_msg)
            send_telegram_message_thread(final_msg, "XANDER", 0, 0)

        except Exception:
            err_msg = f"🚨 LIMIT Selling Failed"
            log_and_print(err_msg)
            send_telegram_message_thread(err_msg, "XANDER", 0, 0)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# LIMIT-BREACH / MARKET DATA HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def format_price(value):
    if value is None:
        return ""
    try:
        # Filter out absurd placeholders
        if value != 0.0 and abs(value) < 1e6:
            return f" ${value:.2f}"
    except Exception:
        pass
    return ""
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def decimal_price(value):
    try:
        if value is None:
            return None
        price = Decimal(str(value)).quantize(LIMIT_BREACH_PRICE_QUANT)
        if not price.is_finite() or price <= 0:
            return None
        return price
    except (InvalidOperation, ValueError, TypeError):
        return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def format_decimal_price(value):
    if value is None:
        return "N.A."
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def log_limit_breach_skip(ticker, reason, detail):
    key = f"limit_breach_skip_{ticker}_{reason}"
    if should_log_repeated_status(key):
        log_warning_and_print(f"[LIMIT_BREACH][SKIP] {ticker} | reason={reason} | detail={detail}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fetch_limit_breach_market_data(ticker):
    try:
        result = subprocess.run(
            ["python", "IBKR_MD.py", ticker],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=ibkrFolder_path
        )
    except subprocess.TimeoutExpired:
        log_limit_breach_skip(ticker, "MARKET_DATA_TIMEOUT", "IBKR_MD.py timed out after 15s")
        return None
    except Exception as exc:
        log_limit_breach_skip(ticker, "MARKET_DATA_SUBPROCESS_FAILED", f"{type(exc).__name__}: {exc}")
        return None

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if result.returncode != 0:
        detail = stderr or stdout or f"returncode={result.returncode}"
        log_limit_breach_skip(ticker, "MARKET_DATA_HELPER_FAILED", detail)
        return None

    try:
        snapshot = json.loads(stdout)
    except json.JSONDecodeError:
        log_limit_breach_skip(ticker, "MARKET_DATA_PARSE_FAILED", f"raw={stdout[:300]}")
        return None

    if snapshot.get("error"):
        log_limit_breach_skip(ticker, "MARKET_DATA_ERROR", snapshot.get("error"))
        return None

    bid = decimal_price(snapshot.get("bid"))
    ask = decimal_price(snapshot.get("ask"))
    last = decimal_price(snapshot.get("last"))

    if bid is None:
        log_limit_breach_skip(
            ticker,
            "INVALID_TRIGGER_PRICE",
            f"bid={snapshot.get('bid')} ask={snapshot.get('ask')} last={snapshot.get('last')}",
        )
        return None

    if ask is not None and ask < bid:
        log_limit_breach_skip(ticker, "INVALID_BID_ASK", f"bid={format_decimal_price(bid)} ask={format_decimal_price(ask)}")
        return None

    spread_pct = None
    if ask is not None:
        midpoint = (bid + ask) / Decimal("2")
        if midpoint > 0:
            spread_pct = ((ask - bid) / midpoint) * Decimal("100")
            if spread_pct > Decimal(str(LIMIT_BREACH_MAX_SPREAD_PCT)):
                log_limit_breach_skip(
                    ticker,
                    "WIDE_BID_ASK_SPREAD",
                    f"spread={float(spread_pct):.2f}% max={LIMIT_BREACH_MAX_SPREAD_PCT:.2f}% bid={format_decimal_price(bid)} ask={format_decimal_price(ask)}",
                )
                return None

    return {
        "ticker": ticker,
        "bid": bid,
        "ask": ask,
        "last": last,
        "price_source": "bid",
        "trigger_price": bid,
        "spread_pct": spread_pct,
        "raw": snapshot,
    }
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def evaluate_limit_breach(ticker, current_tp_price, current_sl_price, market_snapshot):
    trigger_price = market_snapshot["trigger_price"]
    tp_price = decimal_price(current_tp_price)
    sl_price = decimal_price(current_sl_price)

    if sl_price is not None and trigger_price <= sl_price:
        difference = trigger_price - sl_price
        diff_pct = (difference / sl_price) * Decimal("100")
        return {
            "breach_type": "STOP_LOSS",
            "limit_price": sl_price,
            "trigger_price": trigger_price,
            "difference": difference,
            "difference_pct": diff_pct,
        }

    if tp_price is not None and trigger_price >= tp_price:
        difference = trigger_price - tp_price
        diff_pct = (difference / tp_price) * Decimal("100")
        return {
            "breach_type": "TAKE_PROFIT",
            "limit_price": tp_price,
            "trigger_price": trigger_price,
            "difference": difference,
            "difference_pct": diff_pct,
        }

    return None
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def reserve_limit_breach_sell(ticker):
    now = time_module.time()
    last_requested = limit_breach_sell_pending.get(ticker)
    if last_requested and now - last_requested < LIMIT_BREACH_PENDING_COOLDOWN_SECONDS:
        remaining = LIMIT_BREACH_PENDING_COOLDOWN_SECONDS - (now - last_requested)
        log_limit_breach_skip(
            ticker,
            "SELL_ALREADY_PENDING",
            f"last_request_age={int(now - last_requested)}s cooldown_remaining={int(remaining)}s",
        )
        return False

    limit_breach_sell_pending[ticker] = now
    return True
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def format_limit_breach_message(ticker, breach, market_snapshot, position_qty):
    limit_price = breach["limit_price"]
    trigger_price = breach["trigger_price"]
    difference = breach["difference"]
    difference_pct = breach["difference_pct"]
    bid = market_snapshot.get("bid")
    ask = market_snapshot.get("ask")
    last = market_snapshot.get("last")

    return (
        f"⛔ {ticker} breached limit price. Initiating sell execution.\n"
        f"Type: {breach['breach_type']}\n"
        f"Limit: {format_decimal_price(limit_price)}\n"
        f"Current/Trigger price: {format_decimal_price(trigger_price)}\n"
        f"Difference: {float(difference):+.4f} ({float(difference_pct):+.2f}%)\n"
        f"Price source: {market_snapshot.get('price_source', 'bid')}\n"
        f"Bid/Ask/Last: {format_decimal_price(bid)} / {format_decimal_price(ask)} / {format_decimal_price(last)}\n"
        f"Position qty: {position_qty}\n"
        f"Order action: SELL"
    )
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_last_market_execution():
    
    safe_ibkr_connect(False)

    sgt = pytz.timezone(XANDER_TIMEZONE)
    now = datetime.now(sgt)

    # Define session start: Today at 12:00 AM SGT
    session_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def format_pretty(dt):
        if platform.system() == "Windows":
            return dt.strftime("%#d %b %y %#I.%M%p")
        else:
            return dt.strftime("%-d %b %y %-I.%M%p")

    summary = (
        f"🧾 Executions TODAY (since {format_pretty(session_start)} SGT):\n"
    )

    # Use correct UTC filter format
    filter_time = (session_start.astimezone(pytz.utc)).strftime('%Y%m%d-%H:%M:%S')
    exec_filter = ExecutionFilter(time=filter_time)
    executions = ib.reqExecutions(exec_filter)
    ib.sleep(0.5)

    any_found = False
    for execDetail in executions:
        exec_time = execDetail.execution.time
        exec_time_sgt = exec_time.astimezone(sgt) if exec_time.tzinfo else pytz.utc.localize(exec_time).astimezone(sgt)

        if exec_time_sgt >= session_start:
            any_found = True
            summary += (
                f"- [{exec_time_sgt.strftime('%H:%M:%S')}] {execDetail.contract.symbol} "
                f"{execDetail.execution.side} {execDetail.execution.shares} @ {execDetail.execution.avgPrice:.2f} "
                f"(Order ID: {execDetail.execution.orderId})\n"
            )

    if not any_found:
        summary += "No executions today.\n"

    # Include open/pending orders
    open_trades = [t for t in ib.trades() if t.orderStatus.status not in ('Filled', 'Cancelled')]
    if open_trades:
        summary += "\n📋 Open Orders:\n"
        for t in open_trades:
            order = t.order
            contract = t.contract
            order_type = order.orderType.upper()
            if order_type == "LMT":
                price_info = f" ${order.lmtPrice:.2f}"
            elif order_type in ("STP", "STP LMT"):
                price_info = f" ${order.auxPrice:.2f}"
            elif order_type == "TRAIL":
                price_info = format_price(order.trailStopPrice) or format_price(order.auxPrice)
            else:
                price_info = ""

            summary += (
                f"- {contract.symbol} {order.action.upper()} {order.totalQuantity} @ {order_type}{price_info} "
                f"(Order ID: {order.orderId})\n"
            )
    else:
        summary += "\n📋 Open Orders: None"

    send_telegram_message_thread(summary.strip(), "XANDER", 0, 0)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ACCOUNT / REPORTING HELPERS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_unrealized_pl():
    
    safe_ibkr_connect(False)

    portfolio = ib.portfolio()

    if not portfolio:
        send_telegram_message_thread("📭 No open positions.", "XANDER", 0, 0)

    else:

        msg_lines = ["📊 Unrealized P&L Update\n"]
        total_pnl = 0.0

        for p in portfolio:
            pnl = p.unrealizedPNL or 0.0
            total_pnl += pnl
            msg_lines.append(
                f"• {p.contract.symbol}: {p.position} @ {p.averageCost:.2f} "
                f"(Now {p.marketPrice:.2f}) → Unrealized P&L: {pnl:.2f} USD"
            )

        msg_lines.append(f"\n💰 Total Unrealized P&L: {total_pnl:.2f} USD")
        send_telegram_message_thread("\n".join(msg_lines), "XANDER", 0, 0)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def create_sells():

    filename = "GATEWAY_CREATE_SELLS_IBKR.txt"
    file_path = os.path.join(functionsFolder_path, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    os.remove(file_path)
    
    # Split by "," and strip each part
    tickers = [item.strip() for item in content.split(",") if item.strip()]
    cancel_all_orders(tickers)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def replace_sells():

    filename = "GATEWAY_REPLACE_SELLS_IBKR.txt"
    file_path = os.path.join(functionsFolder_path, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    os.remove(file_path)

    tickers = [item.strip().upper() for item in content.split(",") if item.strip()]
    open_files = os.listdir(open_directory)
    isWithinMarketHrs = is_market_open_sgt(False)

    for ticker in tickers:

        match = next((f for f in open_files if f"_{ticker}_" in f), None)
        if not match:
            continue

        file_path = os.path.join(open_directory, match)

        if not isWithinMarketHrs:
            place_limit_sell(ib, ticker, file_path)
            continue

        etfs = ["TQQQ", "SOXL", "ERX", "FAS", "CURE", "WANT", "DFEN", "DUSL", "DRN"]

        if ticker in {"SPXL", "TQQQ"}:
            trailing_percent = 0.35
        elif ticker in etfs:
            trailing_percent = 0.5
        else:
            trailing_percent = return_val_from_marketCap(file_path, "TRAILING", warn_on_fallback=True)

        pos = next((p for p in ib.positions() if p.contract.symbol.upper() == ticker), None)
        if not pos or pos.position <= 0:
            continue

        lotSize = int(round(pos.position))

        ib.reqOpenOrders()
        ib.sleep(0.2)

        active_statuses = {
            "Submitted", "PreSubmitted", "PendingSubmit",
            "ApiPending", "PendingCancel"
        }

        open_trails = [
            tr for tr in ib.openTrades()
            if tr.contract.symbol.upper() == ticker
            and tr.order.orderType.upper() == "TRAIL"
            and tr.orderStatus.status in active_statuses
        ]

        result = subprocess.run(
            ["python", "IBKR_MD.py", ticker],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=ibkrFolder_path
        )

        bid_price = None
        
        if result.returncode == 0:
            t_live = json.loads(result.stdout)
            bid_price = t_live['bid']

        if not bid_price or bid_price <= 0:
            continue

        new_implied_stop = bid_price * (1 - trailing_percent / 100)

        replace_allowed = True

        if open_trails:
            tr = open_trails[0]

            old_stop = tr.orderStatus.stopPrice
            if old_stop and old_stop > 0:
                if new_implied_stop <= old_stop:
                    send_telegram_message_thread(
                        f"🛑 TRAIL for {ticker} not replaced "
                        f"(would loosen stop {old_stop:.4f} → {new_implied_stop:.4f})",
                        "XANDER", 0, 0
                    )
                    replace_allowed = False

        if not replace_allowed:
            continue

        for tr in open_trails:
            ib.cancelOrder(tr.order)

        if open_trails:
            ib.sleep(1.0)
            send_telegram_message_thread(
                f"♻️ TRAIL for {ticker} cancelled (tightening protection).",
                "XANDER", 0, 0
            )

        contract = Stock(ticker, "SMART", "USD")

        trailing_stop = Order(
            action="SELL",
            orderType="TRAIL",
            totalQuantity=lotSize,
            trailingPercent=trailing_percent,
            outsideRth=True,
            tif="GTC",
            transmit=True
        )

        trade = ib.placeOrder(contract, trailing_stop)
        ib.sleep(1.0)

        status = trade.orderStatus.status
        send_telegram_message_thread(
            f"📉 TRAIL STOP for {lotSize} of {ticker} placed "
            f"({trailing_percent:.2f}% trail). Status: {status}",
            "XANDER", 0, 0
        )
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fetch_spreads():

    filename = "GATEWAY_FETCH_SPREADS_IBKR.txt"
    file_path = os.path.join(functionsFolder_path, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    os.remove(file_path)

    # Split by "," and strip each part
    tickers = [item.strip() for item in content.split(",") if item.strip()]

    for ticker in tickers:
        try:
            result = subprocess.run(
                ["python", "IBKR_MD.py", ticker],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=ibkrFolder_path
            )

            if result.returncode == 0 and result.stdout.strip():
                try:
                    t_live = json.loads(result.stdout)

                    msg_lines = [
                        f"📊 {t_live.get('ticker', ticker)}",
                        f"🔴 Bid: {t_live.get('bid', 'N.A.')}",
                        f"🟢 Ask: {t_live.get('ask', 'N.A.')}",
                        f"⚪ Last: {t_live.get('last', 'N.A.')}"
                    ]

                    send_telegram_message_thread("\n".join(msg_lines), "XANDER", 0, 0)

                except json.JSONDecodeError:
                    send_telegram_message_thread(
                        f"⚠️ Could not parse market data for {ticker}. Raw: {result.stdout}",
                        "XANDER", 0, 0
                    )
            else:
                err_msg = result.stderr.strip() or "Unknown error"
                send_telegram_message_thread(
                    f"🚨 Failed to fetch market data for {ticker}: {err_msg}",
                    "XANDER", 0, 0
                )

        except subprocess.TimeoutExpired:
            send_telegram_message_thread(
                f"⏱️ Timeout while fetching market data for {ticker}.",
                "XANDER", 0, 0
            )
        except Exception as e:
            log_and_print(e)
            send_telegram_message_thread(
                f"❌ Unexpected error for {ticker}: {e}",
                "XANDER", 0, 0
            )
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_ibkr_balance(isTele):
    
    safe_ibkr_connect(False)
    summary_list = fetch_account_summary_snapshot()
    summary_dict = {item.tag: item.value for item in summary_list}

    cash = float(summary_dict.get('TotalCashValue', 'N/A'))
    
    if not isTele:
        return cash
    
    base_balance = fetch_base_balance()
    avail_funds = fetch_avail_balance()

    if cash < 0:
        usdIBKRBalance = -convert_sgd_to_usd(abs(cash))
    else:
        usdIBKRBalance = convert_sgd_to_usd(cash)

    msg = (
        f"📊 Account Balance Summary\n"
        f"• Base Balance (Non-RTH-Session): ${float(base_balance):,.2f}\n"
        f"• IBKR Cash Balance: ${usdIBKRBalance:,.2f}\n"
        f"• Available Funds: ${float(avail_funds):,.2f}"
    )

    send_telegram_message_thread(msg, "XANDER", 0, 0)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def select_account_value(summary_list, tag):
    fallback = None
    for item in summary_list:
        if getattr(item, "tag", None) != tag:
            continue
        currency = str(getattr(item, "currency", "") or "").upper()
        if currency and currency != "BASE":
            return item
        if fallback is None:
            fallback = item
    return fallback
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def account_value_float(item):
    return float(getattr(item, "value", ""))
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def write_lucian_balance_snapshot_response():
    response_path = os.path.join(functionsFolder_path, LUCIAN_BALANCE_SNAPSHOT_RESPONSE_FILE)
    timestamp = datetime.now(pytz.timezone(XANDER_TIMEZONE)).isoformat(timespec="seconds")

    try:
        if not safe_ibkr_connect_check(True):
            raise RuntimeError("IBKR is not connected")

        summary_list = fetch_account_summary_snapshot()
        balance_item = select_account_value(summary_list, "NetLiquidation")
        metric = "NetLiquidation"
        if balance_item is None:
            balance_item = select_account_value(summary_list, "TotalCashValue")
            metric = "TotalCashValue"
        if balance_item is None:
            raise RuntimeError("NetLiquidation and TotalCashValue are unavailable")

        currency = str(getattr(balance_item, "currency", "") or "USD").upper()
        payload = {
            "ok": True,
            "timestamp": timestamp,
            "net_liquidation": round(account_value_float(balance_item), 2),
            "currency": currency,
            "source": "IBKR",
            "metric": metric,
        }
        atomic_write_json(response_path, payload)
        ibkrautomate_logger.info(
            "Lucian balance snapshot response written | metric=%s | currency=%s",
            metric,
            currency,
        )
    except Exception as exc:
        payload = {
            "ok": False,
            "timestamp": timestamp,
            "error": f"{type(exc).__name__}: {exc}",
            "source": "IBKR",
        }
        atomic_write_json(response_path, payload)
        ibkrautomate_logger.warning("Lucian balance snapshot failed: %s - %s", type(exc).__name__, exc)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fetch_base_balance():

    file_path = os.path.join(functionsFolder_path, 'GATEWAY_IBKR_BaseBalance.txt')

    with open(file_path, "r") as f:
        balance_str = f.read().strip()

    if balance_str == "":
        return -1
    return float(balance_str)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fetch_avail_balance():

    file_path = os.path.join(functionsFolder_path, 'GATEWAY_IBKR_AvailBalance.txt')

    with open(file_path, "r") as f:
        balance_str = f.read().strip()

    if balance_str == "":
        return -1
    return float(balance_str)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_base_balance(balance):

    file_path = os.path.join(functionsFolder_path, 'GATEWAY_IBKR_BaseBalance.txt')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(balance))
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def update_avail_balance(balance):

    file_path = os.path.join(functionsFolder_path, 'GATEWAY_IBKR_AvailBalance.txt')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(balance))
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def respawn_avail_balance(amount, action):
    with lock:

        current_balance = fetch_avail_balance()

        if (action == "BUY"):
            new_balance = current_balance - amount
        elif (action == "SELL"):
            new_balance = current_balance + amount

        update_avail_balance(new_balance)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def refresh_base_balance():
    balance = get_ibkr_balance(False)
    usdBalance = convert_sgd_to_usd(balance)
    update_base_balance(usdBalance)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def refresh_avail_balance(isManualUpdate):
    isWithinMarketHrs = is_market_open_sgt(False)
    if isWithinMarketHrs and isManualUpdate:
        balance = convert_sgd_to_usd(get_ibkr_balance(False))
    else:
        balance = fetch_base_balance()
    update_avail_balance(balance)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def reprocess_incoming_orders():
    for filename in os.listdir(incoming_directory):
        full_path = os.path.join(incoming_directory, filename)

        if filename.endswith('.processing'):
            name_with_txt = filename[:-len('.processing')]  # e.g. "XANDER_BUY_SPXL_ABC123_0.txt"

            if not name_with_txt.endswith('.txt'):
                continue  # skip malformed

            base_name = os.path.splitext(name_with_txt)[0]  # e.g. "XANDER_BUY_SPXL_ABC123_0"

            if base_name.endswith('_0'):
                updated_base = base_name[:-2] + '_1'  # safely replace _0 with _1
                new_name = updated_base + '.txt'
                new_path = os.path.join(incoming_directory, new_name)
                os.rename(full_path, new_path)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# FILE WATCHER / MAIN LOOP
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def fileCheck():

    for file_name in os.listdir(functionsFolder_path):

        file_path = os.path.join(functionsFolder_path, file_name)

        if (file_name == "STATUSCHECK_IBKR.txt"):
            os.remove(file_path)

        if (file_name == "GATEWAY_IBKR.txt"):
            isConnected = is_ibkr_connected()
            if isConnected:
                os.remove(file_path)

        if (file_name == "GATEWAY_CREATE_SELLS_IBKR.txt"):
            create_sells() # os.remove(file_path) handled within

        if (file_name == "GATEWAY_CANCEL_ORDERS_IBKR.txt"):
            try:
                cancel_all_orders()
            finally:
                os.remove(file_path)

        if (file_name == "GATEWAY_GET_PAST_ORDERS_IBKR.txt"):
            try:
                get_last_market_execution()
            finally:
                os.remove(file_path)

        if (file_name == "GATEWAY_GET_UNREALIZED_PNL_IBKR.txt"):
            try:
                get_unrealized_pl()
            finally:
                os.remove(file_path)

        if (file_name == "GATEWAY_GET_BALANCE_IBKR.txt"):
            try:
                get_ibkr_balance(True)
            finally:
                os.remove(file_path)

        if (file_name == "GATEWAY_REPLACE_SELLS_IBKR.txt"):
            replace_sells() # os.remove(file_path) handled within

        if (file_name == "GATEWAY_FETCH_SPREADS_IBKR.txt"):
            fetch_spreads() # os.remove(file_path) handled within

        if (file_name == "GATEWAY_REFRESH_BALANCE_IBKR.txt"):
            try:
                refresh_base_balance()
                refresh_avail_balance(False)
                get_ibkr_balance(True)
            finally:
                os.remove(file_path)

        if (file_name == "GATEWAY_REFRESH_AVAIL_BALANCE_IBKR.txt"):
            try:
                refresh_avail_balance(True)
                get_ibkr_balance(True)
            finally:
                os.remove(file_path)

        if (file_name == LUCIAN_BALANCE_SNAPSHOT_REQUEST_FILE):
            try:
                write_lucian_balance_snapshot_response()
            finally:
                os.remove(file_path)

        if (file_name == "GATEWAY_DISCONNECT_IBKR.txt"):
            ib.disconnect()
            os.remove(file_path)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def main():

    last_sync_time_1 = time_module.time()
    last_sync_time_2 = time_module.time()
    close_shop_triggered = False
    update_balance_triggered = False

    while True:
        try:

            # Run daily tasks
            sgt = pytz.timezone(XANDER_TIMEZONE)
            now = datetime.now(sgt)
            current_weekday = now.weekday()

            closing_time = None
            adjustedHours = check_adjusted_trading_hrs()
            
            if (adjustedHours is not None and adjustedHours == "AM"):
                market_close_time = tradingHalfDay
                closing_time = (datetime.combine(datetime.today(), market_close_time) - timedelta(minutes=1)).time() # Putting 1 min before to add buffer for closing orders
            else:
                closing_time = (datetime.combine(datetime.today(), tradingClose) - timedelta(minutes=1)).time() # Putting 1 min before to add buffer for closing orders

            '''# (1) Initiate close all orders at 3.59AM
            if (closing_time <= now.time() < tradingClose) and current_weekday not in [0, 6]:
                if not close_shop_triggered:
                    cancel_all_orders()
                    refresh_base_balance()
                    close_shop_triggered = True'''
            
            # (1) Initiate close all orders 1 min before official closing
            if (closing_time <= now.time() < tradingClose) and current_weekday not in [0, 6]:
                if not close_shop_triggered:
                    cancel_all_orders()
                    refresh_base_balance()
                    close_shop_triggered = True

            # (2) Scan for orders in INCOMING 
            handle_signals_concurrently()

            # (3) Monitor order fills & sync closed orders with IBKR every 2 seconds
            current_time = time_module.time()

            # (4) Health check
            fileCheck()

            # (5) Initiate update balance window period
            if (updateBalWindowOpen <= now.time() < updateBalWindowClose) and current_weekday not in [5, 6]:
                if not update_balance_triggered:
                    refresh_base_balance()
                    refresh_avail_balance(False)
                    update_balance_triggered = True

            # (6) Monitor order fills & sync closed orders with IBKR every 2 seconds
            if current_time - last_sync_time_1 >= 2:
                monitor_order_fill()
                sync_closed_orders_to_folder()
                last_sync_time_1 = current_time

            # (7) Perform audit every 3 minutes
            if current_time - last_sync_time_2 >= 180:
                audit()
                last_sync_time_2 = current_time

            # Reset flag of close all orders
            if now.time() >= tradingClose:
                close_shop_triggered = False

            # Reset flag of close all orders
            if now.time() >= tradingOpen:
                update_balance_triggered = False

        except Exception as loop_error:
                
            # Log and print
            loop_error = f"Loop error: {loop_error}"

            if str(loop_error) != "Loop error: Not connected" and should_log_repeated_status(f"main_loop_{loop_error}"):
                log_and_print(loop_error)

            # Add a retry immediately
            if "socket disconnect" in str(loop_error).lower() or "winerror 10054" in str(loop_error).lower():

                # Check and reprocess incoming
                try:
                    msg = "🚨 Socket disconnected abruptly. Triggering immediate reprocessing..."
                    send_telegram_message_thread(msg, "XANDER", 0, 0)
                    reprocess_incoming_orders()

                except:
                    log_and_print("Reprocessing failed.")

        time_module.sleep(1)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# STARTUP / ENTRYPOINT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Initialize and log start
initialized_info = "IBKR.py initialized..."
ibkrautomate_logger.info(initialized_info)
print(initialized_info)

safe_ibkr_connect(True)

# Initialize IBKR Automation
if __name__ == "__main__":
    os.makedirs(incoming_directory, exist_ok=True)
    os.makedirs(open_directory, exist_ok=True)
    os.makedirs(closed_directory, exist_ok=True)
    os.makedirs(error_directory, exist_ok=True)
    os.makedirs(ignored_directory, exist_ok=True)
    os.makedirs(stale_validating_directory, exist_ok=True)
    recover_stale_validating_files()
    threading.Thread(target=dispatch_ibkr_signal, daemon=True).start()
    main()
    
