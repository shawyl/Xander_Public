"""Similarity-check utility for Xander post comparisons.

Provides local embedding and similarity helpers used to compare post content
without exposing private trading decisions.

AI status: Maintained with AI.
"""

import os
import json
import logging
import faiss
import numpy as np
import hashlib
import warnings
import contextlib
from dotenv import load_dotenv

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

load_dotenv()

BOTHUB_ROOT = os.getenv("BOTHUB_ROOT")
if BOTHUB_ROOT is None or BOTHUB_ROOT.strip() == "":
    raise RuntimeError("Missing required environment variable: BOTHUB_ROOT")

def bot_path(*parts):
    return os.path.join(BOTHUB_ROOT, *parts)

try:
    import huggingface_hub
    from huggingface_hub.cli import _output as hf_output
    hf_output.Output.warning = lambda self, message: None
except Exception:
    pass

from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

# CONFIG

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

MODEL = load_sentence_transformer_model("all-MiniLM-L6-v2")  # Change this if needed
POST_DIR = bot_path(".BOT_Launch", "SimilarityCheck", "POSTS")
HOURS_LOOKBACK = 24
THRESHOLD = 0.55
SELF_MATCH_WINDOW = 30  # seconds

def save_post(text: str):
    os.makedirs(POST_DIR, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S') + f"{now.microsecond // 1000:03d}"
    filename = f"Post_TEST_{timestamp}.txt"
    filepath = os.path.join(POST_DIR, filename)

    embedding = MODEL.encode(text, normalize_embeddings=True).tolist()
    data = {
        "timestamp": now.isoformat(),
        "text": text,
        "embedding": embedding,
        "id": hashlib.sha256(text.encode('utf-8')).hexdigest()
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {filename}")
    return filepath

def load_recent_embeddings():
    now = datetime.now()
    vecs, texts, ids, timestamps = [], [], [], []
    count = 0

    for fname in os.listdir(POST_DIR):
        if not fname.endswith('.txt'):
            continue
        try:
            with open(os.path.join(POST_DIR, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
                ts = datetime.fromisoformat(data["timestamp"])
                if (now - ts).total_seconds() <= HOURS_LOOKBACK * 3600:
                    vecs.append(np.array(data["embedding"], dtype='float32'))
                    texts.append(data["text"])
                    ids.append(data.get("id", ""))
                    timestamps.append(ts)
                    count += 1
        except Exception as e:
            print(f"[ERROR] Failed to load {fname}: {e}")
            continue

    print(f"[LOADED] {count} valid posts in last {HOURS_LOOKBACK}h")
    return np.array(vecs), texts, ids, timestamps

def check_similarity(new_text: str):
    new_vec = MODEL.encode([new_text], normalize_embeddings=True)
    now = datetime.now()

    past_vecs, past_texts, past_ids, past_timestamps = load_recent_embeddings()

    if len(past_vecs) == 0:
        print("No previous posts found.")
        return

    if new_vec.shape[1] != past_vecs.shape[1]:
        print("Shape mismatch — check model consistency.")
        return

    index = faiss.IndexFlatIP(new_vec.shape[1])
    index.add(past_vecs)
    D, I = index.search(new_vec, k=3)

    for i in range(len(D[0])):
        score = float(D[0][i])
        idx = int(I[0][i])
        match_text = past_texts[idx]
        match_time = past_timestamps[idx]
        time_diff = abs((now - match_time).total_seconds())

        # Only skip true self-match: exact text and very recent timestamp
        if match_text.strip() == new_text.strip() and time_diff < SELF_MATCH_WINDOW:
            continue

        if score >= THRESHOLD:
            print(f"\n[SCORE] Similarity: {score:.4f}")
            print(f"[MATCHED POST] ({match_time.strftime('%Y-%m-%d %H:%M:%S')})\n{match_text}")
            print("✅ Considered SIMILAR")
            return

    print("❌ No sufficiently similar post found.")

if __name__ == "__main__":
    print("\n=== SIMILARITY TEST ===")
    new_post = input("Enter a new post to check: ").strip()
    save_post(new_post)
    check_similarity(new_post)
