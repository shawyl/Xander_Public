"""Telethon session initialization helper.

Creates or validates a local Telegram client session used by source ingestion.
Credentials are loaded from environment variables rather than embedded here.

AI status: Maintained with AI.
"""

import asyncio
import os
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

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

API_ID = require_int_env("TELEGRAM_API_ID")
API_HASH = require_env("TELEGRAM_API_HASH")

async def main():
    client = TelegramClient("session_walter", API_ID, API_HASH)
    await client.start()
    me = await client.get_me()
    print("Logged in as:", me.username or me.first_name)
    await client.disconnect()

asyncio.run(main())
