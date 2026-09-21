from __future__ import annotations

"""
try.py - Telegram Launcher Sederhana

Tugas file ini SENGAJA dibuat seminimal mungkin:
1. Menjalankan polling Telegram.
2. Menangani /try untuk memuat/reload main.py.
3. Meneruskan message Telegram lain ke main.py.
4. Menyediakan endpoint HTTP sederhana untuk health check Render.

Semua logic trading, Binance, strategy, trade state, timeout, PnL,
GitHub, dan penyimpanan data berada di main.py / module lain.

Kontrak main.py yang dipakai launcher:
    async def on_start(context): ...
    async def handle_update(update, context): ...
    async def on_stop(context): ...

Callbacks context:
    send_message(chat_id, text)
    send_document(chat_id, document_path, caption)

Function boleh sync maupun async.
"""

import asyncio
import importlib.util
import logging
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "trades.env")

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
MAIN_FILE = (os.getenv("MAIN_FILE") or "main.py").strip()
PORT = int(os.getenv("PORT", "10000"))
TG_POLL_TIMEOUT = max(5, int(os.getenv("TG_POLL_TIMEOUT", "30")))
HTTP_TIMEOUT = max(10, int(os.getenv("HTTP_TIMEOUT", "30")))

try:
    ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
except ValueError as exc:
    raise RuntimeError("ALLOWED_USER_ID harus berupa integer.") from exc

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN belum diset di .env")
if not ALLOWED_USER_ID:
    raise RuntimeError("ALLOWED_USER_ID belum diset di .env")


# -----------------------------------------------------------------------------
# STATE
# -----------------------------------------------------------------------------
log = logging.getLogger("launcher")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

MAIN_LOCK = asyncio.Lock()
MAIN_MODULE: ModuleType | None = None
MAIN_RUNNING = False
STOP_EVENT = asyncio.Event()


# -----------------------------------------------------------------------------
# SMALL HTTP SERVER - hanya untuk Render / health check
# -----------------------------------------------------------------------------
app = Flask(__name__)


@app.get("/")
def index():
    return jsonify(
        {
            "ok": True,
            "service": "telegram-launcher",
            "main_running": MAIN_RUNNING,
        }
    )


@app.get("/healthz")
def healthz():
    return jsonify(
        {
            "ok": True,
            "service": "telegram-launcher",
            "main_running": MAIN_RUNNING,
            "timestamp": time.time(),
        }
    )


def run_http_server() -> None:
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


# -----------------------------------------------------------------------------
# TELEGRAM - launcher hanya menangani komunikasi dasar
# -----------------------------------------------------------------------------
def tg_call(method: str, payload: dict | None = None, timeout: int = HTTP_TIMEOUT):
    response = requests.post(
        f"{TG_API}/{method}",
        json=payload or {},
        timeout=timeout,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Telegram {method}: HTTP {response.status_code}: {response.text[:800]}"
        )

    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram {method}: {body}")

    return body.get("result")


def tg_send(chat_id: int, text: str) -> None:
    # Telegram membatasi panjang message. Pecah pesan panjang menjadi beberapa bagian.
    text = str(text)
    for start in range(0, len(text), 3900):
        chunk = text[start : start + 3900]
        try:
            tg_call("sendMessage", {"chat_id": chat_id, "text": chunk})
        except Exception:
            log.exception("Gagal mengirim Telegram ke chat_id=%s", chat_id)
            return


def tg_send_document(chat_id: int, document_path: str, caption: str = "") -> None:
    """Kirim file lokal sebagai Telegram document."""
    path = Path(str(document_path))
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Dokumen tidak ditemukan: {path}")

    with path.open("rb") as document:
        response = requests.post(
            f"{TG_API}/sendDocument",
            data={
                "chat_id": str(chat_id),
                "caption": str(caption or "")[:1024],
            },
            files={
                "document": (path.name, document),
            },
            timeout=HTTP_TIMEOUT,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Telegram sendDocument: HTTP {response.status_code}: "
            f"{response.text[:800]}"
        )

    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(
            f"Telegram sendDocument: {body}"
        )


# -----------------------------------------------------------------------------
# MAIN LOADER
# -----------------------------------------------------------------------------
def load_main_module() -> ModuleType:
    path = BASE_DIR / MAIN_FILE
    if not path.exists():
        raise FileNotFoundError(f"{MAIN_FILE} tidak ditemukan di {BASE_DIR}")

    # Compile dulu agar syntax error main.py muncul sebelum module dipakai.
    source = path.read_text(encoding="utf-8")
    code = compile(source, str(path), "exec")

    module_name = "trading_main_runtime"
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak dapat membuat module spec untuk {MAIN_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # Gunakan exec code hasil compile agar error menunjuk file main.py dengan benar.
    exec(code, module.__dict__)
    return module


async def _call(module: ModuleType, function_name: str, context: dict) -> object:
    function = getattr(module, function_name, None)
    if function is None:
        return None

    result = function(context)
    if asyncio.iscoroutine(result):
        return await result
    return result


async def start_or_reload_main(chat_id: int) -> None:
    """Menjalankan main.py. Jika sudah aktif, main lama dihentikan lalu dimuat ulang."""
    global MAIN_MODULE, MAIN_RUNNING

    async with MAIN_LOCK:
        old_module = MAIN_MODULE

        if old_module is not None and MAIN_RUNNING:
            log.info("/try diterima saat main masih aktif -> reload main.py")
            MAIN_RUNNING = False
            try:
                await _call(
                    old_module,
                    "on_stop",
                    {
                        "chat_id": chat_id,
                        "user_id": ALLOWED_USER_ID,
                        "send_message": tg_send,
                        "send_document": tg_send_document,
                        "launcher": "try.py",
                    },
                )
            except Exception:
                log.exception("on_stop() main lama gagal")

        MAIN_MODULE = None
        sys.modules.pop("trading_main_runtime", None)

        try:
            module = load_main_module()

            required = ("on_start", "handle_update", "on_stop")
            missing = [name for name in required if not callable(getattr(module, name, None))]
            if missing:
                raise RuntimeError(
                    f"{MAIN_FILE} wajib memiliki function: {', '.join(missing)}"
                )

            context = {
                "chat_id": chat_id,
                "user_id": ALLOWED_USER_ID,
                "send_message": tg_send,
                "send_document": tg_send_document,
                "launcher": "try.py",
                "main_file": str(BASE_DIR / MAIN_FILE),
                "is_running": lambda: MAIN_RUNNING,
            }

            MAIN_RUNNING = True
            MAIN_MODULE = module

            result = await _call(module, "on_start", context)
            if result is False:
                MAIN_RUNNING = False
                MAIN_MODULE = None
                sys.modules.pop("trading_main_runtime", None)
                raise RuntimeError("main.py menolak startup melalui on_start().")

            log.info("main.py aktif")

        except Exception:
            MAIN_RUNNING = False
            MAIN_MODULE = None
            sys.modules.pop("trading_main_runtime", None)
            raise


async def stop_main() -> None:
    """Dipakai saat process launcher berhenti."""
    global MAIN_MODULE, MAIN_RUNNING

    async with MAIN_LOCK:
        module = MAIN_MODULE
        MAIN_RUNNING = False
        MAIN_MODULE = None

        if module is not None:
            try:
                await _call(
                    module,
                    "on_stop",
                    {
                        "chat_id": ALLOWED_USER_ID,
                        "user_id": ALLOWED_USER_ID,
                        "send_message": tg_send,
                        "send_document": tg_send_document,
                        "launcher": "try.py",
                    },
                )
            except Exception:
                log.exception("on_stop() gagal")

        sys.modules.pop("trading_main_runtime", None)


# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------
def authorized(message: dict) -> bool:
    chat_id = int((message.get("chat") or {}).get("id") or 0)
    user_id = int((message.get("from") or {}).get("id") or 0)
    return chat_id == ALLOWED_USER_ID and user_id == ALLOWED_USER_ID


def get_command(text: str) -> str:
    if not text:
        return ""
    return text.split(maxsplit=1)[0].split("@", 1)[0].lower()


async def route_message(message: dict, update: dict) -> None:
    if not authorized(message):
        return

    chat_id = int((message.get("chat") or {}).get("id") or 0)
    text = str(message.get("text") or "").strip()
    command = get_command(text)

    # Satu-satunya command yang benar-benar dimiliki launcher.
    if command == "/try":
        try:
            await start_or_reload_main(chat_id)
            # Welcome / startup message utama berasal dari main.py melalui on_start().
            if not MAIN_RUNNING:
                tg_send(chat_id, "❌ main.py gagal aktif.")
        except Exception as exc:
            log.exception("Gagal menjalankan main.py")
            tg_send(chat_id, f"❌ Gagal menjalankan main.py\n{exc}")
        return

    # Selain /try: semuanya diteruskan ke main.py.
    module = MAIN_MODULE
    if module is None or not MAIN_RUNNING:
        tg_send(chat_id, "ℹ️ main.py belum aktif. Kirim /try terlebih dahulu.")
        return

    try:
        context = {
            "chat_id": chat_id,
            "user_id": int((message.get("from") or {}).get("id") or 0),
            "send_message": tg_send,
            "launcher": "try.py",
            "main_file": str(BASE_DIR / MAIN_FILE),
            "is_running": lambda: MAIN_RUNNING,
        }

        await _call_update(module, update, context)
    except Exception as exc:
        log.exception("main.py gagal memproses update")
        tg_send(chat_id, f"❌ main.py error\n{exc}")


async def _call_update(module: ModuleType, update: dict, context: dict) -> None:
    handler = getattr(module, "handle_update")
    result = handler(update, context)
    if asyncio.iscoroutine(result):
        await result


# -----------------------------------------------------------------------------
# TELEGRAM POLLING
# -----------------------------------------------------------------------------
async def telegram_loop() -> None:
    offset: int | None = None
    backoff = 2

    # Launcher memakai polling. Pastikan webhook lama tidak mengganggu getUpdates.
    try:
        tg_call("deleteWebhook", {"drop_pending_updates": False}, timeout=20)
    except Exception:
        log.exception("Gagal deleteWebhook")

    tg_send(
        ALLOWED_USER_ID,
        "🚀 Launcher online.\n\nKirim /try untuk menjalankan main.py.",
    )

    while not STOP_EVENT.is_set():
        try:
            payload = {
                "timeout": TG_POLL_TIMEOUT,
                "allowed_updates": ["message"],
            }
            if offset is not None:
                payload["offset"] = offset

            updates = await asyncio.to_thread(
                tg_call,
                "getUpdates",
                payload,
                TG_POLL_TIMEOUT + 10,
            )
            backoff = 2

            for update in updates or []:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = update_id + 1

                message = update.get("message")
                if isinstance(message, dict):
                    await route_message(message, update)

        except Exception as exc:
            log.warning("Telegram polling error: %s", exc)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# -----------------------------------------------------------------------------
# PROCESS ENTRY POINT
# -----------------------------------------------------------------------------
async def async_main() -> None:
    threading.Thread(
        target=run_http_server,
        name="http-health",
        daemon=True,
    ).start()

    try:
        await telegram_loop()
    finally:
        STOP_EVENT.set()
        await stop_main()


if __name__ == "__main__":
    asyncio.run(async_main())
