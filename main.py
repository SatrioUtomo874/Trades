from __future__ import annotations

"""
SMCAutoTrade PERMANENT LAUNCHER

main.py hanya bertanggung jawab atas:
  - Flask/Render HTTP server
  - satu Telegram polling loop
  - /try  -> load + start start.py
  - /end  -> stop + unload start.py
  - /ganti -> upload/replace file ke GitHub
  - semua update Telegram lainnya -> start.py

IMPORTANT:
  start.py TIDAK boleh melakukan Telegram getUpdates/polling sendiri.
  Semua Telegram traffic masuk lewat launcher ini.

START.PY CONTRACT (fleksibel):
  def on_start(context): optional
  def on_stop(context): optional
  def handle_update(update, context): recommended

Legacy-compatible fallback:
  def handle_command(text, context): optional

Context yang diberikan ke start.py berisi:
  chat_id, user_id, stop_event, send_message, launcher, start_file
"""

import base64
import hashlib
import importlib.util
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import quote

import requests
from flask import Flask, jsonify

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()
START_FILE = Path(os.getenv("START_FILE", "start.py")).resolve()
PORT = int(os.getenv("PORT", "10000"))
TG_POLL_TIMEOUT = max(5, int(os.getenv("TG_POLL_TIMEOUT", "30")))
TG_ERROR_BACKOFF_MAX = max(10, int(os.getenv("TG_ERROR_BACKOFF_MAX", "30")))
START_STOP_TIMEOUT = max(1, int(os.getenv("START_STOP_TIMEOUT", "10")))
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()

try:
    ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
except ValueError as exc:
    raise RuntimeError("ALLOWED_USER_ID harus berupa integer.") from exc

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN belum diset.")
if ALLOWED_USER_ID == 0:
    raise RuntimeError("ALLOWED_USER_ID belum diset atau bernilai 0.")

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("launcher")

# ---------------------------------------------------------------------------
# FLASK / RENDER
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.get("/")
def root() -> Any:
    return jsonify(
        {
            "ok": True,
            "service": "SMCAutoTrade Launcher",
            "start_running": start_is_running(),
            "start_file": START_FILE.name,
        }
    )


@app.get("/healthz")
def healthz() -> Any:
    return jsonify(
        {
            "ok": True,
            "service": "SMCAutoTrade Launcher",
            "telegram_polling": True,
            "github_configured": bool(GITHUB_TOKEN and REPO_NAME),
            "start_running": start_is_running(),
        }
    )


def run_flask() -> None:
    # Render requires the process to listen on 0.0.0.0 and on $PORT.
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


class TelegramError(RuntimeError):
    pass


class TelegramConflict(TelegramError):
    pass


def tg_call(method: str, payload: dict[str, Any] | None = None, timeout: int = 40) -> Any:
    try:
        response = requests.post(
            f"{TG_API}/{method}",
            json=payload or {},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise TelegramError(f"{method}: network error: {exc}") from exc

    if response.status_code == 409:
        raise TelegramConflict(response.text[:800])
    if response.status_code >= 400:
        raise TelegramError(
            f"{method}: HTTP {response.status_code}: {response.text[:800]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise TelegramError(f"{method}: invalid JSON response") from exc

    if not body.get("ok"):
        raise TelegramError(f"{method}: {body}")
    return body.get("result")


def tg_send(chat_id: int, text: Any) -> None:
    text = str(text)
    if len(text) <= 3900:
        try:
            tg_call("sendMessage", {"chat_id": chat_id, "text": text})
        except Exception as exc:
            log.warning("Telegram sendMessage gagal: %s", exc)
        return

    for i in range(0, len(text), 3900):
        try:
            tg_call(
                "sendMessage",
                {"chat_id": chat_id, "text": text[i : i + 3900]},
            )
        except Exception as exc:
            log.warning("Telegram sendMessage chunk gagal: %s", exc)
            return


def tg_delete_webhook() -> None:
    try:
        tg_call("deleteWebhook", {"drop_pending_updates": False}, timeout=20)
        log.info("[TG] webhook cleared; polling mode active")
    except Exception as exc:
        log.warning("[TG] deleteWebhook gagal: %s", exc)


def tg_get_file_bytes(file_id: str) -> bytes:
    info = tg_call("getFile", {"file_id": file_id}, timeout=20)
    file_path = str(info["file_path"])
    try:
        response = requests.get(
            f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}",
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TelegramError(f"download Telegram file gagal: {exc}") from exc
    return response.content


# ---------------------------------------------------------------------------
# START.PY LIFECYCLE
# ---------------------------------------------------------------------------

_START_LOCK = threading.RLock()
_START_MODULE: ModuleType | None = None
_START_RUNNING = False
_START_STOP_EVENT = threading.Event()
_START_CONTEXT: dict[str, Any] = {}


def start_is_running() -> bool:
    with _START_LOCK:
        return _START_RUNNING and _START_MODULE is not None


def _load_start_module() -> ModuleType:
    if not START_FILE.exists():
        raise FileNotFoundError(f"{START_FILE.name} tidak ditemukan di {START_FILE.parent}")

    source = START_FILE.read_text(encoding="utf-8")
    compile(source, str(START_FILE), "exec")

    # Load from exact path; this avoids relying on PYTHONPATH and supports a
    # future start.py replacement without touching the launcher.
    module_name = f"launcher_start_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, START_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak bisa membuat module spec untuk {START_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _build_start_context(chat_id: int | None = None, user_id: int | None = None) -> dict[str, Any]:
    return {
        "launcher": "main.py",
        "start_file": str(START_FILE),
        "chat_id": chat_id,
        "user_id": user_id,
        "stop_event": _START_STOP_EVENT,
        "send_message": tg_send,
        "is_running": start_is_running,
    }


def start_bot(chat_id: int) -> str:
    global _START_MODULE, _START_RUNNING, _START_CONTEXT

    with _START_LOCK:
        if start_is_running():
            return f"▶️ {START_FILE.name} sudah berjalan."

        module = _load_start_module()
        _START_STOP_EVENT.clear()
        context = _build_start_context(chat_id, ALLOWED_USER_ID)

        on_start = getattr(module, "on_start", None)
        if callable(on_start):
            result = on_start(dict(context))
            if result is False:
                raise RuntimeError("start.py menolak startup melalui on_start().")

        _START_MODULE = module
        _START_CONTEXT = context
        _START_RUNNING = True

    log.info("[START] %s aktif", START_FILE.name)
    return (
        f"🟢 <b>{START_FILE.name} aktif.</b>\n"
        "Telegram polling tetap dijalankan oleh main.py.\n"
        "Semua command lain diteruskan utuh ke start.py."
    )


def _unload_start_module(module: ModuleType | None) -> None:
    if module is None:
        return
    sys.modules.pop(module.__name__, None)


def stop_bot() -> str:
    global _START_MODULE, _START_RUNNING, _START_CONTEXT

    with _START_LOCK:
        module = _START_MODULE
        if not _START_RUNNING or module is None:
            _START_RUNNING = False
            _START_MODULE = None
            _START_CONTEXT = {}
            _START_STOP_EVENT.set()
            return f"ℹ️ {START_FILE.name} sedang tidak berjalan."

        _START_STOP_EVENT.set()
        try:
            on_stop = getattr(module, "on_stop", None)
            if callable(on_stop):
                result = on_stop(dict(_START_CONTEXT))
                if result is not False:
                    log.info("[END] on_stop() selesai")
        except Exception:
            log.exception("[END] on_stop() gagal")
            # /end must still unload the module and mark it stopped.
        finally:
            _START_RUNNING = False
            _START_MODULE = None
            _START_CONTEXT = {}
            _unload_start_module(module)

    log.info("[END] %s unloaded", START_FILE.name)
    return f"⏹️ <b>{START_FILE.name} dihentikan.</b> Launcher tetap hidup."


def forward_update(update: dict[str, Any]) -> str | None:
    with _START_LOCK:
        module = _START_MODULE
        if not _START_RUNNING or module is None:
            return "ℹ️ start.py belum berjalan. Gunakan /try terlebih dahulu."

        context = dict(_START_CONTEXT)
        message = update.get("message") or {}
        context["chat_id"] = (message.get("chat") or {}).get("id")
        context["user_id"] = (message.get("from") or {}).get("id")

        # Preferred extensible contract: receives the complete Telegram update.
        handler = getattr(module, "handle_update", None)
        if callable(handler):
            result = handler(update, context)
            return None if result is None else str(result)

        # Backward-compatible simple contract.
        handler = getattr(module, "handle_command", None)
        if callable(handler):
            text = str(message.get("text") or message.get("caption") or "").strip()
            result = handler(text, context)
            return None if result is None else str(result)

        return (
            "⚠️ start.py aktif tetapi belum menyediakan `handle_update(update, context)` "
            "atau `handle_command(text, context)`."
        )


# ---------------------------------------------------------------------------
# GITHUB /GANTI
# ---------------------------------------------------------------------------

_SAFE_PATH = re.compile(r"^[A-Za-z0-9._/\-]+$")


def validate_github_path(path: str) -> str:
    path = str(path).strip().lstrip("/")
    if not path or ".." in Path(path).parts or not _SAFE_PATH.fullmatch(path):
        raise ValueError("Path GitHub tidak valid.")
    return path


def _github_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_get_sha(path: str) -> str | None:
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    response = requests.get(
        f"https://api.github.com/repos/{REPO_NAME}/contents/{encoded}",
        headers=_github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=20,
    )
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub GET HTTP {response.status_code}: {response.text[:600]}"
        )
    return response.json().get("sha")


def github_replace(path: str, content: bytes) -> str:
    if not GITHUB_TOKEN or not REPO_NAME:
        raise RuntimeError("GITHUB_TOKEN atau REPO_NAME belum diset.")

    path = validate_github_path(path)
    sha = github_get_sha(path)
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))

    payload: dict[str, Any] = {
        "message": f"Replace {path} via /ganti",
        "content": base64.b64encode(content).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(
        f"https://api.github.com/repos/{REPO_NAME}/contents/{encoded}",
        headers=_github_headers(),
        json=payload,
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub PUT HTTP {response.status_code}: {response.text[:800]}"
        )

    return str((response.json().get("commit") or {}).get("sha") or "")[:12]


def handle_ganti(message: dict[str, Any]) -> str:
    document = message.get("document")
    if not isinstance(document, dict):
        return (
            "📦 <b>/ganti</b>\n\n"
            "Kirim file sebagai attachment dengan caption:\n"
            "<code>/ganti</code> → gunakan nama file asli\n"
            "<code>/ganti folder/nama.py</code> → gunakan path GitHub tersebut."
        )

    caption = str(message.get("caption") or "").strip()
    parts = caption.split(maxsplit=1)
    requested_path = parts[1].strip() if len(parts) > 1 else str(document.get("file_name") or "")
    if not requested_path:
        return "❌ Nama/path file tidak ditemukan."

    try:
        path = validate_github_path(requested_path)
        content = tg_get_file_bytes(str(document["file_id"]))
        commit = github_replace(path, content)
        digest = hashlib.sha256(content).hexdigest()[:16]
        return (
            "✅ <b>/ganti berhasil</b>\n"
            f"File: <code>{document.get('file_name', path)}</code>\n"
            f"GitHub: <code>{path}</code>\n"
            f"Branch: <code>{GITHUB_BRANCH}</code>\n"
            f"Commit: <code>{commit or 'created'}</code>\n"
            f"SHA256: <code>{digest}</code>"
        )
    except Exception as exc:
        log.exception("[GANTI] gagal")
        return f"❌ <b>/ganti gagal</b>\n<code>{exc}</code>"


# ---------------------------------------------------------------------------
# MESSAGE ROUTER
# ---------------------------------------------------------------------------


def is_authorized(message: dict[str, Any]) -> bool:
    chat_id = int((message.get("chat") or {}).get("id") or 0)
    user_id = int((message.get("from") or {}).get("id") or 0)
    return chat_id == ALLOWED_USER_ID and user_id == ALLOWED_USER_ID


def route_message(message: dict[str, Any], update: dict[str, Any]) -> None:
    if not is_authorized(message):
        return

    chat_id = int((message.get("chat") or {}).get("id"))
    text = str(message.get("text") or "").strip()
    caption = str(message.get("caption") or "").strip()

    # /ganti + document is handled by launcher and never forwarded to start.py.
    if isinstance(message.get("document"), dict) and caption.lower().startswith("/ganti"):
        tg_send(chat_id, handle_ganti(message))
        return

    command = text.split(maxsplit=1)[0].split("@", 1)[0].lower() if text else ""

    try:
        if command == "/try":
            tg_send(chat_id, start_bot(chat_id))
            return

        if command == "/end":
            tg_send(chat_id, stop_bot())
            return

        if command == "/ganti":
            tg_send(chat_id, handle_ganti(message))
            return

        # /start and /help are launcher-level only when start.py is not active.
        # Once start.py is active they are forwarded, so start.py can customize them.
        if command == "/help" and not start_is_running():
            tg_send(
                chat_id,
                "🤖 <b>Launcher</b>\n\n"
                "/try — jalankan start.py\n"
                "/end — hentikan start.py\n"
                "/ganti — upload/replace file di GitHub\n"
                "Command lain bekerja setelah start.py aktif.",
            )
            return

        # EVERYTHING ELSE belongs to start.py.
        result = forward_update(update)
        if result:
            tg_send(chat_id, result)

    except Exception as exc:
        log.exception("[ROUTER] command %s gagal", command or "<empty>")
        tg_send(chat_id, f"❌ Command gagal: <code>{exc}</code>")


# ---------------------------------------------------------------------------
# TELEGRAM POLLING
# ---------------------------------------------------------------------------

_STOP = threading.Event()


def telegram_loop() -> None:
    tg_delete_webhook()
    offset: int | None = None
    backoff = 2

    while not _STOP.is_set():
        try:
            payload: dict[str, Any] = {
                "timeout": TG_POLL_TIMEOUT,
                "allowed_updates": ["message"],
            }
            if offset is not None:
                payload["offset"] = offset

            updates = tg_call(
                "getUpdates",
                payload,
                timeout=TG_POLL_TIMEOUT + 10,
            )
            backoff = 2

            for update in updates or []:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = update_id + 1
                message = update.get("message")
                if isinstance(message, dict):
                    route_message(message, update)

        except TelegramConflict as exc:
            # Critical: Telegram 409 must not kill Flask/Render or the launcher.
            log.error(
                "[TG POLLING CONFLICT] token dipakai webhook/instance lain: %s",
                exc,
            )
            time.sleep(min(backoff, TG_ERROR_BACKOFF_MAX))
            backoff = min(backoff * 2, TG_ERROR_BACKOFF_MAX)

        except TelegramError as exc:
            log.warning("[TG POLLING] %s", exc)
            time.sleep(min(backoff, TG_ERROR_BACKOFF_MAX))
            backoff = min(backoff * 2, TG_ERROR_BACKOFF_MAX)

        except Exception:
            log.exception("[TG POLLING] unexpected error")
            time.sleep(min(backoff, TG_ERROR_BACKOFF_MAX))
            backoff = min(backoff * 2, TG_ERROR_BACKOFF_MAX)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    # Start HTTP first: Render can detect the port even while Telegram is slow.
    threading.Thread(target=run_flask, name="render-http", daemon=True).start()
    threading.Thread(target=telegram_loop, name="telegram-poller", daemon=True).start()

    log.info(
        "[LAUNCHER] ready | port=%s | start=%s | github=%s",
        PORT,
        START_FILE,
        bool(GITHUB_TOKEN and REPO_NAME),
    )

    try:
        while not _STOP.wait(60):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        _STOP.set()
        try:
            stop_bot()
        except Exception:
            log.exception("[SHUTDOWN] start.py cleanup gagal")


if __name__ == "__main__":
    main()
