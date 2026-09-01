from __future__ import annotations

"""
SMCAutoTrade LAUNCHER V4

Design goals:
- Telegram polling is owned ONLY by this launcher.
- GitHub repository is the source of truth for project Python files.
- Before /try, sync the repository's Python files into the Render filesystem.
- Do NOT hard-code/pin strategy.py in the launcher.
- START_FILE remains configurable, but markdown/link accidents are normalized.
- /ganti writes to GitHub and updates the local cache immediately.
- Any Python module imported by start.py can therefore be refreshed from GitHub.
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
from urllib.parse import quote, urlparse

import requests
from flask import Flask, jsonify

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()

_RAW_START_FILE = (os.getenv("START_FILE") or "start.py").strip()
PORT = int(os.getenv("PORT", "10000"))
TG_POLL_TIMEOUT = max(5, int(os.getenv("TG_POLL_TIMEOUT", "30")))
TG_ERROR_BACKOFF_MAX = max(10, int(os.getenv("TG_ERROR_BACKOFF_MAX", "30")))
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()

try:
    ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
except ValueError as exc:
    raise RuntimeError("ALLOWED_USER_ID harus berupa integer.") from exc

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN belum diset.")
if ALLOWED_USER_ID == 0:
    raise RuntimeError("ALLOWED_USER_ID belum diset atau bernilai 0.")


def normalize_file_target(value: str, default: str = "start.py") -> str:
    """Turn accidental Markdown/URL values into a safe repository-relative path."""
    raw = str(value or "").strip()
    if not raw:
        return default

    # Markdown link: [strategy.py](https://...)
    m = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", raw)
    if m:
        raw = m.group(1).strip()

    # Full URL: https://host/path/strategy.py
    if "://" in raw:
        parsed = urlparse(raw)
        raw = Path(parsed.path).name or default

    # Strip surrounding quotes/backticks and leading slash.
    raw = raw.strip(" `\"'").lstrip("/")
    raw = raw.replace("\\", "/")

    # Never allow traversal.
    parts = [p for p in raw.split("/") if p not in {"", "."}]
    if ".." in parts:
        return default
    raw = "/".join(parts)

    if not raw or raw.endswith("/"):
        return default
    return raw


START_FILE = (BASE_DIR / normalize_file_target(_RAW_START_FILE)).resolve()
try:
    START_FILE.relative_to(BASE_DIR)
except ValueError as exc:
    raise RuntimeError("START_FILE harus berada di dalam folder aplikasi.") from exc

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("launcher.v4")

# ---------------------------------------------------------------------------
# FLASK / RENDER
# ---------------------------------------------------------------------------

app = Flask(__name__)


def start_is_running() -> bool:
    with _START_LOCK:
        return _START_RUNNING and _START_MODULE is not None


@app.get("/")
def root() -> Any:
    return jsonify(
        {
            "ok": True,
            "service": "SMCAutoTrade Launcher V4",
            "start_running": start_is_running(),
            "start_file": str(START_FILE.relative_to(BASE_DIR)),
            "github_sync": bool(GITHUB_TOKEN and REPO_NAME),
        }
    )


@app.get("/healthz")
def healthz() -> Any:
    return jsonify(
        {
            "ok": True,
            "service": "SMCAutoTrade Launcher V4",
            "telegram_polling": True,
            "github_configured": bool(GITHUB_TOKEN and REPO_NAME),
            "start_running": start_is_running(),
        }
    )


def run_flask() -> None:
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
    chunks = [text[i : i + 3900] for i in range(0, len(text), 3900)] or [""]
    for chunk in chunks:
        try:
            tg_call("sendMessage", {"chat_id": chat_id, "text": chunk})
        except Exception as exc:
            log.warning("Telegram sendMessage gagal: %s", exc)
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
# GITHUB
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
    path = validate_github_path(path)
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


def github_get_file_bytes(path: str) -> bytes:
    path = validate_github_path(path)
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    response = requests.get(
        f"https://api.github.com/repos/{REPO_NAME}/contents/{encoded}",
        headers=_github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub download HTTP {response.status_code}: {response.text[:800]}"
        )
    body = response.json()
    if body.get("encoding") != "base64" or not body.get("content"):
        raise RuntimeError(f"GitHub file {path} tidak mengandung content base64.")
    return base64.b64decode(body["content"])


def github_list_tree() -> list[str]:
    """Return all repository .py files from the selected branch."""
    if not REPO_NAME or not GITHUB_TOKEN:
        raise RuntimeError("REPO_NAME/GITHUB_TOKEN belum dikonfigurasi.")

    branch_encoded = quote(GITHUB_BRANCH, safe="")
    response = requests.get(
        f"https://api.github.com/repos/{REPO_NAME}/git/trees/{branch_encoded}",
        headers=_github_headers(),
        params={"recursive": "1"},
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub tree HTTP {response.status_code}: {response.text[:800]}"
        )

    body = response.json()
    if body.get("truncated"):
        raise RuntimeError("GitHub repository tree terpotong; repo terlalu besar untuk sync aman.")

    result: list[str] = []
    for item in body.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = str(item.get("path") or "")
        if path.endswith(".py"):
            result.append(validate_github_path(path))
    return result


def sync_python_project_from_github() -> dict[str, Any]:
    """
    Synchronize every repository .py file locally, excluding launcher files.
    This keeps GitHub as source-of-truth and lets start.py discover/import
    strategy modules without the launcher knowing their names.
    """
    paths = github_list_tree()

    synced = 0
    skipped = 0
    failures: list[str] = []

    for repo_path in sorted(paths):
        # Never overwrite the currently running launcher from inside itself.
        name = Path(repo_path).name.lower()
        if name.startswith("main") and name.endswith(".py"):
            skipped += 1
            continue

        local_path = (BASE_DIR / repo_path).resolve()
        try:
            local_path.relative_to(BASE_DIR)
        except ValueError:
            failures.append(f"{repo_path}: path keluar dari BASE_DIR")
            continue

        try:
            content = github_get_file_bytes(repo_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content)
            synced += 1
            log.info("[GITHUB SYNC] %s -> %s", repo_path, local_path)
        except Exception as exc:
            failures.append(f"{repo_path}: {exc}")
            log.exception("[GITHUB SYNC] gagal: %s", repo_path)

    return {"found": len(paths), "synced": synced, "skipped": skipped, "failures": failures}

# ---------------------------------------------------------------------------
# START.PY LIFECYCLE
# ---------------------------------------------------------------------------

_START_LOCK = threading.RLock()
_START_MODULE: ModuleType | None = None
_START_RUNNING = False
_START_STOP_EVENT = threading.Event()
_START_CONTEXT: dict[str, Any] = {}


def _load_start_module() -> ModuleType:
    if not START_FILE.exists():
        raise FileNotFoundError(f"{START_FILE.name} tidak ditemukan di {START_FILE.parent}")

    source = START_FILE.read_text(encoding="utf-8")
    compile(source, str(START_FILE), "exec")

    module_name = f"launcher_start_v4_{int(time.time() * 1000)}"
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
        "launcher": "main_v4.py",
        "start_file": str(START_FILE),
        "base_dir": str(BASE_DIR),
        "repo_name": REPO_NAME,
        "github_branch": GITHUB_BRANCH,
        "chat_id": chat_id,
        "user_id": user_id,
        "stop_event": _START_STOP_EVENT,
        "send_message": tg_send,
        "is_running": start_is_running,
        "sync_project": sync_python_project_from_github,
    }


def start_bot(chat_id: int) -> str:
    global _START_MODULE, _START_RUNNING, _START_CONTEXT

    with _START_LOCK:
        if start_is_running():
            return f"▶️ {START_FILE.name} sudah berjalan."

    # GitHub is authoritative. Sync before every /try.
    if not GITHUB_TOKEN or not REPO_NAME:
        raise RuntimeError("GitHub belum dikonfigurasi; launcher tidak bisa melakukan source sync.")

    tg_send(chat_id, "🔄 <b>SYNC V4</b>\nMemeriksa repository GitHub sebelum start...")
    sync_result = sync_python_project_from_github()
    log.info("[GITHUB SYNC] result=%s", sync_result)
    if sync_result["failures"]:
        raise RuntimeError(
            "GitHub sync gagal:\n" + "\n".join(sync_result["failures"][:10])
        )

    with _START_LOCK:
        module = _load_start_module()
        _START_STOP_EVENT.clear()
        context = _build_start_context(chat_id, ALLOWED_USER_ID)

        on_start = getattr(module, "on_start", None)
        if callable(on_start):
            result = on_start(dict(context))
            if result is False:
                raise RuntimeError("start module menolak startup melalui on_start().")

        _START_MODULE = module
        _START_CONTEXT = context
        _START_RUNNING = True

    log.info("[START V4] %s aktif", START_FILE.name)
    return (
        f"🟢 <b>Launcher V4</b>\n"
        f"Start module: <code>{START_FILE.relative_to(BASE_DIR)}</code>\n"
        f"GitHub sync: ✅ {sync_result['synced']} Python files\n"
        "Telegram polling: launcher-owned"
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
                on_stop(dict(_START_CONTEXT))
        except Exception:
            log.exception("[END] on_stop gagal")
        finally:
            _START_RUNNING = False
            _START_MODULE = None
            _START_CONTEXT = {}
            _unload_start_module(module)

    return f"⏹️ <b>{START_FILE.name} dihentikan.</b> Launcher tetap hidup."


def forward_update(update: dict[str, Any]) -> str | None:
    with _START_LOCK:
        module = _START_MODULE
        if not _START_RUNNING or module is None:
            return "ℹ️ start module belum berjalan. Gunakan /try terlebih dahulu."

        context = dict(_START_CONTEXT)
        message = update.get("message") or {}
        context["chat_id"] = (message.get("chat") or {}).get("id")
        context["user_id"] = (message.get("from") or {}).get("id")

        handler = getattr(module, "handle_update", None)
        if callable(handler):
            result = handler(update, context)
            return None if result is None else str(result)

        handler = getattr(module, "handle_command", None)
        if callable(handler):
            text = str(message.get("text") or message.get("caption") or "").strip()
            result = handler(text, context)
            return None if result is None else str(result)

        return "⚠️ start module aktif tetapi tidak menyediakan handle_update/handle_command."

# ---------------------------------------------------------------------------
# /GANTI
# ---------------------------------------------------------------------------


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
    requested_path = normalize_file_target(requested_path, str(document.get("file_name") or ""))
    if not requested_path:
        return "❌ Nama/path file tidak ditemukan."

    try:
        content = tg_get_file_bytes(str(document["file_id"]))
        commit = github_replace(requested_path, content)

        local_path = (BASE_DIR / requested_path).resolve()
        local_path.relative_to(BASE_DIR)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)

        digest = hashlib.sha256(content).hexdigest()[:16]
        return (
            "✅ <b>/ganti berhasil</b>\n"
            f"File: <code>{document.get('file_name', requested_path)}</code>\n"
            f"GitHub: <code>{requested_path}</code>\n"
            f"Branch: <code>{GITHUB_BRANCH}</code>\n"
            f"Commit: <code>{commit or 'created'}</code>\n"
            f"Local cache: ✅\n"
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

        if command == "/sync":
            if not GITHUB_TOKEN or not REPO_NAME:
                tg_send(chat_id, "❌ GitHub belum dikonfigurasi.")
                return
            result = sync_python_project_from_github()
            tg_send(
                chat_id,
                "🔄 <b>SYNC COMPLETE</b>\n"
                f"Found: {result['found']}\n"
                f"Synced: {result['synced']}\n"
                f"Skipped: {result['skipped']}\n"
                f"Failures: {len(result['failures'])}",
            )
            return

        if command == "/help" and not start_is_running():
            tg_send(
                chat_id,
                "🤖 <b>Launcher V4</b>\n\n"
                "/try — sync GitHub lalu jalankan start module\n"
                "/end — hentikan start module\n"
                "/sync — sync semua Python file dari GitHub\n"
                "/ganti — upload/replace file di GitHub\n"
                "Command lain diteruskan ke start module setelah aktif.",
            )
            return

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
            log.error("[TG POLLING CONFLICT] %s", exc)
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
    log.info(
        "[LAUNCHER V4] ready | base=%s | start=%s | repo=%s | branch=%s | github=%s",
        BASE_DIR,
        START_FILE.relative_to(BASE_DIR),
        REPO_NAME,
        GITHUB_BRANCH,
        bool(GITHUB_TOKEN),
    )

    threading.Thread(target=run_flask, name="render-http", daemon=True).start()
    threading.Thread(target=telegram_loop, name="telegram-poller", daemon=True).start()

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
            log.exception("[SHUTDOWN] cleanup gagal")


if __name__ == "__main__":
    main()
