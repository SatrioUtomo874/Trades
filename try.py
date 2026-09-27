from __future__ import annotations

"""
try.py - Telegram Launcher

Tugas launcher:
1. Menjalankan polling Telegram.
2. /try menyinkronkan main.py + strategy.py dari GitHub lalu reload main.py.
3. /end menghentikan main.py.
4. /ganti mengganti main.py atau strategy.py di GitHub dan menyalin versi baru
   ke runtime lokal.
5. Meneruskan command/update lain ke main.py.
6. Menyediakan endpoint HTTP sederhana untuk Render health check.

main.py tetap menjadi engine. strategy.py tetap menjadi modul analisis terpisah.
"""

import ast
import asyncio
import base64
import importlib.util
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType
from urllib.parse import quote

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
STRATEGY_FILE = (os.getenv("STRATEGY_FILE") or "strategy.py").strip()
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()
PORT = int(os.getenv("PORT", "10000"))
TG_POLL_TIMEOUT = max(5, int(os.getenv("TG_POLL_TIMEOUT", "30")))
HTTP_TIMEOUT = max(10, int(os.getenv("HTTP_TIMEOUT", "30")))
GITHUB_REQUEST_SECONDS = max(10, int(os.getenv("GITHUB_REQUEST_SECONDS", "30")))

try:
    ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
except ValueError as exc:
    raise RuntimeError("ALLOWED_USER_ID harus berupa integer.") from exc

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN belum diset di .env")
if not ALLOWED_USER_ID:
    raise RuntimeError("ALLOWED_USER_ID belum diset di .env")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN belum diset di .env")
if not REPO_NAME or "/" not in REPO_NAME:
    raise RuntimeError("REPO_NAME belum diset atau formatnya tidak valid.")


# -----------------------------------------------------------------------------
# STATE
# -----------------------------------------------------------------------------
log = logging.getLogger("launcher")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
TG_FILE_API = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}"
GITHUB_API = "https://api.github.com"

MAIN_LOCK = asyncio.Lock()
MAIN_MODULE: ModuleType | None = None
MAIN_RUNNING = False
STOP_EVENT = asyncio.Event()

# /ganti state. Hanya ada satu user yang diizinkan sehingga state global cukup.
GANTI_WAITING_SELECTION = False
GANTI_TARGET: str | None = None


# -----------------------------------------------------------------------------
# PATH HELPERS
# -----------------------------------------------------------------------------
def safe_runtime_path(relative_path: str) -> Path:
    """Resolve path lokal dan cegah path keluar dari folder launcher."""
    raw = Path(str(relative_path).strip())
    if raw.is_absolute():
        raise ValueError(f"Runtime path harus relatif: {relative_path}")

    root = BASE_DIR.resolve()
    path = (BASE_DIR / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Runtime path keluar dari BASE_DIR: {relative_path}") from exc
    return path


def github_content_url(path: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path.strip("/").split("/"))
    return f"{GITHUB_API}/repos/{REPO_NAME}/contents/{encoded}"


def github_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }


# -----------------------------------------------------------------------------
# SMALL HTTP SERVER - Render health check
# -----------------------------------------------------------------------------
app = Flask(__name__)


@app.get("/")
def index():
    return jsonify(
        {
            "ok": True,
            "service": "telegram-launcher",
            "main_running": MAIN_RUNNING,
            "main_file": MAIN_FILE,
            "strategy_file": STRATEGY_FILE,
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
            "main_file": MAIN_FILE,
            "strategy_file": STRATEGY_FILE,
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
# TELEGRAM
# -----------------------------------------------------------------------------
def tg_call(
    method: str,
    payload: dict | None = None,
    timeout: int = HTTP_TIMEOUT,
):
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
    text = str(text)
    for start in range(0, len(text), 3900):
        chunk = text[start : start + 3900]
        try:
            tg_call("sendMessage", {"chat_id": chat_id, "text": chunk})
        except Exception:
            log.exception("Gagal mengirim Telegram ke chat_id=%s", chat_id)
            return


def tg_send_document(chat_id: int, document_path: str, caption: str = "") -> None:
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
        raise RuntimeError(f"Telegram sendDocument: {body}")


def download_telegram_document(file_id: str) -> tuple[bytes, str]:
    result = tg_call("getFile", {"file_id": file_id}, timeout=HTTP_TIMEOUT)
    file_path = str((result or {}).get("file_path") or "").strip()
    if not file_path:
        raise RuntimeError("Telegram getFile tidak mengembalikan file_path.")

    response = requests.get(
        f"{TG_FILE_API}/{file_path}",
        timeout=HTTP_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Telegram file download: HTTP {response.status_code}: "
            f"{response.text[:800]}"
        )

    return response.content, Path(file_path).name


# -----------------------------------------------------------------------------
# SOURCE VALIDATION
# -----------------------------------------------------------------------------
def _function_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def validate_python_source(source: bytes, relative_path: str) -> None:
    text = source.decode("utf-8")
    try:
        compile(text, relative_path, "exec")
        names = _function_names(text)
    except SyntaxError as exc:
        location = f"line {exc.lineno or '?'}"
        raise RuntimeError(
            f"{relative_path} memiliki syntax error di {location}: {exc.msg}"
        ) from exc

    normalized = Path(relative_path).name.lower()
    if normalized == Path(MAIN_FILE).name.lower():
        required = {"on_start", "handle_update", "on_stop"}
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(
                f"{relative_path} wajib memiliki function: {', '.join(missing)}"
            )

    if normalized == Path(STRATEGY_FILE).name.lower():
        if "generate_setup" not in names:
            raise RuntimeError(
                f"{relative_path} wajib menyediakan function generate_setup()."
            )


# -----------------------------------------------------------------------------
# GITHUB
# -----------------------------------------------------------------------------
def github_get_file(path: str) -> tuple[bytes, str | None]:
    response = requests.get(
        github_content_url(path),
        headers=github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=GITHUB_REQUEST_SECONDS,
    )

    if response.status_code == 404:
        raise FileNotFoundError(
            f"File {path} tidak ditemukan di GitHub branch {GITHUB_BRANCH}."
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub GET {path}: HTTP {response.status_code}: "
            f"{response.text[:800]}"
        )

    body = response.json()
    if body.get("type") != "file":
        raise RuntimeError(f"GitHub path bukan file: {path}")

    encoded = str(body.get("content") or "").replace("\n", "")
    try:
        content = base64.b64decode(encoded)
    except Exception as exc:
        raise RuntimeError(f"GitHub file {path} gagal di-decode.") from exc

    return content, str(body.get("sha") or "") or None


def github_put_file(path: str, content: bytes, commit_message: str) -> str:
    last_error: Exception | None = None

    for _attempt in range(3):
        try:
            try:
                _old_content, sha = github_get_file(path)
            except FileNotFoundError:
                sha = None

            payload = {
                "message": commit_message,
                "content": base64.b64encode(content).decode("ascii"),
                "branch": GITHUB_BRANCH,
            }
            if sha:
                payload["sha"] = sha

            response = requests.put(
                github_content_url(path),
                headers=github_headers(),
                json=payload,
                timeout=GITHUB_REQUEST_SECONDS,
            )

            if response.status_code == 409:
                raise RuntimeError("GITHUB_CONFLICT")

            if response.status_code >= 400:
                raise RuntimeError(
                    f"GitHub PUT {path}: HTTP {response.status_code}: "
                    f"{response.text[:800]}"
                )

            body = response.json()
            return str((body.get("content") or {}).get("sha") or "")

        except RuntimeError as exc:
            last_error = exc
            if str(exc) != "GITHUB_CONFLICT":
                raise
            time.sleep(0.5)
        except Exception as exc:
            last_error = exc
            raise

    raise RuntimeError(f"GitHub gagal memperbarui {path}: {last_error}")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except Exception:
                pass


# -----------------------------------------------------------------------------
# SYNC MAIN + STRATEGY
# -----------------------------------------------------------------------------
def sync_runtime_files() -> list[str]:
    """Download main.py + strategy.py dari GitHub dan replace secara atomik.

    Kedua file divalidasi sebelum satu pun file lokal diganti. Jadi /try tidak
    meninggalkan runtime dalam keadaan setengah-ter-update ketika salah satu
    sumber invalid.
    """
    targets = [MAIN_FILE, STRATEGY_FILE]
    downloaded: dict[str, bytes] = {}

    for target in targets:
        content, _sha = github_get_file(target)
        validate_python_source(content, target)
        downloaded[target] = content

    for target, content in downloaded.items():
        atomic_write(safe_runtime_path(target), content)

    return targets


# -----------------------------------------------------------------------------
# MAIN LOADER
# -----------------------------------------------------------------------------
def load_main_module() -> ModuleType:
    path = safe_runtime_path(MAIN_FILE)
    if not path.exists():
        raise FileNotFoundError(f"{MAIN_FILE} tidak ditemukan di {BASE_DIR}")

    source = path.read_text(encoding="utf-8")
    code = compile(source, str(path), "exec")

    module_name = "trading_main_runtime"
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak dapat membuat module spec untuk {MAIN_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

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
    """Sync main.py + strategy.py lalu start/reload main.py."""
    global MAIN_MODULE, MAIN_RUNNING, GANTI_TARGET, GANTI_WAITING_SELECTION

    async with MAIN_LOCK:
        # Sync dilakukan SEBELUM main lama dihentikan.
        synced = await asyncio.to_thread(sync_runtime_files)
        log.info("/try sync selesai: %s", ", ".join(synced))

        old_module = MAIN_MODULE

        if old_module is not None and MAIN_RUNNING:
            log.info("/try diterima saat main aktif -> reload main.py")
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
                        "main_file": str(safe_runtime_path(MAIN_FILE)),
                        "strategy_file": str(safe_runtime_path(STRATEGY_FILE)),
                    },
                )
            except Exception:
                log.exception("on_stop() main lama gagal")

        MAIN_MODULE = None
        sys.modules.pop("trading_main_runtime", None)

        try:
            module = load_main_module()

            required = ("on_start", "handle_update", "on_stop")
            missing = [
                name
                for name in required
                if not callable(getattr(module, name, None))
            ]
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
                "main_file": str(safe_runtime_path(MAIN_FILE)),
                "strategy_file": str(safe_runtime_path(STRATEGY_FILE)),
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

            GANTI_WAITING_SELECTION = False
            GANTI_TARGET = None
            log.info("main.py aktif")

        except Exception:
            MAIN_RUNNING = False
            MAIN_MODULE = None
            sys.modules.pop("trading_main_runtime", None)
            raise


async def stop_main() -> bool:
    """Hentikan main.py. Return True jika sebelumnya aktif."""
    global MAIN_MODULE, MAIN_RUNNING

    async with MAIN_LOCK:
        module = MAIN_MODULE
        was_running = bool(module is not None and MAIN_RUNNING)
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
                        "main_file": str(safe_runtime_path(MAIN_FILE)),
                        "strategy_file": str(safe_runtime_path(STRATEGY_FILE)),
                    },
                )
            except Exception:
                log.exception("on_stop() gagal")

        sys.modules.pop("trading_main_runtime", None)
        return was_running


# -----------------------------------------------------------------------------
# /GANTI
# -----------------------------------------------------------------------------
def _ganti_menu() -> str:
    return (
        "🔄 GANTI FILE\n\n"
        "Pilih file yang ingin diganti:\n\n"
        f"1. {MAIN_FILE}\n"
        f"2. {STRATEGY_FILE}\n\n"
        "Setelah memilih, kirim file .py sebagai Document Telegram."
    )


def _resolve_ganti_target(value: str) -> str | None:
    answer = str(value or "").strip().lower()
    main_name = Path(MAIN_FILE).name.lower()
    strategy_name = Path(STRATEGY_FILE).name.lower()

    if answer in {"1", main_name, MAIN_FILE.lower(), "/ganti " + main_name}:
        return MAIN_FILE
    if answer in {"2", strategy_name, STRATEGY_FILE.lower(), "/ganti " + strategy_name}:
        return STRATEGY_FILE
    return None


async def start_ganti(chat_id: int, argument: str = "") -> None:
    global GANTI_WAITING_SELECTION, GANTI_TARGET

    target = _resolve_ganti_target(argument) if argument else None
    if target:
        GANTI_WAITING_SELECTION = False
        GANTI_TARGET = target
        tg_send(
            chat_id,
            f"📎 TARGET GANTI: {target}\n\n"
            "Sekarang kirim file Python sebagai Document Telegram.\n"
            "Gunakan /back untuk membatalkan.",
        )
        return

    GANTI_WAITING_SELECTION = True
    GANTI_TARGET = None
    tg_send(chat_id, _ganti_menu())


async def _handle_ganti_selection(chat_id: int, text: str) -> None:
    global GANTI_WAITING_SELECTION, GANTI_TARGET

    target = _resolve_ganti_target(text)
    if not target:
        tg_send(
            chat_id,
            "❌ Pilihan tidak valid.\n\n" + _ganti_menu(),
        )
        return

    GANTI_WAITING_SELECTION = False
    GANTI_TARGET = target
    tg_send(
        chat_id,
        f"📎 TARGET GANTI: {target}\n\n"
        "Sekarang kirim file Python sebagai Document Telegram.\n"
        "Gunakan /back untuk membatalkan.",
    )


async def handle_ganti_document(chat_id: int, document: dict) -> None:
    global GANTI_WAITING_SELECTION, GANTI_TARGET

    target = GANTI_TARGET
    if not target:
        tg_send(chat_id, "❌ Belum ada target /ganti. Kirim /ganti terlebih dahulu.")
        return

    file_id = str(document.get("file_id") or "").strip()
    original_name = str(document.get("file_name") or "document.py").strip()

    if not file_id:
        tg_send(chat_id, "❌ Telegram document tidak memiliki file_id.")
        return

    if not original_name.lower().endswith(".py"):
        tg_send(chat_id, "❌ /ganti hanya menerima file Python (.py).")
        return

    tg_send(chat_id, f"⏳ Memproses {original_name} → {target} ...")

    try:
        content, telegram_name = await asyncio.to_thread(
            download_telegram_document,
            file_id,
        )
        del telegram_name

        validate_python_source(content, target)

        sha = await asyncio.to_thread(
            github_put_file,
            target,
            content,
            f"launcher: replace {target}",
        )

        atomic_write(
            safe_runtime_path(target),
            content,
        )

        GANTI_WAITING_SELECTION = False
        GANTI_TARGET = None

        tg_send(
            chat_id,
            "✅ FILE BERHASIL DIGANTI\n\n"
            f"Target: {target}\n"
            f"GitHub: {REPO_NAME}/{target}\n"
            f"Branch: {GITHUB_BRANCH}\n"
            f"Commit SHA: {(sha[:10] + '...') if sha else '-'}\n\n"
            "Runtime lokal juga sudah diperbarui.\n"
            "Gunakan /try untuk reload main.py + memastikan kedua file "
            "sinkron kembali dari GitHub.",
        )

    except Exception as exc:
        log.exception("/ganti gagal untuk %s", target)
        tg_send(
            chat_id,
            "❌ /ganti gagal\n\n"
            f"Target: {target}\n"
            f"Error: {exc}",
        )


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


def get_argument(text: str) -> str:
    parts = str(text or "").strip().split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else ""


async def route_message(message: dict, update: dict) -> None:
    global GANTI_WAITING_SELECTION, GANTI_TARGET

    if not authorized(message):
        return

    chat_id = int((message.get("chat") or {}).get("id") or 0)
    text = str(message.get("text") or "").strip()
    command = get_command(text)
    document = message.get("document")

    # /back selalu tersedia di launcher untuk membatalkan /ganti.
    if command == "/back" and (GANTI_WAITING_SELECTION or GANTI_TARGET):
        GANTI_WAITING_SELECTION = False
        GANTI_TARGET = None
        tg_send(chat_id, "✅ /ganti dibatalkan.")
        return

    if command == "/try":
        try:
            await start_or_reload_main(chat_id)
            if MAIN_RUNNING:
                tg_send(
                    chat_id,
                    "🔄 /try selesai.\n"
                    f"Sinkron: {MAIN_FILE} + {STRATEGY_FILE}\n"
                    "main.py aktif.",
                )
            else:
                tg_send(chat_id, "❌ main.py gagal aktif.")
        except Exception as exc:
            log.exception("Gagal menjalankan /try")
            tg_send(chat_id, f"❌ Gagal menjalankan /try\n\n{exc}")
        return

    if command == "/end":
        try:
            stopped = await stop_main()
            if stopped:
                tg_send(chat_id, "🛑 main.py dihentikan. Runtime state main.py sudah dibersihkan oleh on_stop().")
            else:
                tg_send(chat_id, "ℹ️ main.py memang sedang tidak aktif.")
        except Exception as exc:
            log.exception("Gagal menjalankan /end")
            tg_send(chat_id, f"❌ /end gagal\n\n{exc}")
        return

    if command == "/ganti":
        await start_ganti(chat_id, get_argument(text))
        return

    if command == "/healthz":
        tg_send(
            chat_id,
            "HEALTHZ\n\n"
            f"Launcher: ONLINE\n"
            f"Main: {'RUNNING' if MAIN_RUNNING else 'STOPPED'}\n"
            f"Main File: {MAIN_FILE}\n"
            f"Strategy File: {STRATEGY_FILE}",
        )
        return

    # Upload document diprioritaskan jika /ganti sedang menunggu file.
    if document is not None and GANTI_TARGET:
        await handle_ganti_document(chat_id, document)
        return

    if GANTI_WAITING_SELECTION and text:
        await _handle_ganti_selection(chat_id, text)
        return

    # Selain command launcher: teruskan semuanya ke main.py.
    module = MAIN_MODULE
    if module is None or not MAIN_RUNNING:
        tg_send(chat_id, "ℹ️ main.py belum aktif. Kirim /try terlebih dahulu.")
        return

    try:
        context = {
            "chat_id": chat_id,
            "user_id": int((message.get("from") or {}).get("id") or 0),
            "send_message": tg_send,
            "send_document": tg_send_document,
            "launcher": "try.py",
            "main_file": str(safe_runtime_path(MAIN_FILE)),
            "strategy_file": str(safe_runtime_path(STRATEGY_FILE)),
            "is_running": lambda: MAIN_RUNNING,
        }

        await _call_update(module, update, context)
    except Exception as exc:
        log.exception("main.py gagal memproses update")
        tg_send(chat_id, f"❌ main.py error\n{exc}")


async def _call_update(module: ModuleType, update: dict, context: dict) -> None:
    handler = getattr(module, "handle_update", None)
    if not callable(handler):
        raise RuntimeError(f"{MAIN_FILE} tidak memiliki handle_update().")

    result = handler(update, context)
    if asyncio.iscoroutine(result):
        await result


# -----------------------------------------------------------------------------
# TELEGRAM POLLING
# -----------------------------------------------------------------------------
async def telegram_loop() -> None:
    offset: int | None = None
    backoff = 2

    try:
        tg_call("deleteWebhook", {"drop_pending_updates": False}, timeout=20)
    except Exception:
        log.exception("Gagal deleteWebhook")

    tg_send(
        ALLOWED_USER_ID,
        "🚀 Launcher online.\n\n"
        "Gunakan /try untuk sinkronisasi + menjalankan main.py.\n"
        "Gunakan /ganti untuk mengganti main.py / strategy.py.",
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
