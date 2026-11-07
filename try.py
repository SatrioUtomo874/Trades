from __future__ import annotations

"""Minimal Telegram launcher. It has exactly three operational duties:

1. /try  -> load and start local main.py; forward Telegram updates to it.
2. /ganti -> accept any Telegram document(s), save each under its original
   filename locally and upload it to the GitHub repository root. No filename,
   module, timeframe, strategy, or Python-contract validation is performed.
3. /end  -> call main.py's on_stop() and unload the engine; launcher/polling
   and the Render web process remain online.

The launcher deliberately does not inspect strategy constants or trading state.
"""

import asyncio
import base64
import importlib
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
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "trades.env")

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0") or "0")
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()

# Canonical runtime/source names are deliberately fixed. Descriptive names are
# for downloaded artifacts only; GitHub root and local runtime must use these.
MAIN_FILE = "main.py"

PORT = max(1, int(os.getenv("PORT", "10000") or "10000"))
TG_POLL_TIMEOUT = max(5, min(50, int(os.getenv("TG_POLL_TIMEOUT", "30") or "30")))
HTTP_TIMEOUT = max(10, int(os.getenv("HTTP_TIMEOUT", "30") or "30"))
GITHUB_TIMEOUT = max(10, int(os.getenv("GITHUB_REQUEST_SECONDS", "30") or "30"))

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN belum diset.")
if not ALLOWED_USER_ID:
    raise RuntimeError("ALLOWED_USER_ID belum diset atau tidak valid.")


# -----------------------------------------------------------------------------
# Launcher-only runtime state (memory only; never trading data)
# -----------------------------------------------------------------------------
log = logging.getLogger("volumatic.launcher")
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

# /ganti state is launcher UI state only; it never stores trading data.
GANTI_ACTIVE = False


# -----------------------------------------------------------------------------
# Path safety + generic file upload
# -----------------------------------------------------------------------------
def safe_runtime_path(relative_path: str) -> Path:
    raw = Path(str(relative_path).strip())
    if raw.is_absolute():
        raise ValueError(f"Runtime path harus relatif: {relative_path}")
    root = BASE_DIR.resolve()
    path = (BASE_DIR / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path keluar dari folder launcher: {relative_path}") from exc
    return path


def github_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_content_url(path: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path.strip("/").split("/"))
    return f"{GITHUB_API}/repos/{REPO_NAME}/contents/{encoded}"


def github_put_source(path: str, content: bytes, commit_message: str) -> str:
    """Replace one source file in GitHub. This is source deployment, not trade persistence."""
    url = github_content_url(path)
    last_error: Exception | None = None

    for attempt in range(3):
        response = requests.get(
            url,
            headers=github_headers(),
            params={"ref": GITHUB_BRANCH},
            timeout=GITHUB_TIMEOUT,
        )
        if response.status_code == 404:
            sha = None
        elif response.status_code >= 400:
            raise RuntimeError(
                f"GitHub GET sebelum /ganti {path}: HTTP {response.status_code}: {response.text[:600]}"
            )
        else:
            body = response.json()
            if body.get("type") != "file":
                raise RuntimeError(f"Target GitHub bukan file: {path}")
            sha = str(body.get("sha") or "").strip() or None

        payload = {
            "message": commit_message,
            "content": base64.b64encode(content).decode("ascii"),
            "branch": GITHUB_BRANCH,
        }
        if sha:
            payload["sha"] = sha

        put = requests.put(
            url,
            headers=github_headers(),
            json=payload,
            timeout=GITHUB_TIMEOUT,
        )
        if put.status_code in (409, 422) and attempt < 2:
            last_error = RuntimeError(f"GitHub conflict HTTP {put.status_code}: {put.text[:400]}")
            awaitable_sleep = 0.35 * (attempt + 1)
            time.sleep(awaitable_sleep)
            continue
        if put.status_code >= 400:
            raise RuntimeError(f"GitHub PUT {path}: HTTP {put.status_code}: {put.text[:600]}")
        result = put.json()
        return str((result.get("content") or {}).get("sha") or "")

    raise RuntimeError(f"GitHub gagal mengganti {path}: {last_error}")


def download_telegram_document(file_id: str) -> tuple[bytes, str]:
    result = tg_call("getFile", {"file_id": file_id}, timeout=HTTP_TIMEOUT)
    file_path = str((result or {}).get("file_path") or "").strip()
    if not file_path:
        raise RuntimeError("Telegram getFile tidak mengembalikan file_path.")
    response = requests.get(f"{TG_FILE_API}/{file_path}", timeout=HTTP_TIMEOUT)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Telegram document download: HTTP {response.status_code}: {response.text[:500]}"
        )
    if len(response.content) > 20 * 1024 * 1024:
        raise RuntimeError("File terlalu besar; batas /ganti adalah 20 MB.")
    return response.content, Path(file_path).name


def _atomic_write_source(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
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
            except OSError:
                log.exception("Gagal membersihkan temporary source file.")


# -----------------------------------------------------------------------------
# Telegram transport (launcher only)
# -----------------------------------------------------------------------------
def tg_call(method: str, payload: dict | None = None, timeout: int = HTTP_TIMEOUT):
    response = requests.post(f"{TG_API}/{method}", json=payload or {}, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"Telegram {method}: HTTP {response.status_code}: {response.text[:600]}")
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram {method}: {body}")
    return body.get("result")


def tg_send(chat_id: int, message: str) -> None:
    message = str(message)
    for start in range(0, len(message), 3900):
        try:
            tg_call("sendMessage", {"chat_id": chat_id, "text": message[start:start + 3900]})
        except Exception:
            log.exception("Telegram sendMessage gagal.")
            return


def tg_send_document(chat_id: int, document_path: str, caption: str = "") -> None:
    """Transport helper for main.py exports; launcher does not store the document."""
    path = Path(str(document_path))
    if not path.is_file():
        raise FileNotFoundError(f"Dokumen tidak ditemukan: {path}")
    with path.open("rb") as source:
        response = requests.post(
            f"{TG_API}/sendDocument",
            data={"chat_id": str(chat_id), "caption": str(caption or "")[:1024]},
            files={"document": (path.name, source)},
            timeout=HTTP_TIMEOUT,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Telegram sendDocument: HTTP {response.status_code}: {response.text[:600]}")
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram sendDocument: {body}")


# -----------------------------------------------------------------------------
# Render health endpoint
# -----------------------------------------------------------------------------
app = Flask(__name__)


@app.get("/")
def index():
    return jsonify({
        "ok": True,
        "service": "volumatic-launcher",
        "main_running": MAIN_RUNNING,
        "main_file": MAIN_FILE,
    })


@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "service": "volumatic-launcher",
        "main_running": MAIN_RUNNING,
        "timestamp": time.time(),
    })


def run_http_server() -> None:
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False, threaded=True)


# -----------------------------------------------------------------------------
# Load main.py without strategy-specific validation.
# -----------------------------------------------------------------------------
def _clear_local_imports() -> None:
    """Forget previously imported project modules so /try loads fresh files."""
    root = BASE_DIR.resolve()
    for name, module in list(sys.modules.items()):
        if name in {"__main__", "builtins"} or module is None:
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            Path(filename).resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def load_engine_modules() -> ModuleType:
    """Load main.py as-is. Its own imports decide which strategy/modules it needs."""
    main_path = safe_runtime_path(MAIN_FILE)
    if not main_path.is_file():
        raise FileNotFoundError(f"{MAIN_FILE} tidak ditemukan di {BASE_DIR}.")
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    _clear_local_imports()
    module_name = "trading_main_runtime"
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak bisa memuat {MAIN_FILE}.")
    main_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = main_module
    try:
        spec.loader.exec_module(main_module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    for name in ("on_start", "handle_update", "on_stop"):
        if not callable(getattr(main_module, name, None)):
            raise RuntimeError(f"main.py tidak menyediakan fungsi {name}().")
    return main_module


async def _call_main(module: ModuleType, name: str, context: dict) -> object:
    callback = getattr(module, name, None)
    if not callable(callback):
        return None
    result = callback(context)
    if asyncio.iscoroutine(result):
        return await result
    return result


def _main_context(chat_id: int) -> dict:
    return {
        "chat_id": chat_id,
        "user_id": ALLOWED_USER_ID,
        "send_message": tg_send,
        "send_document": tg_send_document,
        "launcher": "volumatic-launcher",
        "main_file": str(safe_runtime_path(MAIN_FILE)),
        "is_running": lambda: MAIN_RUNNING,
    }


async def start_or_reset_main(chat_id: int) -> None:
    "/try: stop an old local engine, then execute the current main.py."
    global MAIN_MODULE, MAIN_RUNNING, GANTI_ACTIVE
    async with MAIN_LOCK:
        old_module = MAIN_MODULE
        if old_module is not None and MAIN_RUNNING:
            MAIN_RUNNING = False
            try:
                await _call_main(old_module, "on_stop", _main_context(chat_id))
            except Exception:
                log.exception("[LAUNCHER] on_stop main lama gagal saat /try reset.")

        MAIN_MODULE = None
        sys.modules.pop("trading_main_runtime", None)
        module = load_engine_modules()
        MAIN_MODULE = module
        MAIN_RUNNING = True
        GANTI_ACTIVE = False
        try:
            result = await _call_main(module, "on_start", _main_context(chat_id))
            if result is False:
                raise RuntimeError("main.py on_start() mengembalikan False.")
        except Exception:
            MAIN_RUNNING = False
            MAIN_MODULE = None
            sys.modules.pop("trading_main_runtime", None)
            raise


async def stop_main(chat_id: int) -> bool:
    "/end: stop main.py only; keep Telegram polling and Render alive."
    global MAIN_MODULE, MAIN_RUNNING, GANTI_ACTIVE
    async with MAIN_LOCK:
        module = MAIN_MODULE
        was_running = bool(module is not None and MAIN_RUNNING)
        MAIN_RUNNING = False
        MAIN_MODULE = None
        GANTI_ACTIVE = False
        try:
            if module is not None:
                await _call_main(module, "on_stop", _main_context(chat_id))
        finally:
            sys.modules.pop("trading_main_runtime", None)
        return was_running


# -----------------------------------------------------------------------------
# /GANTI - upload any document(s), with no source/strategy assumptions
# -----------------------------------------------------------------------------
def github_upload_any(filename: str, content: bytes) -> str:
    if not GITHUB_TOKEN or not REPO_NAME or "/" not in REPO_NAME:
        raise RuntimeError("Atur GITHUB_TOKEN dan REPO_NAME agar /ganti dapat mengunggah file ke GitHub.")
    return github_put_source(filename, content, f"launcher: upload {filename}")


async def start_ganti(chat_id: int) -> None:
    global GANTI_ACTIVE
    GANTI_ACTIVE = True
    tg_send(
        chat_id,
        "⌁ MODE /GANTI AKTIF\n\n"
        "Kirim file apa saja sebagai Document Telegram.\n"
        "Nama file dipertahankan; launcher tidak memeriksa isi, ekstensi, strategi, atau timeframe.\n"
        "Setiap file akan disimpan di folder aplikasi dan diunggah ke root GitHub.\n\n"
        "Kamu boleh mengirim beberapa file berturut-turut.\n"
        "/try untuk memuat main.py terbaru  |  /back untuk membatalkan.",
    )


async def handle_ganti_document(chat_id: int, document: dict) -> None:
    filename = Path(str(document.get("file_name") or "uploaded_file")).name.strip()
    if not filename or filename in {".", ".."}:
        tg_send(chat_id, "✕ Nama file tidak valid.")
        return
    file_id = str(document.get("file_id") or "").strip()
    if not file_id:
        tg_send(chat_id, "✕ Telegram document tidak memiliki file_id.")
        return
    try:
        tg_send(chat_id, f"⟳ /GANTI menerima {filename}; mengunduh dan mengunggah ke GitHub...")
        content, _ = await asyncio.to_thread(download_telegram_document, file_id)
        # GitHub is updated first. The local copy changes only after successful upload.
        sha = await asyncio.to_thread(github_upload_any, filename, content)
        _atomic_write_source(safe_runtime_path(filename), content)
        tg_send(
            chat_id,
            "✓ FILE BERHASIL DIUNGGAH\n\n"
            f"FILE   : {filename}\n"
            f"UKURAN : {len(content):,} bytes\n"
            f"REPO   : {REPO_NAME} / {GITHUB_BRANCH}\n"
            f"COMMIT : {(sha[:12] + '…') if sha else 'updated'}\n\n"
            "File lokal dan GitHub sudah diperbarui. Jika file tersebut dipakai main.py, jalankan /try untuk reload engine.\n"
            "/ganti tetap aktif; kirim file berikutnya atau /back.",
        )
    except Exception as exc:
        log.exception("[LAUNCHER] /ganti gagal untuk %s", filename)
        tg_send(chat_id, f"✕ /GANTI GAGAL\nFILE : {filename}\nERROR: {str(exc)[:2200]}")


# -----------------------------------------------------------------------------
# Telegram command router: /try, /end, /ganti, /back, /healthz.
# -----------------------------------------------------------------------------
def is_authorized(message: dict) -> bool:
    chat_id = int((message.get("chat") or {}).get("id") or 0)
    user_id = int((message.get("from") or {}).get("id") or 0)
    return chat_id == ALLOWED_USER_ID and user_id == ALLOWED_USER_ID


def message_text(message: dict) -> str:
    return str(message.get("text") or message.get("caption") or "").strip()


def command_parts(text: str) -> tuple[str, str]:
    parts = str(text or "").strip().split(maxsplit=1)
    if not parts:
        return "", ""
    command = parts[0].split("@", 1)[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    return command, argument


async def route_message(message: dict, update: dict) -> None:
    global GANTI_ACTIVE
    if not is_authorized(message):
        return
    chat_id = int((message.get("chat") or {}).get("id") or 0)
    text = message_text(message)
    command, _argument = command_parts(text)
    document = message.get("document")

    if command == "/back":
        GANTI_ACTIVE = False
        tg_send(chat_id, "✓ Mode /ganti dibatalkan. Launcher dan engine tidak berubah.")
        return

    if command == "/try":
        tg_send(chat_id, "⟳ /TRY — menjalankan ulang main.py...")
        try:
            await start_or_reset_main(chat_id)
            tg_send(
                chat_id,
                "✓ MAIN.PY ONLINE\n\n"
                "Telegram polling tetap aktif dan semua update berikutnya diteruskan ke main.py.handle_update().\n"
                "/end untuk menghentikan main.py saja.",
            )
        except Exception as exc:
            log.exception("[LAUNCHER] /try gagal.")
            tg_send(chat_id, f"✕ /try gagal\n{str(exc)[:2500]}\n\nLauncher dan polling Telegram tetap online.")
        return

    if command == "/end":
        try:
            stopped = await stop_main(chat_id)
            if stopped:
                tg_send(chat_id, "■ main.py dihentikan melalui on_stop().\nLauncher, polling Telegram, dan Render tetap ONLINE.")
            else:
                tg_send(chat_id, "i main.py memang sedang STOPPED.\nLauncher, polling Telegram, dan Render tetap ONLINE.")
        except Exception as exc:
            log.exception("[LAUNCHER] /end gagal.")
            tg_send(chat_id, f"✕ /end gagal\n{str(exc)[:2000]}")
        return

    if command == "/ganti":
        await start_ganti(chat_id)
        return

    if document is not None and GANTI_ACTIVE:
        await handle_ganti_document(chat_id, document)
        return

    if command == "/healthz":
        tg_send(chat_id, f"LAUNCHER: ONLINE\nMAIN.PY: {'RUNNING' if MAIN_RUNNING else 'STOPPED'}\nGANTI: {'ACTIVE' if GANTI_ACTIVE else 'IDLE'}")
        return

    # Any other command/update belongs to main.py. The launcher does not
    # maintain a whitelist, so newly added main.py commands work automatically.
    module = MAIN_MODULE
    if module is None or not MAIN_RUNNING:
        tg_send(chat_id, "i main.py belum aktif. Gunakan /try.")
        return
    try:
        result = module.handle_update(update, _main_context(chat_id))
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:
        log.exception("[LAUNCHER] forward update ke main.py gagal.")
        tg_send(chat_id, f"✕ main.py gagal memproses update: {str(exc)[:1800]}")


# -----------------------------------------------------------------------------
# Telegram polling + Render process lifetime
# -----------------------------------------------------------------------------
async def telegram_loop() -> None:
    offset: int | None = None
    backoff = 2
    try:
        tg_call("deleteWebhook", {"drop_pending_updates": False}, timeout=20)
    except Exception:
        log.exception("[LAUNCHER] deleteWebhook gagal; polling tetap dicoba.")

    tg_send(
        ALLOWED_USER_ID,
        "⌁ VOLUMATIC LAUNCHER ONLINE\n\n"
        "/try  — jalankan main.py\n"
        "/ganti — unggah file apa saja (bisa beberapa file)\n"
        "/end  — hentikan main.py saja\n"
        "/back — batalkan mode /ganti\n"
        "/healthz — status launcher dan main.py",
    )

    while not STOP_EVENT.is_set():
        try:
            payload = {"timeout": TG_POLL_TIMEOUT, "allowed_updates": ["message"]}
            if offset is not None:
                payload["offset"] = offset
            updates = await asyncio.to_thread(tg_call, "getUpdates", payload, TG_POLL_TIMEOUT + 10)
            backoff = 2
            for update in updates or []:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = update_id + 1
                message = update.get("message")
                if isinstance(message, dict):
                    await route_message(message, update)
        except Exception as exc:
            log.warning("[LAUNCHER] Telegram polling reconnect: %s", exc)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def async_main() -> None:
    threading.Thread(target=run_http_server, name="volumatic-health", daemon=True).start()
    try:
        await telegram_loop()
    finally:
        STOP_EVENT.set()
        try:
            await stop_main(ALLOWED_USER_ID)
        except Exception:
            log.exception("[LAUNCHER] cleanup engine saat process shutdown gagal.")


if __name__ == "__main__":
    asyncio.run(async_main())
