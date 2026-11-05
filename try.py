from __future__ import annotations

"""Volumatic Trend - minimal Telegram launcher.

Responsibilities of this file are intentionally limited to:
- keep Telegram polling and Render health endpoints alive;
- /try: sync main + strategy source from GitHub, then load/start them;
- /end: stop main engine only; this launcher and Render service stay alive;
- /ganti: receive a Python source file from Telegram and update the matching
  main/strategy source file in GitHub and the local runtime;
- /back: cancel an active /ganti selection;
- forward every other Telegram update to main.handle_update().

This launcher DOES NOT own or persist Binance cooldowns, ban lists, trades,
orders, positions, settings, events, or strategy state. Those belong to main.py.
Its only mutable session state is the temporary /ganti source-upload selection.
strategy.py remains a pure signal/indicator module.
"""

import ast
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

# Latest descriptive filenames. They can still be overridden in Render env.
MAIN_FILE = (os.getenv("MAIN_FILE") or "main_volumatic_rate_guard_batch_telemetry_20261105.py").strip()
STRATEGY_FILE = (os.getenv("STRATEGY_FILE") or "strategy_volumatic_ema_filter_20261105.py").strip()

PORT = max(1, int(os.getenv("PORT", "10000") or "10000"))
TG_POLL_TIMEOUT = max(5, min(50, int(os.getenv("TG_POLL_TIMEOUT", "30") or "30")))
HTTP_TIMEOUT = max(10, int(os.getenv("HTTP_TIMEOUT", "30") or "30"))
GITHUB_TIMEOUT = max(10, int(os.getenv("GITHUB_REQUEST_SECONDS", "30") or "30"))

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN belum diset.")
if not ALLOWED_USER_ID:
    raise RuntimeError("ALLOWED_USER_ID belum diset atau tidak valid.")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN belum diset; /try membutuhkan sinkronisasi source read-only.")
if not REPO_NAME or "/" not in REPO_NAME:
    raise RuntimeError("REPO_NAME belum diset atau format owner/repo tidak valid.")


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
STRATEGY_MODULE: ModuleType | None = None
MAIN_RUNNING = False
STOP_EVENT = asyncio.Event()

# /ganti state is launcher UI state only; it never stores trading data.
GANTI_WAITING_SELECTION = False
GANTI_TARGET: str | None = None


# -----------------------------------------------------------------------------
# Path + source validation
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


def _top_level_functions(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def validate_python_source(content: bytes, relative_path: str) -> None:
    try:
        source = content.decode("utf-8")
        compile(source, relative_path, "exec")
        functions = _top_level_functions(source)
    except (UnicodeDecodeError, SyntaxError) as exc:
        where = f"line {getattr(exc, 'lineno', '?')}"
        raise RuntimeError(f"{relative_path} tidak valid ({where}): {exc}") from exc

    basename = Path(relative_path).name.lower()
    if basename == Path(MAIN_FILE).name.lower():
        required = {"on_start", "handle_update", "on_stop"}
        missing = sorted(required - functions)
        if missing:
            raise RuntimeError(f"{relative_path} tidak memiliki kontrak launcher: {', '.join(missing)}")
        return

    if basename == Path(STRATEGY_FILE).name.lower():
        required_functions = {
            "signals_since", "detect_signals_since", "detect_latest_signal",
            "calculate_indicators", "analyze", "reference_risk", "trade_result_r",
        }
        missing = sorted(required_functions - functions)
        if missing:
            raise RuntimeError(f"{relative_path} tidak kompatibel: fungsi wajib {', '.join(missing)}")

        # Execute in a temporary registered module so dataclasses resolve __module__.
        validation_module = ModuleType("_volumatic_strategy_validation")
        validation_module.__file__ = relative_path
        sys.modules[validation_module.__name__] = validation_module
        try:
            exec(compile(source, relative_path, "exec"), validation_module.__dict__)
            required_values = {
                "TIMEFRAME_MS", "TIMEFRAME_INTERVALS", "STRATEGY_NAME", "STRATEGY_VERSION",
                "TREND_LENGTH", "ATR_LENGTH", "REFERENCE_BUFFER_ATR",
            }
            missing_values = sorted(name for name in required_values if name not in validation_module.__dict__)
            if missing_values:
                raise RuntimeError(f"{relative_path} tidak memiliki konstanta: {', '.join(missing_values)}")
            tf_ms = validation_module.TIMEFRAME_MS
            tf_intervals = validation_module.TIMEFRAME_INTERVALS
            if not isinstance(tf_ms, dict) or set(tf_ms) != {"H4", "D1"}:
                raise RuntimeError("strategy TIMEFRAME_MS harus memuat H4 dan D1.")
            if not isinstance(tf_intervals, dict) or set(tf_intervals) != {"H4", "D1"}:
                raise RuntimeError("strategy TIMEFRAME_INTERVALS harus memuat H4 dan D1.")
        finally:
            sys.modules.pop(validation_module.__name__, None)


def github_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_content_url(path: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path.strip("/").split("/"))
    return f"{GITHUB_API}/repos/{REPO_NAME}/contents/{encoded}"


def github_get_source(path: str) -> bytes:
    response = requests.get(
        github_content_url(path),
        headers=github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=GITHUB_TIMEOUT,
    )
    if response.status_code == 404:
        raise FileNotFoundError(f"{path} tidak ditemukan di {REPO_NAME}/{GITHUB_BRANCH}.")
    if response.status_code >= 400:
        raise RuntimeError(f"GitHub GET {path}: HTTP {response.status_code}: {response.text[:600]}")

    body = response.json()
    if body.get("type") != "file":
        raise RuntimeError(f"GitHub path bukan file: {path}")

    encoded = str(body.get("content") or "").replace("\n", "")
    if encoded:
        try:
            return base64.b64decode(encoded, validate=False)
        except Exception as exc:
            raise RuntimeError(f"Konten GitHub {path} gagal di-decode.") from exc

    # GitHub may omit `content` for large files; use its raw download URL.
    download_url = str(body.get("download_url") or "").strip()
    if not download_url:
        raise RuntimeError(f"GitHub tidak menyediakan konten atau download_url untuk {path}.")
    raw = requests.get(download_url, headers=github_headers(), timeout=GITHUB_TIMEOUT)
    if raw.status_code >= 400:
        raise RuntimeError(f"GitHub raw download {path}: HTTP {raw.status_code}")
    return raw.content


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


def sync_source_pair() -> list[str]:
    """Read-only GitHub sync. Only source-code files are written locally.

    No trade/cooldown/ban/settings/event data is read or written by this launcher.
    Both source files are downloaded and validated before either local file changes.
    """
    downloaded = {
        MAIN_FILE: github_get_source(MAIN_FILE),
        STRATEGY_FILE: github_get_source(STRATEGY_FILE),
    }
    for name, content in downloaded.items():
        validate_python_source(content, name)

    for name, content in downloaded.items():
        _atomic_write_source(safe_runtime_path(name), content)
    return list(downloaded)


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
        "strategy_file": STRATEGY_FILE,
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
# Load strategy first, then main. No business state is kept in launcher.
# -----------------------------------------------------------------------------
def load_engine_modules() -> ModuleType:
    global STRATEGY_MODULE
    strategy_path = safe_runtime_path(STRATEGY_FILE)
    main_path = safe_runtime_path(MAIN_FILE)
    if not strategy_path.is_file():
        raise FileNotFoundError(f"{STRATEGY_FILE} tidak ditemukan.")
    if not main_path.is_file():
        raise FileNotFoundError(f"{MAIN_FILE} tidak ditemukan.")

    strategy_source = strategy_path.read_text(encoding="utf-8")
    validate_python_source(strategy_source.encode("utf-8"), STRATEGY_FILE)
    sys.modules.pop("strategy", None)
    importlib.invalidate_caches()

    strategy_module = ModuleType("strategy")
    strategy_module.__file__ = str(strategy_path)
    strategy_module.__package__ = ""
    sys.modules["strategy"] = strategy_module
    try:
        exec(compile(strategy_source, str(strategy_path), "exec"), strategy_module.__dict__)
    except Exception:
        sys.modules.pop("strategy", None)
        raise

    if not callable(getattr(strategy_module, "signals_since", None)):
        raise RuntimeError("strategy.py tidak menyediakan signals_since().")

    main_source = main_path.read_text(encoding="utf-8")
    validate_python_source(main_source.encode("utf-8"), MAIN_FILE)
    module_name = "trading_main_runtime"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak bisa memuat {MAIN_FILE}.")
    main_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = main_module
    try:
        exec(compile(main_source, str(main_path), "exec"), main_module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        sys.modules.pop("strategy", None)
        raise

    if getattr(main_module, "volumatic", None) is not strategy_module:
        raise RuntimeError("main.py tidak memakai instance strategy.py yang baru dimuat.")
    for name in ("on_start", "handle_update", "on_stop"):
        if not callable(getattr(main_module, name, None)):
            raise RuntimeError(f"main.py tidak menyediakan {name}().")

    STRATEGY_MODULE = strategy_module
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
        "strategy_file": str(safe_runtime_path(STRATEGY_FILE)),
        "is_running": lambda: MAIN_RUNNING,
    }


async def start_or_reset_main(chat_id: int) -> None:
    """/try: sync source pair, stop current engine if any, then start fresh pair."""
    global MAIN_MODULE, MAIN_RUNNING, STRATEGY_MODULE
    async with MAIN_LOCK:
        # Sync/validate happens before stopping a running engine. GitHub failure
        # therefore leaves the current engine untouched.
        synced = await asyncio.to_thread(sync_source_pair)
        log.info("[LAUNCHER] source sync OK: %s", ", ".join(synced))

        old_module = MAIN_MODULE
        if old_module is not None and MAIN_RUNNING:
            MAIN_RUNNING = False
            try:
                await _call_main(old_module, "on_stop", _main_context(chat_id))
            except Exception:
                log.exception("[LAUNCHER] on_stop main lama gagal saat /try reset.")

        MAIN_MODULE = None
        STRATEGY_MODULE = None
        sys.modules.pop("trading_main_runtime", None)
        sys.modules.pop("strategy", None)

        module = load_engine_modules()
        MAIN_MODULE = module
        MAIN_RUNNING = True
        try:
            result = await _call_main(module, "on_start", _main_context(chat_id))
            if result is False:
                raise RuntimeError("main.py menolak startup melalui on_start().")
        except Exception:
            MAIN_RUNNING = False
            MAIN_MODULE = None
            STRATEGY_MODULE = None
            sys.modules.pop("trading_main_runtime", None)
            sys.modules.pop("strategy", None)
            raise

        log.info("[LAUNCHER] /try selesai; engine Volumatic aktif.")


async def stop_main(chat_id: int) -> bool:
    """/end: stop the trading engine only; keep Render + Telegram launcher alive."""
    global MAIN_MODULE, MAIN_RUNNING, STRATEGY_MODULE
    async with MAIN_LOCK:
        module = MAIN_MODULE
        was_running = bool(module is not None and MAIN_RUNNING)
        MAIN_RUNNING = False
        MAIN_MODULE = None
        try:
            if module is not None:
                await _call_main(module, "on_stop", _main_context(chat_id))
        finally:
            sys.modules.pop("trading_main_runtime", None)
            sys.modules.pop("strategy", None)
            STRATEGY_MODULE = None
        return was_running


# -----------------------------------------------------------------------------
# /GANTI - source deployment only (no trading/business persistence)
# -----------------------------------------------------------------------------
def _ganti_menu() -> str:
    return (
        "⌘ SOURCE UPDATE / GANTI\n\n"
        "Pilih file yang akan diganti:\n"
        f"1. {Path(MAIN_FILE).name}\n"
        f"2. {Path(STRATEGY_FILE).name}\n\n"
        "Kirim angka 1/2, lalu kirim file .py sebagai Document Telegram.\n"
        f"Atau langsung gunakan /ganti {Path(MAIN_FILE).name}\n"
        f"atau /ganti {Path(STRATEGY_FILE).name}\n\n"
        "Gunakan /back untuk membatalkan."
    )


def _resolve_ganti_target(value: str) -> str | None:
    answer = str(value or "").strip().lower()
    main_aliases = {"1", "main", "main.py", Path(MAIN_FILE).name.lower(), MAIN_FILE.lower()}
    strategy_aliases = {"2", "strategy", "strategy.py", Path(STRATEGY_FILE).name.lower(), STRATEGY_FILE.lower()}
    if answer in main_aliases:
        return MAIN_FILE
    if answer in strategy_aliases:
        return STRATEGY_FILE
    return None


def _infer_ganti_target(document: dict) -> str | None:
    filename = Path(str(document.get("file_name") or "").strip()).name.lower()
    if filename in {Path(MAIN_FILE).name.lower(), "main.py"}:
        return MAIN_FILE
    if filename in {Path(STRATEGY_FILE).name.lower(), "strategy.py"}:
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
            "⌁ TARGET SOURCE UPDATE\n\n"
            f"TARGET : {target}\n"
            "Kirim file Python (.py) sebagai Document Telegram.\n"
            "Setelah berhasil diunggah ke GitHub, jalankan /try untuk reload engine.\n"
            "Gunakan /back untuk membatalkan.",
        )
        return
    GANTI_WAITING_SELECTION = True
    GANTI_TARGET = None
    tg_send(chat_id, _ganti_menu())


async def handle_ganti_document(chat_id: int, document: dict) -> None:
    global GANTI_WAITING_SELECTION, GANTI_TARGET
    target = GANTI_TARGET
    if not target:
        tg_send(chat_id, "✕ Target /ganti belum dipilih. Gunakan /ganti terlebih dahulu.")
        return

    file_id = str(document.get("file_id") or "").strip()
    filename = Path(str(document.get("file_name") or "source.py")).name
    if not file_id:
        tg_send(chat_id, "✕ Telegram document tidak memiliki file_id.")
        return
    if not filename.lower().endswith(".py"):
        tg_send(chat_id, "✕ /ganti hanya menerima file Python (.py).")
        return

    tg_send(chat_id, f"⟳ SOURCE UPDATE\nMengunduh dan memvalidasi {filename} → {Path(target).name} ...")
    try:
        content, _telegram_filename = await asyncio.to_thread(download_telegram_document, file_id)
        validate_python_source(content, target)
        sha = await asyncio.to_thread(
            github_put_source,
            target,
            content,
            f"launcher: replace {Path(target).name}",
        )
        _atomic_write_source(safe_runtime_path(target), content)
        GANTI_WAITING_SELECTION = False
        GANTI_TARGET = None
        tg_send(
            chat_id,
            "✓ SOURCE UPDATED\n\n"
            f"TARGET : {target}\n"
            f"REPO   : {REPO_NAME}\n"
            f"BRANCH : {GITHUB_BRANCH}\n"
            f"COMMIT : {(sha[:12] + '…') if sha else 'created/updated'}\n\n"
            "GitHub dan runtime lokal sudah diperbarui. Engine yang sedang berjalan masih memakai modul lama; gunakan /try untuk reload pasangan main + strategy.",
        )
    except Exception as exc:
        log.exception("[LAUNCHER] /ganti gagal untuk %s", target)
        tg_send(chat_id, f"✕ /GANTI GAGAL\nTARGET : {target}\nERROR  : {str(exc)[:2200]}")


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
    global GANTI_WAITING_SELECTION, GANTI_TARGET
    if not is_authorized(message):
        return
    chat_id = int((message.get("chat") or {}).get("id") or 0)
    text = message_text(message)
    command, argument = command_parts(text)
    document = message.get("document")

    if command == "/back" and (GANTI_WAITING_SELECTION or GANTI_TARGET):
        GANTI_WAITING_SELECTION = False
        GANTI_TARGET = None
        tg_send(chat_id, "✓ Sesi /ganti dibatalkan. Launcher dan engine tidak berubah.")
        return

    if command == "/try":
        tg_send(chat_id, "⟳ VOLUMATIC LAUNCHER / RESET\nMemuat source main + strategy dan menginisialisasi engine...")
        try:
            await start_or_reset_main(chat_id)
            tg_send(
                chat_id,
                "✓ VOLUMATIC ENGINE ONLINE\n\n"
                f"MAIN     : {Path(MAIN_FILE).name}\n"
                f"STRATEGY : {Path(STRATEGY_FILE).name}\n"
                "LAUNCHER : tetap online\n"
                "Gunakan /end untuk menghentikan engine tanpa mematikan Render.",
            )
        except Exception as exc:
            log.exception("[LAUNCHER] /try gagal.")
            tg_send(chat_id, f"✕ /try gagal\n{str(exc)[:2500]}\n\nLauncher tetap online; source engine sebelumnya tidak disentuh bila sync gagal.")
        return

    if command == "/end":
        try:
            stopped = await stop_main(chat_id)
            if stopped:
                tg_send(chat_id, "■ VOLUMATIC ENGINE STOPPED\n\nMain engine dihentikan melalui on_stop().\nTelegram launcher dan Render tetap ONLINE.\nGunakan /try untuk menjalankan ulang.")
            else:
                tg_send(chat_id, "i VOLUMATIC ENGINE sudah STOPPED.\nTelegram launcher dan Render tetap ONLINE.\nGunakan /try untuk menjalankan ulang.")
        except Exception as exc:
            log.exception("[LAUNCHER] /end gagal.")
            tg_send(chat_id, f"✕ /end gagal\n{str(exc)[:2000]}")
        return

    # Source documents are launcher-owned only during an explicit /ganti session.
    # Otherwise non-launcher updates continue to main.py unchanged.
    if document is not None:
        if GANTI_TARGET:
            await handle_ganti_document(chat_id, document)
            return
        if command == "/ganti":
            inferred = _resolve_ganti_target(argument) if argument else _infer_ganti_target(document)
            if inferred:
                GANTI_WAITING_SELECTION = False
                GANTI_TARGET = inferred
                await handle_ganti_document(chat_id, document)
            else:
                GANTI_WAITING_SELECTION = True
                GANTI_TARGET = None
                tg_send(chat_id, "⌁ File diterima, tetapi target belum jelas.\n\n" + _ganti_menu())
            return
        if GANTI_WAITING_SELECTION:
            inferred = _infer_ganti_target(document)
            if inferred:
                GANTI_WAITING_SELECTION = False
                GANTI_TARGET = inferred
                await handle_ganti_document(chat_id, document)
            else:
                tg_send(
                    chat_id,
                    "✕ Nama file tidak cocok dengan target. Gunakan /ganti lalu pilih target, atau kirim file dengan nama target yang sesuai.",
                )
            return

    if command == "/ganti":
        await start_ganti(chat_id, argument)
        return

    if GANTI_WAITING_SELECTION and text and not command.startswith("/"):
        target = _resolve_ganti_target(text)
        if target:
            GANTI_WAITING_SELECTION = False
            GANTI_TARGET = target
            tg_send(
                chat_id,
                "⌁ TARGET SOURCE UPDATE\n\n"
                f"TARGET : {target}\n"
                "Sekarang kirim file .py sebagai Document Telegram.\n"
                "Gunakan /back untuk membatalkan.",
            )
        else:
            tg_send(chat_id, "✕ Pilihan tidak valid.\n\n" + _ganti_menu())
        return

    if command == "/healthz":
        tg_send(
            chat_id,
            "VOLUMATIC LAUNCHER STATUS\n\n"
            f"LAUNCHER : ONLINE\nENGINE   : {'RUNNING' if MAIN_RUNNING else 'STOPPED'}\n"
            f"MAIN     : {Path(MAIN_FILE).name}\nSTRATEGY : {Path(STRATEGY_FILE).name}",
        )
        return

    # No launcher-side trading commands or persistence. Everything else belongs
    # to main.py. If main is stopped, tell the user; /try remains available.
    module = MAIN_MODULE
    if module is None or not MAIN_RUNNING:
        tg_send(chat_id, "i Engine belum aktif. Gunakan /try untuk menjalankan main.py + strategy.py.")
        return

    try:
        result = module.handle_update(update, _main_context(chat_id))
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:
        log.exception("[LAUNCHER] forward update ke main.py gagal.")
        tg_send(chat_id, f"✕ Main engine gagal memproses update: {str(exc)[:1800]}")


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
        "/try  — start / reset main + strategy\n"
        "/end  — stop main engine; Render tetap hidup\n"
        "/ganti — update main.py / strategy.py via Telegram document\n"
        "/back — batalkan sesi /ganti\n"
        "/healthz — status launcher dan engine\n\n"
        "Launcher hanya mengelola source deployment. Trading state, cooldown, ban recovery, order, dan trade history dikelola main.py.",
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
