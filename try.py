from __future__ import annotations

import asyncio
import base64
import gc
import html
import importlib.util
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify


# ============================================================
# TRY.PY
# Launcher / Telegram Gateway
#
# Tugas try.py:
#   1. Menjalankan polling Telegram.
#   2. /try  -> sync versi terbaru dari GitHub lalu load main.py.
#   3. /ganti -> mengganti file di GitHub melalui Telegram.
#   4. /end  -> menghentikan main.py dan membersihkan runtime launcher.
#   5. Command lain -> diteruskan ke main.py.
#
# Tidak ada:
#   - logic trading
#   - Binance
#   - strategy
#   - PnL
#   - trade storage
#   - checkpoint
#   - persistent runtime state
#
# Catatan:
#   GitHub adalah sumber file aplikasi.
#   Saat /try dipanggil, branch GitHub terbaru disalin ke working
#   directory lalu main.py terbaru di-load.
#
#   /end tidak menulis state/runtime ke disk. Launcher tetap idle
#   agar /try dapat dipanggil lagi.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Environment tidak pernah diambil dari GitHub archive.
load_dotenv(BASE_DIR / "trades.env")
load_dotenv(BASE_DIR / ".env")

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()
MAIN_FILE = (os.getenv("MAIN_FILE") or "main.py").strip()

PORT = int(os.getenv("PORT", "10000"))
TG_POLL_TIMEOUT = max(5, int(os.getenv("TG_POLL_TIMEOUT", "30")))
TG_ERROR_BACKOFF_MAX = max(10, int(os.getenv("TG_ERROR_BACKOFF_MAX", "60")))
HTTP_TIMEOUT = max(10, int(os.getenv("HTTP_TIMEOUT", "30")))

try:
    ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
except ValueError as exc:
    raise RuntimeError("ALLOWED_USER_ID harus berupa integer.") from exc


if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN belum diset.")

if not ALLOWED_USER_ID:
    raise RuntimeError("ALLOWED_USER_ID belum diset atau bernilai 0.")

if not REPO_NAME:
    raise RuntimeError("REPO_NAME belum diset.")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN belum diset.")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("try.launcher")

app = Flask(__name__)

_MAIN_LOCK = asyncio.Lock()
_SYNC_LOCK = threading.RLock()

_MAIN_MODULE: ModuleType | None = None
_MAIN_RUNNING = False

_STOP = threading.Event()

TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# ============================================================
# HTTP / Render health
# ============================================================

@app.get("/")
def index():
    return jsonify(
        {
            "ok": True,
            "service": "SMCAutoTrade launcher",
            "main_running": _MAIN_RUNNING,
        }
    )


@app.get("/healthz")
def healthz():
    return jsonify(
        {
            "ok": True,
            "service": "SMCAutoTrade launcher",
            "telegram_polling": not _STOP.is_set(),
            "main_running": _MAIN_RUNNING,
            "timestamp": time.time(),
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


# ============================================================
# Telegram
# ============================================================

def tg_call(
    method: str,
    payload: dict | None = None,
    timeout: int = HTTP_TIMEOUT,
):
    try:
        response = requests.post(
            f"{TG_API}/{method}",
            json=payload or {},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Telegram {method}: network error: {exc}") from exc

    if response.status_code == 409:
        raise RuntimeError(
            f"Telegram {method}: HTTP 409 {response.text[:500]}"
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Telegram {method}: HTTP {response.status_code}: "
            f"{response.text[:800]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Telegram {method}: invalid JSON") from exc

    if not body.get("ok"):
        raise RuntimeError(f"Telegram {method}: {body}")

    return body.get("result")


def tg_send(chat_id: int, text: str) -> None:
    text = str(text)

    # Telegram message limit aman di bawah 4096 karakter.
    for start in range(0, len(text), 3900):
        chunk = text[start : start + 3900]

        try:
            tg_call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": chunk,
                },
            )
        except Exception:
            log.exception("Gagal kirim Telegram ke %s", chat_id)
            return


def tg_get_file_bytes(file_id: str) -> bytes:
    info = tg_call(
        "getFile",
        {"file_id": file_id},
        timeout=20,
    )

    file_path = str(info["file_path"])

    response = requests.get(
        f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}",
        timeout=60,
    )
    response.raise_for_status()

    return response.content


# ============================================================
# GitHub
# ============================================================

def github_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }


def github_file_sha(path: str) -> str | None:
    encoded = "/".join(
        quote(part, safe="")
        for part in path.split("/")
    )

    response = requests.get(
        f"https://api.github.com/repos/{REPO_NAME}/contents/{encoded}",
        headers=github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=HTTP_TIMEOUT,
    )

    if response.status_code == 404:
        return None

    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub GET {path}: HTTP {response.status_code} "
            f"{response.text[:600]}"
        )

    return response.json().get("sha")


def validate_github_path(path: str) -> str:
    path = str(path or "").strip().replace("\\", "/").lstrip("/")

    if not path or ".." in Path(path).parts:
        raise ValueError("Path GitHub tidak valid.")

    if not re.fullmatch(r"[A-Za-z0-9._/\-]+", path):
        raise ValueError(
            "Path GitHub hanya boleh berisi huruf, angka, titik, "
            "underscore, slash, dan tanda minus."
        )

    filename = Path(path).name

    # Jangan pernah izinkan secret environment diubah melalui Telegram.
    if (
        filename in {".env", "trades.env"}
        or filename.startswith(".env.")
    ):
        raise ValueError(
            "File environment/secret tidak boleh dipush lewat /ganti."
        )

    return path


def github_replace(path: str, content: bytes) -> str:
    path = validate_github_path(path)

    sha = github_file_sha(path)

    encoded = "/".join(
        quote(part, safe="")
        for part in path.split("/")
    )

    payload = {
        "message": f"Update {path} via Telegram /ganti",
        "content": base64.b64encode(content).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }

    if sha:
        payload["sha"] = sha

    response = requests.put(
        f"https://api.github.com/repos/{REPO_NAME}/contents/{encoded}",
        headers=github_headers(),
        json=payload,
        timeout=60,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub PUT {path}: HTTP {response.status_code} "
            f"{response.text[:800]}"
        )

    return str(
        (response.json().get("commit") or {}).get("sha") or ""
    )[:12]


def safe_extract(
    tar: tarfile.TarFile,
    destination: Path,
) -> list[Path]:
    """
    Extract hanya file biasa dari archive GitHub.
    .env dan folder .git tidak pernah disalin.
    """
    written: list[Path] = []
    root = destination.resolve()

    for member in tar.getmembers():
        if not member.isfile():
            continue

        name = member.name.replace("\\", "/")
        parts = name.split("/", 1)

        # GitHub tarball biasanya punya root directory:
        # <repo>-<sha>/path/to/file
        if len(parts) != 2:
            continue

        rel = Path(parts[1])

        if not rel.parts or ".." in rel.parts:
            raise RuntimeError(
                f"GitHub archive path tidak aman: {name}"
            )

        if rel.name in {".env", "trades.env"}:
            continue

        if rel.name.startswith(".env."):
            continue

        if any(part == ".git" for part in rel.parts):
            continue

        target = (destination / rel).resolve()

        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"GitHub archive mencoba keluar dari folder: {name}"
            ) from exc

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        source = tar.extractfile(member)

        if source is None:
            continue

        with source, target.open("wb") as output:
            shutil.copyfileobj(source, output)

        written.append(rel)

    return written


def sync_repository() -> tuple[int, int]:
    """
    Mengambil versi branch GitHub TERBARU ke working directory.

    Penting:
    - Tidak menghapus apa pun di GitHub.
    - Tidak pernah melakukan DELETE ke GitHub.
    - File di GitHub menjadi sumber versi yang dijalankan.
    """
    url = (
        f"https://api.github.com/repos/{REPO_NAME}/tarball/"
        f"{quote(GITHUB_BRANCH, safe='')}"
    )

    with _SYNC_LOCK:
        response = requests.get(
            url,
            headers=github_headers(),
            timeout=120,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"GitHub tarball gagal: HTTP {response.status_code} "
                f"{response.text[:800]}"
            )

        with tempfile.TemporaryDirectory(
            prefix="repo-sync-"
        ) as temp_dir:
            archive = Path(temp_dir) / "repo.tar.gz"
            stage = Path(temp_dir) / "stage"

            archive.write_bytes(response.content)
            stage.mkdir()

            with tarfile.open(archive, "r:gz") as tar:
                written = safe_extract(tar, stage)

            if not written:
                raise RuntimeError(
                    "GitHub repository kosong atau tidak berisi "
                    "file yang bisa disinkronkan."
                )

            changed = 0

            for rel in written:
                src = stage / rel
                dst = BASE_DIR / rel

                dst.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                try:
                    is_same = (
                        dst.exists()
                        and dst.read_bytes() == src.read_bytes()
                    )
                except OSError:
                    is_same = False

                if not is_same:
                    shutil.copy2(src, dst)
                    changed += 1

    log.info(
        "[SYNC] GitHub -> local | files=%s changed=%s",
        len(written),
        changed,
    )

    return len(written), changed


# ============================================================
# main.py lifecycle
# ============================================================

def load_main_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} tidak ditemukan setelah sync GitHub."
        )

    source = path.read_text(encoding="utf-8")

    # Validasi syntax sebelum module dijalankan.
    compile(source, str(path), "exec")

    module_name = (
        f"launcher_main_{int(time.time() * 1000)}"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Tidak bisa membuat module spec untuk {path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Dibutuhkan oleh beberapa kasus dataclass/type annotation.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


async def start_main(chat_id: int) -> str:
    global _MAIN_MODULE, _MAIN_RUNNING

    async with _MAIN_LOCK:
        if _MAIN_RUNNING and _MAIN_MODULE is not None:
            return (
                f"▶️ <b>{html.escape(MAIN_FILE)}</b> "
                "sudah berjalan."
            )

        # /try selalu mengambil versi GitHub terbaru dulu.
        total, changed = await asyncio.to_thread(
            sync_repository
        )

        # Pastikan tidak ada module main lama yang dipakai.
        old_module = _MAIN_MODULE

        if old_module is not None:
            old_name = getattr(
                old_module,
                "__name__",
                None,
            )

            _MAIN_MODULE = None

            if old_name:
                sys.modules.pop(
                    old_name,
                    None,
                )

            gc.collect()

        path = BASE_DIR / MAIN_FILE
        module = load_main_module(path)

        on_start = getattr(
            module,
            "on_start",
            None,
        )

        handle_update = getattr(
            module,
            "handle_update",
            None,
        )

        on_stop = getattr(
            module,
            "on_stop",
            None,
        )

        if not callable(on_start):
            raise RuntimeError(
                f"{MAIN_FILE} wajib memiliki async on_start(context)."
            )

        if not callable(handle_update):
            raise RuntimeError(
                f"{MAIN_FILE} wajib memiliki async handle_update(update, context)."
            )

        if not callable(on_stop):
            raise RuntimeError(
                f"{MAIN_FILE} wajib memiliki async on_stop(context)."
            )

        context = {
            "launcher": "try.py",
            "chat_id": chat_id,
            "user_id": ALLOWED_USER_ID,
            "start_file": str(path),
            "is_running": lambda: _MAIN_RUNNING,
            "send_message": tg_send,
        }

        result = on_start(dict(context))

        if asyncio.iscoroutine(result):
            result = await result

        if result is False:
            raise RuntimeError(
                "main.py menolak startup melalui on_start()."
            )

        _MAIN_MODULE = module
        _MAIN_RUNNING = True

        return (
            f"🟢 <b>{html.escape(MAIN_FILE)} aktif.</b>\n\n"
            f"GitHub sync: {total} file diperiksa, "
            f"{changed} file diperbarui.\n"
            f"Versi yang dijalankan: <code>{html.escape(MAIN_FILE)}</code>\n\n"
            "Semua command selain command launcher "
            "diteruskan ke main.py."
        )


async def stop_main() -> str:
    global _MAIN_MODULE, _MAIN_RUNNING

    async with _MAIN_LOCK:
        module = _MAIN_MODULE

        if module is None or not _MAIN_RUNNING:
            # Tetap bersihkan referensi bila ada sisa.
            _MAIN_MODULE = None
            _MAIN_RUNNING = False
            gc.collect()

            return (
                "ℹ️ <b>main.py</b> sedang tidak berjalan.\n"
                "Launcher tetap standby."
            )

        _MAIN_RUNNING = False

        try:
            on_stop = getattr(
                module,
                "on_stop",
                None,
            )

            if callable(on_stop):
                context = {
                    "launcher": "try.py",
                    "user_id": ALLOWED_USER_ID,
                    "is_running": lambda: _MAIN_RUNNING,
                    "send_message": tg_send,
                }

                result = on_stop(context)

                if asyncio.iscoroutine(result):
                    await result

        except Exception:
            log.exception(
                "[END] cleanup main.py gagal"
            )

        finally:
            old_name = getattr(
                module,
                "__name__",
                None,
            )

            _MAIN_MODULE = None

            if old_name:
                sys.modules.pop(
                    old_name,
                    None,
                )

            gc.collect()

        # Tidak menulis state/checkpoint apa pun.
        return (
            "⏹️ <b>main.py dihentikan.</b>\n\n"
            "Semua runtime main.py di-memory sudah dilepas.\n"
            "Tidak ada state/checkpoint main.py yang disimpan.\n\n"
            "Launcher tetap standby dan tidak menyimpan sesi trade."
        )


# ============================================================
# Forward command ke main.py
# ============================================================

async def forward_update(update: dict) -> None:
    module = _MAIN_MODULE

    message = update.get("message") or {}

    if module is None or not _MAIN_RUNNING:
        chat_id = int(
            (message.get("chat") or {}).get("id") or 0
        )

        if chat_id:
            tg_send(
                chat_id,
                "ℹ️ main.py belum berjalan.\n"
                "Gunakan /try terlebih dahulu.",
            )

        return

    handler = getattr(
        module,
        "handle_update",
        None,
    )

    if not callable(handler):
        return

    context = {
        "launcher": "try.py",
        "chat_id": (message.get("chat") or {}).get("id"),
        "user_id": (message.get("from") or {}).get("id"),
        "start_file": str(BASE_DIR / MAIN_FILE),
        "is_running": lambda: _MAIN_RUNNING,
        "send_message": tg_send,
    }

    result = handler(
        update,
        context,
    )

    if asyncio.iscoroutine(result):
        await result


# ============================================================
# /ganti
# ============================================================

async def handle_ganti(message: dict) -> str:
    """
    /ganti tetap ada.

    Cara pakai:
      Kirim file Telegram sebagai DOCUMENT
      caption:
        /ganti
        /ganti main.py
        /ganti strategy/strategy_01.py

    Jika tanpa path pada caption, nama file Telegram digunakan.

    File di GitHub DIUPDATE, bukan dihapus.
    """
    document = message.get("document")

    if not isinstance(document, dict):
        return (
            "📦 <b>/ganti</b>\n\n"
            "Kirim file sebagai <b>document</b> dengan caption:\n\n"
            "<code>/ganti</code>\n"
            "atau\n"
            "<code>/ganti folder/nama.py</code>\n\n"
            "File akan di-update ke GitHub."
        )

    caption = str(
        message.get("caption") or ""
    ).strip()

    parts = caption.split(
        maxsplit=1
    )

    requested_path = (
        parts[1].strip()
        if len(parts) == 2
        else str(
            document.get("file_name") or ""
        )
    )

    if not requested_path:
        return "❌ Nama/path file tidak ditemukan."

    try:
        path = validate_github_path(
            requested_path
        )

        file_id = str(
            document.get("file_id") or ""
        )

        if not file_id:
            raise ValueError(
                "file_id Telegram tidak ditemukan."
            )

        content = await asyncio.to_thread(
            tg_get_file_bytes,
            file_id,
        )

        commit = await asyncio.to_thread(
            github_replace,
            path,
            content,
        )

        return (
            "✅ <b>/ganti berhasil</b>\n\n"
            f"File: <code>{html.escape(str(document.get('file_name') or path))}</code>\n"
            f"GitHub: <code>{html.escape(path)}</code>\n"
            f"Branch: <code>{html.escape(GITHUB_BRANCH)}</code>\n"
            f"Commit: <code>{html.escape(commit or 'created')}</code>\n\n"
            "File di GitHub tetap tersimpan.\n"
            "Untuk menjalankan versi terbaru, gunakan "
            "<code>/end</code> lalu <code>/try</code>."
        )

    except Exception as exc:
        log.exception(
            "[GANTI] gagal"
        )

        return (
            "❌ <b>/ganti gagal</b>\n"
            f"<code>{html.escape(str(exc)[:800])}</code>"
        )


# ============================================================
# Router
# ============================================================

def authorized(message: dict) -> bool:
    chat_id = int(
        (message.get("chat") or {}).get("id") or 0
    )

    user_id = int(
        (message.get("from") or {}).get("id") or 0
    )

    return (
        chat_id == ALLOWED_USER_ID
        and user_id == ALLOWED_USER_ID
    )


async def route_message(
    message: dict,
    update: dict,
) -> None:
    if not authorized(message):
        return

    chat_id = int(
        (message.get("chat") or {}).get("id")
    )

    text = str(
        message.get("text") or ""
    ).strip()

    caption = str(
        message.get("caption") or ""
    ).strip()

    # /ganti memakai document Telegram.
    if (
        isinstance(message.get("document"), dict)
        and caption.lower().startswith("/ganti")
    ):
        tg_send(
            chat_id,
            await handle_ganti(message),
        )
        return

    command = (
        text.split(maxsplit=1)[0]
        .split("@", 1)[0]
        .lower()
        if text
        else ""
    )

    try:
        # /try = sync GitHub -> load main.py terbaru.
        if command == "/try":
            tg_send(
                chat_id,
                await start_main(chat_id),
            )
            return

        # /end = hentikan main + bersihkan runtime.
        if command == "/end":
            tg_send(
                chat_id,
                await stop_main(),
            )
            return

        # /ganti = update file ke GitHub.
        if command == "/ganti":
            tg_send(
                chat_id,
                await handle_ganti(message),
            )
            return

        # /healthz tetap menjadi command launcher.
        if command == "/healthz":
            if not _MAIN_RUNNING:
                tg_send(
                    chat_id,
                    "🩺 <b>LAUNCHER</b>\n\n"
                    "Launcher: 🟢 READY\n"
                    "main.py: ⚪ OFFLINE\n\n"
                    "Gunakan /try untuk menjalankan "
                    "versi terbaru dari GitHub.",
                )
            else:
                tg_send(
                    chat_id,
                    "🩺 <b>LAUNCHER</b>\n\n"
                    "Launcher: 🟢 READY\n"
                    "main.py: 🟢 ONLINE",
                )

            return

        # /help hanya ditangani launcher ketika main belum aktif.
        # Setelah main aktif, /help diteruskan ke main.py.
        if command in {"/help", "/start"} and not _MAIN_RUNNING:
            tg_send(
                chat_id,
                "🤖 <b>Launcher</b>\n\n"
                "/try — ambil versi terbaru dari GitHub lalu jalankan main.py\n"
                "/end — hentikan main.py dan bersihkan runtime\n"
                "/ganti — update/replace file di GitHub\n"
                "/healthz — cek launcher dan main.py\n"
                "/help — menu launcher\n\n"
                "Setelah /try, command lain diteruskan ke main.py.",
            )

            return

        # Semua command lain masuk ke main.py.
        await forward_update(update)

    except Exception as exc:
        log.exception(
            "[ROUTER] command %s gagal",
            command or "<empty>",
        )

        tg_send(
            chat_id,
            "❌ <b>Command gagal</b>\n"
            f"<code>{html.escape(str(exc)[:800])}</code>",
        )


# ============================================================
# Telegram polling
# ============================================================

async def telegram_loop() -> None:
    offset: int | None = None
    backoff = 2

    try:
        tg_call(
            "deleteWebhook",
            {"drop_pending_updates": False},
            timeout=20,
        )
    except Exception:
        log.exception(
            "Gagal deleteWebhook"
        )

    try:
        tg_send(
            ALLOWED_USER_ID,
            "🚀 <b>SMCAutoTrade Launcher SIAP</b>\n\n"
            "Status: 🟢 ONLINE\n"
            "main.py: ⚪ OFFLINE\n\n"
            "Gunakan /try untuk menjalankan "
            "versi terbaru dari GitHub.",
        )
    except Exception:
        log.exception(
            "Gagal kirim launcher welcome"
        )

    while not _STOP.is_set():
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
                update_id = update.get(
                    "update_id"
                )

                if isinstance(update_id, int):
                    offset = update_id + 1

                message = update.get(
                    "message"
                )

                if isinstance(message, dict):
                    await route_message(
                        message,
                        update,
                    )

        except Exception as exc:
            log.warning(
                "[TG POLLING] %s",
                exc,
            )

            await asyncio.sleep(backoff)
            backoff = min(
                backoff * 2,
                TG_ERROR_BACKOFF_MAX,
            )


# ============================================================
# Program
# ============================================================

async def async_main() -> None:
    threading.Thread(
        target=run_flask,
        name="render-http",
        daemon=True,
    ).start()

    try:
        await telegram_loop()

    finally:
        _STOP.set()

        # Saat proses host dimatikan, pastikan main juga dibersihkan.
        try:
            await stop_main()
        except Exception:
            log.exception(
                "Cleanup main.py saat shutdown gagal."
            )


if __name__ == "__main__":
    asyncio.run(async_main())
