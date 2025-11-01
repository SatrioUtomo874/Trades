from fastapi import FastAPI
import uvicorn
import threading
import os
import asyncio

from indicator import run_bot  # pastikan fungsi run_bot() di file mu

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Bot is running"}

# Jalankan bot Telegram di thread terpisah
def start_telegram_bot():
    asyncio.run(run_bot())  # pastikan run_bot async-safe

threading.Thread(target=start_telegram_bot).start()

# Jalankan web server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
