from flask import Flask
import os

app = Flask(__name__)

@app.get("/")
def home():
    return "Bot is running on Render!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
