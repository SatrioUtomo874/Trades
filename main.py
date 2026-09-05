"""
Binance client authentication fix patch.

Use this class inside main.py Binance layer.
It fixes signed Futures requests:
- Mainnet default
- X-MBX-APIKEY header
- HMAC SHA256 signature
- server time offset
- Futures balance fallback
"""

import time
import hmac
import hashlib
from urllib.parse import urlencode

import requests


class BinanceAuthClient:
    def __init__(self, api_key, api_secret, testnet=False):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()

        self.base_url = (
            "https://testnet.binancefuture.com"
            if testnet
            else "https://fapi.binance.com"
        )

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-MBX-APIKEY": self.api_key,
        })

        self.time_offset = 0
        self.sync_time()

    def sync_time(self):
        try:
            r = requests.get(f"{self.base_url}/fapi/v1/time", timeout=10)
            server_time = r.json()["serverTime"]
            self.time_offset = server_time - int(time.time() * 1000)
        except Exception:
            self.time_offset = 0

    def timestamp(self):
        return int(time.time() * 1000) + self.time_offset

    def sign(self, params):
        query = urlencode(params)
        signature = hmac.new(
            self.secret_bytes(),
            query.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature

    def secret_bytes(self):
        return self.api_secret.encode("utf-8")

    def signed_get(self, path, params=None):
        params = params or {}
        params["timestamp"] = self.timestamp()
        params["recvWindow"] = 60000

        params["signature"] = self.sign(params)

        response = self.session.get(
            self.base_url + path,
            params=params,
            timeout=15
        )

        data = response.json()

        if response.status_code != 200:
            raise Exception(
                f"Binance {data.get('code')}: {data.get('msg')}"
            )

        return data

    def get_balance_usdt(self):
        errors = []

        endpoints = [
            "/fapi/v3/account",
            "/fapi/v2/account",
        ]

        for endpoint in endpoints:
            try:
                data = self.signed_get(endpoint)

                assets = data.get("assets", [])
                for asset in assets:
                    if asset.get("asset") == "USDT":
                        return float(
                            asset.get(
                                "walletBalance",
                                asset.get("balance", 0)
                            )
                        )

            except Exception as e:
                errors.append(str(e))

        try:
            data = self.signed_get("/fapi/v2/balance")

            for item in data:
                if item.get("asset") == "USDT":
                    return float(item.get("balance", 0))

        except Exception as e:
            errors.append(str(e))

        raise Exception(
            "Balance check failed: " + " | ".join(errors)
        )
