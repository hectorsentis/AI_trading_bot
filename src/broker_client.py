import argparse
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_ACCOUNT_REST_BASE_URL,
    BINANCE_EXECUTION_REST_BASE_URL,
    BINANCE_RECV_WINDOW_MS,
    BINANCE_REST_BASE_URL,
    BINANCE_USE_TESTNET,
    DRY_RUN,
    ENABLE_TRADING,
    HTTP_TIMEOUT_SECONDS,
)


class BinanceCredentialsError(RuntimeError):
    pass


class LiveTradingBlockedError(RuntimeError):
    pass


@dataclass
class BinanceClientConfig:
    client_role: str = "market_data"
    base_url: str = BINANCE_ACCOUNT_REST_BASE_URL
    api_key: str | None = BINANCE_API_KEY
    api_secret: str | None = BINANCE_API_SECRET
    timeout_seconds: int = HTTP_TIMEOUT_SECONDS
    recv_window_ms: int = BINANCE_RECV_WINDOW_MS
    dry_run: bool = DRY_RUN
    enable_trading: bool = ENABLE_TRADING


class BinanceSpotClient:
    """Small Binance Spot REST wrapper with hard dry-run safety defaults."""

    def __init__(self, config: BinanceClientConfig | None = None):
        self.config = config or BinanceClientConfig()
        self.session = requests.Session()
        if self.config.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.config.api_key})

    @classmethod
    def market_data_client(cls) -> "BinanceSpotClient":
        return cls(BinanceClientConfig(client_role="market_data", base_url=BINANCE_REST_BASE_URL, api_key=None, api_secret=None))

    @classmethod
    def account_read_client(cls) -> "BinanceSpotClient":
        return cls(BinanceClientConfig(client_role="account_read", base_url=BINANCE_ACCOUNT_REST_BASE_URL))

    @classmethod
    def simulated_execution_client(cls) -> "BinanceSpotClient":
        return cls(
            BinanceClientConfig(
                client_role="simulated_execution",
                base_url=BINANCE_EXECUTION_REST_BASE_URL,
                dry_run=True,
                enable_trading=False,
            )
        )

    def _url(self, endpoint: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{endpoint}"

    def _request(self, method: str, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        response = self.session.request(
            method=method.upper(),
            url=self._url(endpoint),
            params=params or {},
            timeout=self.config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Binance API error {response.status_code}: {response.text[:500]}")
        if not response.text:
            return {}
        return response.json()

    def _require_credentials(self) -> None:
        if not self.config.api_key or not self.config.api_secret:
            raise BinanceCredentialsError(
                "Binance credentials are missing. Set BINANCE_API_KEY and BINANCE_API_SECRET in environment or .env."
            )

    def _signed_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_credentials()
        payload = dict(params or {})
        payload.setdefault("recvWindow", self.config.recv_window_ms)
        payload["timestamp"] = int(time.time() * 1000)
        query = urlencode(payload, doseq=True)
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload["signature"] = signature
        return payload

    def ping(self) -> dict:
        return self._request("GET", "/api/v3/ping")

    def server_time(self) -> dict:
        return self._request("GET", "/api/v3/time")

    def healthcheck(self) -> dict:
        ping_payload = self.ping()
        time_payload = self.server_time()
        return {
            "ok": True,
            "ping": ping_payload,
            "server_time": time_payload.get("serverTime"),
            "base_url": self.config.base_url,
            "client_role": self.config.client_role,
            "use_testnet": BINANCE_USE_TESTNET,
            "dry_run": self.config.dry_run,
            "enable_trading": self.config.enable_trading,
            "has_credentials": bool(self.config.api_key and self.config.api_secret),
        }

    def exchange_info(self, symbol: str | None = None, symbols: list[str] | None = None) -> dict:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        if symbols:
            params["symbols"] = json.dumps([s.upper() for s in symbols])
        return self._request("GET", "/api/v3/exchangeInfo", params=params)

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": int(limit),
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        return self._request("GET", "/api/v3/klines", params=params)

    def recent_klines(self, symbol: str, interval: str, limit: int = 500) -> list:
        return self.klines(symbol=symbol, interval=interval, limit=limit)

    def ticker_price(self, symbol: str) -> dict:
        return self._request("GET", "/api/v3/ticker/price", params={"symbol": symbol.upper()})

    def account_info(self) -> dict:
        return self._request("GET", "/api/v3/account", params=self._signed_params())

    def balances(self, nonzero_only: bool = True) -> list[dict]:
        account = self.account_info()
        balances = account.get("balances", [])
        if not nonzero_only:
            return balances
        out = []
        for row in balances:
            free = float(row.get("free", 0.0))
            locked = float(row.get("locked", 0.0))
            if free or locked:
                out.append(row)
        return out

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/api/v3/openOrders", params=self._signed_params(params))

    def simulate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        reason: str = "dry_run_simulation",
    ) -> dict:
        return {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": float(quantity),
            "price": float(price) if price is not None else None,
            "status": "SIMULATED",
            "dry_run": True,
            "reason": reason,
            "transactTime": int(time.time() * 1000),
        }

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, **kwargs) -> dict:
        if self.config.client_role not in {"simulated_execution", "execution"}:
            raise LiveTradingBlockedError(
                f"Client role {self.config.client_role!r} is not an execution client."
            )
        if self.config.dry_run:
            return self.simulate_order(symbol, side, order_type, quantity, kwargs.get("price"))
        if not self.config.enable_trading:
            raise LiveTradingBlockedError("ENABLE_TRADING is False; live Binance orders are blocked.")
        raise LiveTradingBlockedError("Live order placement is intentionally disabled in this project step.")


def parse_args():
    parser = argparse.ArgumentParser(description="Safe Binance Spot REST client checks.")
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--exchange-info", action="store_true")
    parser.add_argument("--ticker", action="store_true")
    parser.add_argument("--recent-klines", type=int, default=0)
    parser.add_argument("--account-info", action="store_true")
    parser.add_argument("--open-orders", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    public_client = BinanceSpotClient.market_data_client()
    account_client = BinanceSpotClient.account_read_client()

    if args.healthcheck:
        print(json.dumps({"public_market_data": public_client.healthcheck(), "account_client": account_client.healthcheck()}, ensure_ascii=True, indent=2))
    if args.exchange_info:
        info = public_client.exchange_info(symbol=args.symbol)
        print(json.dumps({"symbol_count": len(info.get("symbols", [])), "symbols": info.get("symbols", [])[:1]}, indent=2))
    if args.ticker:
        print(json.dumps(public_client.ticker_price(args.symbol), ensure_ascii=True, indent=2))
    if args.recent_klines:
        rows = public_client.recent_klines(args.symbol, "1h", limit=args.recent_klines)
        print(json.dumps({"rows": len(rows), "first": rows[0] if rows else None, "last": rows[-1] if rows else None}, indent=2))
    if args.account_info:
        try:
            account = account_client.account_info()
            print(json.dumps({"can_read_account": True, "accountType": account.get("accountType")}, indent=2))
        except BinanceCredentialsError as exc:
            print(json.dumps({"can_read_account": False, "reason": str(exc)}, indent=2))
        except Exception as exc:
            print(json.dumps({"can_read_account": False, "reason": str(exc)}, indent=2))
    if args.open_orders:
        try:
            print(json.dumps(account_client.open_orders(symbol=args.symbol), ensure_ascii=True, indent=2))
        except Exception as exc:
            print(json.dumps({"can_read_open_orders": False, "reason": str(exc)}, indent=2))


if __name__ == "__main__":
    main()
