from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from config import BinanceConfig


@dataclass
class PositionInfo:
    symbol: str
    position_amt: float
    entry_price: float


class BinanceWrapper:
    """
    使用 requests 实现的精简版 Binance 合约 REST 封装，避免依赖 python-binance。
    """

    def __init__(self, cfg: BinanceConfig):
        self.cfg = cfg
        self.base_url = (
            "https://testnet.binancefuture.com"
            if cfg.use_testnet
            else "https://fapi.binance.com"
        )
        # 缓存交易对的精度信息，避免重复请求
        self._symbol_info_cache: Dict[str, Dict[str, Any]] = {}

    # ---------- 基础签名与请求 ----------
    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.cfg.api_key} if self.cfg.api_key else {}

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params, True)
        signature = hmac.new(
            self.cfg.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False, max_retries: int = 3):
        """GET 请求，带重试机制和更长超时时间"""
        params = params or {}
        url = self.base_url + path
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if signed:
                    params = self._sign_params(params)
                resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)  # 递增等待时间：2秒、4秒、6秒
                    print(f"网络请求超时，{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"连接 Binance 失败（已重试 {max_retries} 次）: {e}")
            except requests.exceptions.HTTPError as e:
                # HTTP 错误（如 4xx/5xx），打印 Binance 返回的详细信息后抛出
                resp = getattr(e, "response", None)
                detail: Dict[str, Any] = {}
                try:
                    if resp is not None:
                        detail = resp.json()
                    else:
                        detail = {"error": "no response object"}
                except Exception:
                    if resp is not None:
                        detail = {"raw": resp.text}
                    else:
                        detail = {"error": "no response text"}
                status_code = resp.status_code if resp is not None else "N/A"
                code = detail.get("code")
                msg = detail.get("msg") or detail.get("message")
                extra = ""
                if code == -4045:
                    extra = "说明：该交易对的止损/止盈/追踪委托数量已达到 Binance 限制，请减少保护单或取消旧单。"
                elif code == -2019:
                    extra = "说明：合约账户可用保证金不足，请降低仓位或增加资金。"
                print(f"Binance API 错误: HTTP {status_code} - {detail} {extra}")
                raise
        raise last_error

    def _post(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True,
        max_retries: int = 3,
    ):
        """POST 请求，带重试机制和更长超时时间"""
        params = params or {}
        url = self.base_url + path
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if signed:
                    params = self._sign_params(params)
                resp = requests.post(url, headers=self._headers(), params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)  # 递增等待时间：2秒、4秒、6秒
                    print(f"网络请求超时，{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"连接 Binance 失败（已重试 {max_retries} 次）: {e}")
            except requests.exceptions.HTTPError as e:
                # HTTP 错误（如 4xx/5xx），打印 Binance 返回的详细信息后抛出
                resp = getattr(e, "response", None)
                detail: Dict[str, Any] = {}
                try:
                    if resp is not None:
                        detail = resp.json()
                    else:
                        detail = {"error": "no response object"}
                except Exception:
                    if resp is not None:
                        detail = {"raw": resp.text}
                    else:
                        detail = {"error": "no response text"}
                status_code = resp.status_code if resp is not None else "N/A"
                code = detail.get("code")
                msg = detail.get("msg") or detail.get("message")
                extra = ""
                if code == -4045:
                    extra = "说明：该交易对的止损/止盈/追踪委托数量已达到 Binance 限制，请减少保护单或取消旧单。"
                elif code == -2019:
                    extra = "说明：合约账户可用保证金不足，请降低仓位或增加资金。"
                print(f"Binance API 错误: HTTP {status_code} - {detail} {extra}")
                raise
        raise last_error

    # ---------- 业务封装 ----------
    def get_account_equity(self) -> float:
        """获取账户总权益（USDT 计价，使用 futures 账户信息）。"""
        data = self._get("/fapi/v2/account", signed=True)
        return float(data.get("totalWalletBalance", 0.0))

    def get_symbol_price(self, symbol: Optional[str] = None) -> float:
        symbol = symbol or self.cfg.symbol
        data = self._get("/fapi/v1/ticker/price", params={"symbol": symbol})
        return float(data["price"])

    def get_positions(self) -> List[PositionInfo]:
        data = self._get("/fapi/v2/account", signed=True)
        positions: List[PositionInfo] = []
        for p in data.get("positions", []):
            amt = float(p.get("positionAmt", 0.0))
            if amt == 0:
                continue
            positions.append(
                PositionInfo(
                    symbol=p.get("symbol", ""),
                    position_amt=amt,
                    entry_price=float(p.get("entryPrice") or 0.0),
                )
            )
        return positions

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        取消单个未完成委托（用于清理无用的止损/追踪单）。
        """
        params = {"symbol": symbol, "orderId": order_id}
        url = self.base_url + "/fapi/v1/order"
        try:
            params = self._sign_params(params)
            resp = requests.delete(url, headers=self._headers(), params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            resp = getattr(e, "response", None)
            detail: Dict[str, Any] = {}
            try:
                if resp is not None:
                    detail = resp.json()
                else:
                    detail = {"error": "no response object"}
            except Exception:
                if resp is not None:
                    detail = {"raw": resp.text}
                else:
                    detail = {"error": "no response text"}
            status_code = resp.status_code if resp is not None else "N/A"
            code = detail.get("code")
            extra = ""
            if code == -2011:
                extra = "说明：该订单可能已被成交或取消，无需再次取消。"
            print(f"Binance API 错误: HTTP {status_code} - {detail} {extra}")
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取当前未完成委托，可按交易对过滤。
        返回 Binance 原始字段，供上层检查是否已有止盈/止损/追踪单。
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        try:
            return self._get("/fapi/v1/openOrders", params=params, signed=True)
        except Exception:
            # 调用者负责记录日志，此处仅保持接口兼容
            raise

    def get_all_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        获取账户所有历史交易记录（成交记录）。
        由于币安API单次最多返回1000条，此方法会自动分页获取所有记录。
        
        Args:
            symbol: 交易对，如 'ETHUSDT'，None表示获取所有交易对
            start_time: 开始时间戳（毫秒），None表示不限制
            end_time: 结束时间戳（毫秒），None表示不限制
            limit: 每次请求的最大记录数（1-1000），默认1000
        
        Returns:
            所有历史交易记录列表，按时间倒序排列（最新的在前）
        """
        all_trades: List[Dict[str, Any]] = []
        params: Dict[str, Any] = {"limit": min(max(limit, 1), 1000)}
        
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        # 币安API返回的记录是按时间倒序的（最新的在前）
        # 我们需要分页获取，直到没有更多记录
        while True:
            try:
                trades = self._get("/fapi/v1/userTrades", params=params, signed=True)
                if not trades or len(trades) == 0:
                    break
                
                all_trades.extend(trades)
                
                # 如果返回的记录数少于limit，说明已经获取完所有记录
                if len(trades) < params["limit"]:
                    break
                
                # 使用最后一条记录的时间作为下一次请求的endTime（因为记录是倒序的）
                last_trade_time = trades[-1].get("time")
                if last_trade_time:
                    params["endTime"] = int(last_trade_time) - 1
                else:
                    break
                    
            except Exception as e:
                print(f"⚠️  获取历史交易记录失败: {e}")
                break
        
        return all_trades

    def get_total_exposure(self) -> float:
        """当前总名义仓位价值（USDT）。"""
        positions = self.get_positions()
        total = 0.0
        for pos in positions:
            price = self.get_symbol_price(pos.symbol)
            total += abs(pos.position_amt) * price
        return total

    def get_tradable_symbols(
        self,
        quote_asset: Optional[str] = None,
        max_symbols: Optional[int] = None,
    ) -> List[str]:
        """
        自动识别当前账户允许交易的 USDT 永续合约交易对。
        - 使用 /fapi/v1/exchangeInfo
        - 过滤 status == 'TRADING'
        - 过滤 quoteAsset 等于指定 quote_asset（默认配置里的 base_asset，例如 USDT）
        """
        quote_asset = quote_asset or self.cfg.base_asset
        info = self._get("/fapi/v1/exchangeInfo")
        symbols: List[str] = []
        for s in info.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            if s.get("quoteAsset") != quote_asset:
                continue
            if s.get("contractType") not in ("PERPETUAL", "CURRENT_QUARTER", "NEXT_QUARTER"):
                continue
            symbols.append(s["symbol"])

        if max_symbols is not None and max_symbols > 0:
            symbols = symbols[:max_symbols]
        return symbols

    def get_k_lines(
        self, symbol: Optional[str] = None, interval: Optional[str] = None, limit: int = 100
    ) -> List[List[Any]]:
        symbol = symbol or self.cfg.symbol
        interval = interval or self.cfg.interval
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return self._get("/fapi/v1/klines", params=params)

    def _interval_to_minutes(self, interval: str) -> float:
        """将 Binance interval 转换为分钟"""
        interval = interval.lower()
        if interval.endswith("m"):
            return float(interval[:-1])
        if interval.endswith("h"):
            return float(interval[:-1]) * 60
        if interval.endswith("d"):
            return float(interval[:-1]) * 60 * 24
        if interval.endswith("w"):
            return float(interval[:-1]) * 60 * 24 * 7
        if interval.endswith("mo"):
            return float(interval[:-2]) * 60 * 24 * 30
        # 默认按分钟处理
        try:
            return float(interval)
        except ValueError:
            return 60.0

    def _ema_series(self, values: List[float], period: int) -> List[float]:
        if not values or period <= 0 or len(values) < period:
            return []
        ema_values: List[float] = []
        ema = sum(values[:period]) / period
        ema_values.append(ema)
        multiplier = 2 / (period + 1)
        for price in values[period:]:
            ema = (price - ema) * multiplier + ema
            ema_values.append(ema)
        return ema_values

    def _calculate_ema(self, values: List[float], period: int) -> Optional[float]:
        ema_values = self._ema_series(values, period)
        return ema_values[-1] if ema_values else None

    def _calculate_macd(self, values: List[float]) -> Dict[str, Optional[float]]:
        fast_series = self._ema_series(values, 12)
        slow_series = self._ema_series(values, 26)
        if not fast_series or not slow_series:
            return {"macd": None, "signal": None, "hist": None}

        min_len = min(len(fast_series), len(slow_series))
        fast_aligned = fast_series[-min_len:]
        slow_aligned = slow_series[-min_len:]
        macd_series = [fast - slow for fast, slow in zip(fast_aligned, slow_aligned)]
        signal_series = self._ema_series(macd_series, 9)

        if not macd_series:
            return {"macd": None, "signal": None, "hist": None}

        macd_line = macd_series[-1]
        signal_line = signal_series[-1] if signal_series else None
        hist = macd_line - signal_line if signal_line is not None else None
        return {"macd": macd_line, "signal": signal_line, "hist": hist}

    def _calculate_rsi(self, values: List[float], period: int = 14) -> Optional[float]:
        if not values or len(values) <= period:
            return None
        gains: List[float] = []
        losses: List[float] = []
        for i in range(1, len(values)):
            change = values[i] - values[i - 1]
            gains.append(max(change, 0))
            losses.append(abs(min(change, 0)))
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        if avg_loss == 0:
            return 100.0
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                return 100.0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs)) if rs else 0.0

    def _fetch_interval_klines(self, symbol: str, interval: str, limit: int) -> List[List[Any]]:
        """
        获取指定周期的 K 线。
        Binance 不支持 10m 周期，这里用两个 5m K 线聚合模拟 10m，避免 1120 错误。
        """
        if interval != "10m":
            return self.get_k_lines(symbol=symbol, interval=interval, limit=limit)

        base_interval = "5m"
        factor = 2
        base_limit = min(limit * factor, 1000)
        base_klines = self.get_k_lines(symbol=symbol, interval=base_interval, limit=base_limit)
        aggregated: List[List[Any]] = []

        for i in range(0, len(base_klines), factor):
            chunk = base_klines[i : i + factor]
            if len(chunk) < factor:
                continue

            open_time = chunk[0][0]
            close_time = chunk[-1][6]
            open_price = chunk[0][1]
            close_price = chunk[-1][4]
            high_price = max(float(c[2]) for c in chunk)
            low_price = min(float(c[3]) for c in chunk)
            volume = sum(float(c[5]) for c in chunk)
            quote_volume = sum(float(c[7]) for c in chunk)
            trades = sum(int(c[8]) for c in chunk)
            taker_buy_base = sum(float(c[9]) for c in chunk)
            taker_buy_quote = sum(float(c[10]) for c in chunk)
            ignore_val = chunk[-1][11] if len(chunk[-1]) > 11 else "0"

            aggregated.append(
                [
                    open_time,
                    open_price,
                    f"{high_price:.8f}",
                    f"{low_price:.8f}",
                    close_price,
                    f"{volume:.8f}",
                    close_time,
                    f"{quote_volume:.8f}",
                    str(trades),
                    f"{taker_buy_base:.8f}",
                    f"{taker_buy_quote:.8f}",
                    ignore_val,
                ]
            )
        return aggregated

    def get_multi_interval_summary(self, symbol: str) -> Dict[str, Any]:
        """
        获取覆盖最近 30 天的多周期 K 线摘要，供策略参考。
        返回格式：
        {
            "1d": {...},
            "4h": {...},
            ...
        }
        """
        intervals = ["1d", "4h", "1h", "30m", "15m", "10m", "5m", "1m"]
        target_minutes = 30 * 24 * 60  # 30 天
        summary: Dict[str, Any] = {}

        for interval in intervals:
            minutes_per_candle = self._interval_to_minutes(interval)
            if minutes_per_candle <= 0:
                continue
            limit = int(math.ceil(target_minutes / minutes_per_candle))
            limit = min(max(limit, 50), 1000)  # 限制在 50-1000

            try:
                klines = self._fetch_interval_klines(symbol=symbol, interval=interval, limit=limit)
            except Exception:
                continue

            if not klines:
                continue

            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            latest_close = closes[-1]
            earliest_close = closes[0]
            change_pct = (
                (latest_close - earliest_close) / earliest_close * 100 if earliest_close else 0.0
            )

            high_value = max(highs)
            low_value = min(lows)
            volatility_pct = (
                (high_value - low_value) / earliest_close * 100 if earliest_close else 0.0
            )

            total_volume = sum(volumes)
            avg_volume = total_volume / len(volumes) if volumes else 0.0

            trend = "up" if change_pct > 1 else "down" if change_pct < -1 else "flat"

            macd_metrics = self._calculate_macd(closes)
            summary[interval] = {
                "latest_close": latest_close,
                "change_pct": change_pct,
                "high": high_value,
                "low": low_value,
                "volatility_pct": volatility_pct,
                "total_volume": total_volume,
                "avg_volume": avg_volume,
                "trend": trend,
                "samples": len(klines),
                "ema_12": self._calculate_ema(closes, 12),
                "ema_26": self._calculate_ema(closes, 26),
                "rsi_14": self._calculate_rsi(closes, 14),
                "macd": macd_metrics.get("macd"),
                "macd_signal": macd_metrics.get("signal"),
                "macd_hist": macd_metrics.get("hist"),
            }

        return summary

    def _get_symbol_precision(self, symbol: str) -> Dict[str, Any]:
        """获取交易对的精度信息（数量步长、最小数量、最小名义价值等）"""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        
        info = self._get("/fapi/v1/exchangeInfo")
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                filters = s.get("filters", [])
                step_size = None
                min_qty = None
                min_notional = None
                tick_size = None
                for f in filters:
                    if f.get("filterType") == "LOT_SIZE":
                        step_size = float(f.get("stepSize", "1.0"))
                        min_qty = float(f.get("minQty", "0.0"))
                    elif f.get("filterType") == "PRICE_FILTER":
                        tick_size = float(f.get("tickSize", "0.01"))
                    elif f.get("filterType") == "MIN_NOTIONAL":
                        min_notional = float(f.get("notional", "0.0"))
                
                result = {
                    "step_size": step_size or 0.001,  # 默认 0.001
                    "min_qty": min_qty or 0.001,      # 默认 0.001
                    "min_notional": min_notional or 5.0,  # 默认 5 USDT
                    "tick_size": tick_size or 0.01,   # 默认 0.01
                }
                self._symbol_info_cache[symbol] = result
                return result
        
        # 如果找不到，返回默认值
        default = {"step_size": 0.001, "min_qty": 0.001, "min_notional": 5.0}
        self._symbol_info_cache[symbol] = default
        default = {"step_size": 0.001, "min_qty": 0.001, "min_notional": 5.0, "tick_size": 0.01}
        self._symbol_info_cache[symbol] = default
        return default

    def _format_price(self, price: float, symbol: str) -> float:
        """根据价格精度（tick size）格式化价格，避免 Binance 拒绝"""
        info = self._get_symbol_precision(symbol)
        tick_size = info.get("tick_size", 0.01)
        if tick_size <= 0:
            tick_size = 0.01
        decimals = len(str(tick_size).rstrip("0").split(".")[-1]) if "." in str(tick_size) else 0
        steps = round(price / tick_size)
        formatted = steps * tick_size
        return round(formatted, decimals)

    def set_leverage(self, symbol: str, leverage: int = 10) -> Dict[str, Any]:
        """
        设置合约杠杆倍数，默认为 10x。
        注意：Binance 要求 1 <= leverage <= 125（不同合约上限不同），
        若请求非法会返回错误，此处直接抛出异常并由上层记录日志。
        """
        params = {
            "symbol": symbol,
            "leverage": leverage,
        }
        return self._post("/fapi/v1/leverage", params=params, signed=True)

    def _format_quantity(self, quantity: float, symbol: str) -> float:
        """根据交易对精度格式化数量（向下取整到最近的步长）"""
        info = self._get_symbol_precision(symbol)
        step_size = info["step_size"]
        min_qty = info["min_qty"]

        # 计算步数
        steps = int(quantity / step_size)
        # 格式化数量（向下取整到最近的步长）
        formatted_qty = steps * step_size

        # 确保不小于最小数量
        if formatted_qty < min_qty:
            formatted_qty = min_qty

        # 移除多余的尾随零，但保留足够的精度
        # 例如：如果 step_size = 0.001，保留 3 位小数
        decimals = len(str(step_size).rstrip("0").split(".")[-1]) if "." in str(step_size) else 0
        formatted_qty = round(formatted_qty, decimals)

        return formatted_qty

    def _format_quantity_ceil(self, quantity: float, symbol: str) -> float:
        """根据交易对精度格式化数量（向上取整到最近的步长），用于确保满足最小名义价值"""
        info = self._get_symbol_precision(symbol)
        step_size = info["step_size"]
        min_qty = info["min_qty"]

        # 计算步数（向上取整）
        steps = math.ceil(quantity / step_size)
        # 格式化数量（向上取整到最近的步长）
        formatted_qty = steps * step_size

        # 确保不小于最小数量
        if formatted_qty < min_qty:
            formatted_qty = min_qty

        # 移除多余的尾随零，但保留足够的精度
        decimals = len(str(step_size).rstrip("0").split(".")[-1]) if "." in str(step_size) else 0
        formatted_qty = round(formatted_qty, decimals)

        return formatted_qty

    def place_market_order(
        self, side: str, quantity: float, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        创建市价单（合约）：side: 'buy' or 'sell'
        自动处理数量精度，确保符合 Binance 要求（包括最小名义价值）
        """
        symbol = symbol or self.cfg.symbol
        side_upper = "BUY" if side.lower() == "buy" else "SELL"
        
        # 格式化数量，确保符合 Binance 精度要求
        formatted_qty = self._format_quantity(quantity, symbol)
        
        if formatted_qty <= 0:
            raise ValueError(f"格式化后的数量 {formatted_qty} 无效，原始数量: {quantity}")
        
        # 获取当前价格
        current_price = self.get_symbol_price(symbol)
        
        # 粗略保证金检查：名义价值不能大于当前合约账户资金（保护低资金账户）
        try:
            equity = self.get_account_equity()
            notional = formatted_qty * current_price
            if equity and notional > equity:
                raise ValueError(
                    f"名义价值 {notional:.4f} 超出当前合约账户资金 {equity:.4f}，已放弃市价单，避免保证金不足。"
                )
        except Exception:
            # 若获取失败，继续尝试下单，让上层捕获实际错误
            pass
        
        params = {
            "symbol": symbol,
            "side": side_upper,
            "type": "MARKET",
            "quantity": formatted_qty,
        }
        return self._post("/fapi/v1/order", params=params, signed=True)

    def place_limit_order(
        self,
        side: str,
        quantity: float,
        price: float,
        symbol: Optional[str] = None,
        time_in_force: str = "GTC",
    ) -> Dict[str, Any]:
        """
        创建限价单（合约）：side: 'buy' or 'sell'
        time_in_force: 'GTC' (Good Till Cancel) 或 'IOC' (Immediate Or Cancel) 或 'FOK' (Fill Or Kill)
        """
        symbol = symbol or self.cfg.symbol
        side_upper = "BUY" if side.lower() == "buy" else "SELL"
        
        # 格式化数量和价格
        formatted_qty = self._format_quantity(quantity, symbol)
        formatted_price = self._format_price(price, symbol)
        
        if formatted_qty <= 0:
            raise ValueError(f"格式化后的数量 {formatted_qty} 无效，原始数量: {quantity}")
        if formatted_price <= 0:
            raise ValueError(f"格式化后的价格 {formatted_price} 无效，原始价格: {price}")
        
        # 粗略保证金检查：名义价值不能大于当前合约账户资金（保护低资金账户）
        try:
            equity = self.get_account_equity()
            notional = formatted_qty * formatted_price
            if equity and notional > equity:
                raise ValueError(
                    f"名义价值 {notional:.4f} 超出当前合约账户资金 {equity:.4f}，已放弃限价单，避免保证金不足。"
                )
        except Exception:
            # 若获取失败，继续尝试下单，让上层捕获实际错误
            pass

        params = {
            "symbol": symbol,
            "side": side_upper,
            "type": "LIMIT",
            "quantity": formatted_qty,
            "price": formatted_price,
            "timeInForce": time_in_force,
        }
        return self._post("/fapi/v1/order", params=params, signed=True)

    def place_bracket_orders(
        self,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        在开仓后，为该仓位创建固定止损 / 固定止盈市价单（reduceOnly）:
        - 多单: 止损/止盈均使用 SELL
        - 空单: 止损/止盈均使用 BUY
        为了减少 Binance 止损单数量限制的压力，本函数会：
        - 每个方向仅创建 1 个 STOP_MARKET 固定止损
        - 每个方向仅创建 1 个 TAKE_PROFIT_MARKET 固定止盈
        具体去重逻辑由上层持仓守护线程负责（遇到旧的保护单会自动取消多余的，仅保留最新一组）。
        """
        symbol = symbol or self.cfg.symbol
        close_side = "SELL" if side.lower() == "buy" else "BUY"
        
        # 格式化数量，确保符合 Binance 精度要求
        formatted_qty = self._format_quantity(quantity, symbol)

        results: Dict[str, Any] = {}

        # 固定止损单（STOP_MARKET）
        if stop_loss > 0 and stop_loss != entry_price:
            sl_price = self._format_price(stop_loss, symbol)
            sl_params = {
                "symbol": symbol,
                "side": close_side,
                "type": "STOP_MARKET",
                "stopPrice": sl_price,
                "closePosition": "false",
                "reduceOnly": "true",
                "quantity": formatted_qty,
            }
            results["stop_loss"] = self._post("/fapi/v1/order", params=sl_params, signed=True)

        # 固定止盈单（TAKE_PROFIT_MARKET）
        if take_profit > 0 and take_profit != entry_price:
            tp_price = self._format_price(take_profit, symbol)
            tp_params = {
                "symbol": symbol,
                "side": close_side,
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": tp_price,
                "closePosition": "false",
                "reduceOnly": "true",
                "quantity": formatted_qty,
            }
            results["take_profit"] = self._post("/fapi/v1/order", params=tp_params, signed=True)

        return results

    def place_trailing_stop(
        self,
        side: str,
        quantity: float,
        callback_rate: float,
        activation_price: float,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        创建追踪止损订单 (TRAILING_STOP_MARKET)。
        side: 'buy' 或 'sell'
        callback_rate: 回调比例，例如 0.01 表示 1%
        activation_price: 触发追踪止损开始生效的价格
        """
        symbol = symbol or self.cfg.symbol
        side_upper = "BUY" if side.lower() == "buy" else "SELL"

        formatted_qty = self._format_quantity(quantity, symbol)
        if formatted_qty <= 0:
            raise ValueError(f"格式化后的数量 {formatted_qty} 无效，原始数量: {quantity}")

        if activation_price <= 0:
            raise ValueError("activation_price 必须大于 0")

        activation_price = self._format_price(activation_price, symbol)
        activation_price_str = f"{activation_price:.8f}".rstrip("0").rstrip(".")
        if not activation_price_str:
            activation_price_str = f"{activation_price:.8f}"

        callback_rate_pct = max(0.1, min(callback_rate * 100, 5.0))
        callback_rate_str = f"{callback_rate_pct:.4f}".rstrip("0").rstrip(".")
        if not callback_rate_str:
            callback_rate_str = f"{callback_rate_pct:.4f}"

        params = {
            "symbol": symbol,
            "side": side_upper,
            "type": "TRAILING_STOP_MARKET",
            "quantity": formatted_qty,
            # Binance 要求 callbackRate 为百分比数值（1 代表 1%）
            "callbackRate": callback_rate_str,
            "activationPrice": activation_price_str,
            "reduceOnly": "true",
        }
        return self._post("/fapi/v1/order", params=params, signed=True)

