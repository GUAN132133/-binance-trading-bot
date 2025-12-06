"""
Binance 智能交易系统 - Web 界面
运行方式：py -3.15 web_app.py
然后在浏览器访问：http://localhost:5000
"""
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'binance-trading-bot-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量
trading_thread = None
trading_status = {
    "running": False,
    "start_time": None,
    "equity": 0.0,
    "drawdown": 0.0,
    "positions": [],
    "logs": [],              # 最近日志（最多 100 条）
    "last_update": None,
    "interval": None,        # 当前运行周期（如 '1m', '3m', '5m'）
    "strategy_mode": None,   # 当前策略模式（conservative/balanced/aggressive/scalping/sniper）
    "decisions": [],         # 最近的策略决策记录
    "strategy_feedback": None,  # DeepSeek 自学习反馈（最新）
    "strategy_feedback_history": [],  # 历史学习总结（持久化）
    "recent_market_states": [],  # 最近行情与执行记录
    "last_learning_count": 0,   # 上次自学习时处理的决策条数
    "trailing_stops": {},       # 记录已创建的移动止损订单
}

# 学习总结持久化文件路径
LEARNING_FEEDBACK_FILE = Path(__file__).parent / "learning_feedback.json"
# 知识库文件路径（存储DeepSeek积累的学习经验）
KNOWLEDGE_BASE_FILE = Path(__file__).parent / "knowledge_base.json"

# 记录持仓时间：{symbol: {"entry_time": datetime, "holding_time_minutes": int, "order_id": str}}
_position_holding_times = {}

TRAILING_SETTINGS = {
    "callback_rate": float(os.getenv("TRAILING_CALLBACK_RATE", "0.01")),          # 1%
    "activation_buffer_pct": float(os.getenv("TRAILING_ACTIVATION_BUFFER", "0.003")),  # 0.3%
}

PROTECTIVE_ORDER_TYPES = {"STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"}
MACRO_INTERVALS = ["1d", "4h", "1h"]
MICRO_INTERVALS = ["30m", "15m", "10m", "5m", "1m"]


def load_learning_feedback():
    """从文件加载历史学习总结"""
    global trading_status
    if LEARNING_FEEDBACK_FILE.exists():
        try:
            with open(LEARNING_FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    trading_status["strategy_feedback_history"] = data
                    # 设置最新的反馈
                    if data:
                        trading_status["strategy_feedback"] = data[-1]
                        trading_status["last_learning_count"] = data[-1].get("decision_count", 0)
                    log_message(f"✅ 已加载 {len(data)} 条历史学习总结")
                elif isinstance(data, dict):
                    # 兼容旧格式（单个反馈）
                    trading_status["strategy_feedback"] = data
                    trading_status["strategy_feedback_history"] = [data]
        except Exception as e:
            log_message(f"⚠️  加载历史学习总结失败: {e}")


def load_knowledge_base() -> Dict[str, Any]:
    """从文件加载知识库"""
    if KNOWLEDGE_BASE_FILE.exists():
        try:
            with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                knowledge = json.load(f)
                log_message(f"✅ 已加载知识库（包含 {len(knowledge.get('successful_patterns', []))} 条成功模式，{len(knowledge.get('failed_patterns', []))} 条失败模式）")
                return knowledge
        except Exception as e:
            log_message(f"⚠️  加载知识库失败: {e}")
    
    # 返回默认空知识库结构
    return {
        "successful_patterns": [],      # 成功交易模式
        "failed_patterns": [],         # 失败交易模式
        "best_practices": [],          # 最佳实践
        "market_conditions": [],        # 市场条件与策略匹配
        "timing_experience": [],       # 交易时机经验
        "stop_loss_take_profit_experience": [],  # 止损止盈经验
        "last_updated": None,          # 最后更新时间
    }


def save_knowledge_base(knowledge: Dict[str, Any]):
    """保存知识库到文件"""
    try:
        knowledge["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)
        log_message(f"✅ 知识库已保存（包含 {len(knowledge.get('successful_patterns', []))} 条成功模式，{len(knowledge.get('failed_patterns', []))} 条失败模式）")
    except Exception as e:
        log_message(f"⚠️  保存知识库失败: {e}")


def save_learning_feedback(feedback: dict):
    """保存学习总结到文件（追加到历史记录）"""
    global trading_status
    try:
        # 添加到历史记录
        trading_status["strategy_feedback_history"].append(feedback)
        # 只保留最近 100 条历史记录
        if len(trading_status["strategy_feedback_history"]) > 100:
            trading_status["strategy_feedback_history"] = trading_status["strategy_feedback_history"][-100:]
        
        # 保存到文件
        with open(LEARNING_FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(trading_status["strategy_feedback_history"], f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_message(f"⚠️  保存学习总结失败: {e}")


def _group_open_orders_by_symbol(binance) -> Dict[str, List[Dict[str, Any]]]:
    """获取当前所有未完成委托并按交易对分组"""
    try:
        orders = binance.get_open_orders()
    except Exception as e:
        log_message(f"⚠️  获取当前委托失败，无法检查止盈止损：{e}")
        return {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for order in orders or []:
        symbol = order.get("symbol")
        if not symbol:
            continue
        grouped.setdefault(symbol, []).append(order)
    return grouped


def _cleanup_orders_for_symbol(
    binance,
    symbol: str,
    position_side: str,
    symbol_orders: List[Dict[str, Any]],
    current_position_qty: Optional[float] = None,
) -> None:
    """
    对单个交易对的委托单做精简。
    已移除所有订单数量限制，不再自动取消订单。
    """
    # 此函数保留为空实现，不再执行任何订单清理操作
    pass


def _cleanup_protective_orders_for_symbol(
    binance,
    symbol: str,
    closing_side: str,
    symbol_orders: List[Dict[str, Any]],
    max_stop: int = 1,
    max_take_profit: int = 1,
    max_trailing: int = 1,
) -> None:
    """
    对单个交易对的保护类委托（止损 / 止盈 / 追踪止损）做精简（兼容旧逻辑）：
    - 每个方向的 STOP_MARKET（固定止损）最多保留 max_stop 个
    - 每个方向的 TAKE_PROFIT_MARKET（固定止盈）最多保留 max_take_profit 个
    - 每个方向的 TRAILING_STOP_MARKET（移动止盈止损）最多保留 max_trailing 个
    其余多余的旧单会被自动取消，避免堆积导致 Binance Reach max stop order limit。
    """
    closing_side = closing_side.upper()
    stop_orders: List[Dict[str, Any]] = []
    tp_orders: List[Dict[str, Any]] = []
    trailing_orders: List[Dict[str, Any]] = []

    for order in symbol_orders:
        if order.get("side", "").upper() != closing_side:
            continue
        o_type = order.get("type")
        if o_type == "STOP_MARKET":
            stop_orders.append(order)
        elif o_type == "TAKE_PROFIT_MARKET":
            tp_orders.append(order)
        elif o_type == "TRAILING_STOP_MARKET":
            trailing_orders.append(order)

    def _limit_and_cancel(orders: List[Dict[str, Any]], max_allowed: int, label: str):
        if max_allowed <= 0 or len(orders) <= max_allowed:
            return
        # 按 orderId 从小到大排序，保留最新的 max_allowed 个，其余依次取消
        sorted_orders = sorted(
            orders,
            key=lambda o: int(o.get("orderId", 0)),
        )
        to_cancel = sorted_orders[:-max_allowed]
        for o in to_cancel:
            order_id = o.get("orderId")
            if not order_id:
                continue
            try:
                binance.cancel_order(symbol=symbol, order_id=int(order_id))
                log_message(
                    f"{symbol}: 取消多余{label}保护单 {o.get('type')} #{order_id}，仅保留最新 {max_allowed} 个。"
                )
            except Exception as e:
                log_message(
                    f"{symbol}: 自动取消多余{label}保护单失败（{o.get('type')} #{order_id}）：{e}"
                )

    _limit_and_cancel(stop_orders, max_stop, "固定止损/止盈")
    _limit_and_cancel(tp_orders, max_take_profit, "固定止损/止盈")
    _limit_and_cancel(trailing_orders, max_trailing, "移动止盈/止损")


def _has_trailing_stop(symbol_orders: List[Dict[str, Any]], closing_side: str) -> bool:
    closing_side = closing_side.upper()
    for order in symbol_orders:
        if order.get("type") == "TRAILING_STOP_MARKET" and order.get("side", "").upper() == closing_side:
            return True
    return False


def _has_any_protective_order(symbol_orders: List[Dict[str, Any]], closing_side: str) -> bool:
    closing_side = closing_side.upper()
    for order in symbol_orders:
        if order.get("side", "").upper() != closing_side:
            continue
        if order.get("type") in PROTECTIVE_ORDER_TYPES:
            return True
    return False


def create_orders_from_deepseek_plan(
    binance,
    symbol: str,
    position_side: str,
    position_quantity: float,
    orders_config: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    根据DeepSeek返回的orders配置，创建委托单：
    - 1个移动全仓止损
    - 3个买入限价单（用于做T）
    - 3个卖出限价单（用于做T）
    - 1个移动分批止盈
    """
    results: Dict[str, Any] = {
        "trailing_stop_loss": None,
        "trailing_take_profit": None,
        "buy_orders": [],
        "sell_orders": [],
    }
    
    if not orders_config or position_quantity <= 0:
        return results
    
    position_side = position_side.lower()
    closing_side = "SELL" if position_side == "long" else "BUY"
    opening_side = "BUY" if position_side == "long" else "SELL"
    
    for order_config in orders_config:
        order_type = order_config.get("type", "")
        quantity_pct = float(order_config.get("quantity_pct", 0))
        
        if quantity_pct <= 0:
            continue
        
        order_quantity = position_quantity * quantity_pct
        formatted_qty = binance._format_quantity(order_quantity, symbol)
        
        if formatted_qty <= 0:
            continue
        
        try:
            if order_type == "trailing_stop_loss":
                # 移动全仓止损
                callback_rate = float(order_config.get("callback_rate", 0.01))
                activation_price = float(order_config.get("activation_price", 0))
                if activation_price <= 0:
                    current_price = binance.get_symbol_price(symbol)
                    if position_side == "long":
                        activation_price = current_price * 1.003  # 默认激活价
                    else:
                        activation_price = current_price * 0.997
                
                order = binance.place_trailing_stop(
                    side=closing_side.lower(),
                    quantity=formatted_qty,
                    callback_rate=callback_rate,
                    activation_price=activation_price,
                    symbol=symbol,
                )
                results["trailing_stop_loss"] = order
                log_message(
                    f"{symbol}: 已创建移动全仓止损（数量: {formatted_qty:.6f}, "
                    f"回调率: {callback_rate*100:.2f}%, 激活价: {activation_price:.4f}）"
                )
                
            elif order_type == "trailing_take_profit":
                # 移动分批止盈
                callback_rate = float(order_config.get("callback_rate", 0.01))
                activation_price = float(order_config.get("activation_price", 0))
                if activation_price <= 0:
                    current_price = binance.get_symbol_price(symbol)
                    if position_side == "long":
                        activation_price = current_price * 1.01  # 默认激活价
                    else:
                        activation_price = current_price * 0.99
                
                order = binance.place_trailing_stop(
                    side=closing_side.lower(),
                    quantity=formatted_qty,
                    callback_rate=callback_rate,
                    activation_price=activation_price,
                    symbol=symbol,
                )
                results["trailing_take_profit"] = order
                log_message(
                    f"{symbol}: 已创建移动分批止盈（数量: {formatted_qty:.6f}, "
                    f"回调率: {callback_rate*100:.2f}%, 激活价: {activation_price:.4f}）"
                )
                
            elif order_type == "limit_buy":
                # 买入限价单（用于做T）
                price = float(order_config.get("price", 0))
                if price <= 0:
                    continue
                
                formatted_price = binance._format_price(price, symbol)
                order = binance.place_limit_order(
                    side="buy",
                    quantity=formatted_qty,
                    price=formatted_price,
                    symbol=symbol,
                )
                results["buy_orders"].append(order)
                purpose = order_config.get("purpose", "")
                log_message(
                    f"{symbol}: 已创建买入限价单（价格: {formatted_price:.4f}, "
                    f"数量: {formatted_qty:.6f}, 目的: {purpose}）"
                )
                
            elif order_type == "limit_sell":
                # 卖出限价单（用于做T）
                price = float(order_config.get("price", 0))
                if price <= 0:
                    continue
                
                formatted_price = binance._format_price(price, symbol)
                order = binance.place_limit_order(
                    side="sell",
                    quantity=formatted_qty,
                    price=formatted_price,
                    symbol=symbol,
                )
                results["sell_orders"].append(order)
                purpose = order_config.get("purpose", "")
                log_message(
                    f"{symbol}: 已创建卖出限价单（价格: {formatted_price:.4f}, "
                    f"数量: {formatted_qty:.6f}, 目的: {purpose}）"
                )
                
        except Exception as e:
            log_message(f"{symbol}: 创建{order_type}订单失败：{e}")
    
    return results


def ensure_trailing_stop_for_position(
    binance,
    symbol: str,
    position_side: str,
    quantity: float,
    entry_price: float,
    source: str = "系统自动",
) -> bool:
    """
    为指定持仓创建移动止盈止损（Trailing Stop），确保所有仓位都有动态保护。
    position_side: 'long' / 'short'
    """
    if quantity <= 0 or entry_price <= 0:
        return False

    callback_rate = max(TRAILING_SETTINGS["callback_rate"], 0.001)
    buffer_pct = max(TRAILING_SETTINGS["activation_buffer_pct"], 0.0005)
    qty = abs(quantity)
    position_side = position_side.lower()
    close_side = "sell" if position_side == "long" else "buy"

    if position_side == "long":
        activation_price = entry_price * (1 + buffer_pct)
    else:
        activation_price = entry_price * (1 - buffer_pct)

    try:
        order = binance.place_trailing_stop(
            side=close_side,
            quantity=qty,
            callback_rate=callback_rate,
            activation_price=activation_price,
            symbol=symbol,
        )
        trading_status.setdefault("trailing_stops", {})[f"{symbol}:{position_side}"] = order.get("orderId")
        log_message(
            f"{symbol}: 已创建移动止盈止损（来源：{source}，激活价 {activation_price:.4f}，回调 {callback_rate*100:.2f}%）"
        )
        return True
    except Exception as e:
        log_message(f"{symbol}: 创建移动止盈止损失败（{source}）：{e}")
        return False


def record_market_state(decision_entry: dict, market_data: Dict[str, Any]):
    """记录最近的行情与执行情况，供 DeepSeek 实时学习"""
    state = {
        "timestamp": decision_entry.get("timestamp"),
        "symbol": decision_entry.get("symbol"),
        "strategy": decision_entry.get("strategy"),
        "interval": decision_entry.get("interval"),
        "action": decision_entry.get("action"),
        "status": decision_entry.get("status"),
        "equity": decision_entry.get("equity"),
    }
    if market_data:
        ticker = market_data.get("ticker") or {}
        orderbook = market_data.get("orderbook") or {}
        state.update(
            {
                "price": ticker.get("last"),
                "change_pct": ticker.get("change"),
                "volume": ticker.get("volume"),
                "orderbook_imbalance": orderbook.get("imbalance"),
                "bid_depth": orderbook.get("bid_depth"),
                "ask_depth": orderbook.get("ask_depth"),
            }
        )
    trading_status.setdefault("recent_market_states", []).append(state)
    trading_status["recent_market_states"] = trading_status["recent_market_states"][-30:]


def emergency_close_all_positions(binance) -> bool:
    """当触发最大回撤时，立即平掉所有合约持仓"""
    global _position_holding_times
    try:
        positions = binance.get_positions()
    except Exception as e:
        log_message(f"❌ 紧急平仓失败：无法获取当前持仓 - {e}")
        return False

    success = True
    for pos in positions:
        qty = abs(pos.position_amt)
        if qty <= 0:
            continue
        close_side = "sell" if pos.position_amt > 0 else "buy"
        try:
            binance.place_market_order(side=close_side, quantity=qty, symbol=pos.symbol)
            log_message(f"{pos.symbol}: 已执行紧急平仓（方向 {close_side.upper()}，数量 {qty}）")
        except Exception as e:
            log_message(f"{pos.symbol}: 紧急平仓失败：{e}")
            success = False

    if success:
        _position_holding_times.clear()
        trading_status.setdefault("trailing_stops", {}).clear()
    return success


def _aggregate_trend_layers(summary: Dict[str, Any], intervals: List[str]) -> Dict[str, Any]:
    stats = []
    for interval in intervals:
        data = summary.get(interval)
        if isinstance(data, dict) and data:
            stats.append((interval, data))
    if not stats:
        return {}

    votes = {"up": 0, "down": 0, "flat": 0}
    change_pcts: List[float] = []
    rsi_values: List[float] = []
    volatility: List[float] = []

    for interval, data in stats:
        trend = data.get("trend") or "flat"
        votes[trend if trend in votes else "flat"] += 1
        change_pcts.append(float(data.get("change_pct") or 0.0))
        volatility.append(abs(float(data.get("volatility_pct") or 0.0)))
        rsi = data.get("rsi_14")
        if rsi is not None:
            rsi_values.append(float(rsi))

    dominant_trend = max(votes, key=votes.get)
    avg_change_pct = sum(change_pcts) / len(change_pcts) if change_pcts else 0.0
    avg_volatility = sum(volatility) / len(volatility) if volatility else 0.0
    avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else None

    summary_text = (
        f"{'/'.join([i for i, _ in stats])} 主导趋势 {dominant_trend}，"
        f"平均涨跌 {avg_change_pct:.2f}%，平均波动 {avg_volatility:.2f}%"
    )
    if avg_rsi is not None:
        summary_text += f"，平均 RSI14 {avg_rsi:.2f}"

    return {
        "intervals": [i for i, _ in stats],
        "dominant_trend": dominant_trend,
        "trend_votes": votes,
        "avg_change_pct": avg_change_pct,
        "avg_volatility_pct": avg_volatility,
        "avg_rsi": avg_rsi,
        "summary": summary_text,
    }


def analyze_trend_layers(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not summary:
        return {}

    macro = _aggregate_trend_layers(summary, MACRO_INTERVALS)
    micro = _aggregate_trend_layers(summary, MICRO_INTERVALS)

    def trend_to_bias(trend: str) -> str:
        if trend == "up":
            return "long"
        if trend == "down":
            return "short"
        return "range"

    macro_bias = trend_to_bias(macro.get("dominant_trend")) if macro else "range"
    micro_bias = trend_to_bias(micro.get("dominant_trend")) if micro else "range"

    guidance = ""
    if macro and micro:
        if macro_bias == "long" and micro_bias in {"short", "range"}:
            guidance = "宏观多头，小周期回调：逢低加仓，多次低买高卖锁定利润。"
        elif macro_bias == "short" and micro_bias in {"long", "range"}:
            guidance = "宏观空头，小周期反弹：高位做空或逐步减仓，等待再次低买回补。"
        elif macro_bias == micro_bias and macro_bias in {"long", "short"}:
            guidance = "宏观与小周期同向：顺势分批加仓/减仓，扩大涨势收益。"
        else:
            guidance = "宏观震荡：轻仓区间交易或观望，等待突破。"

    return {
        "macro": macro,
        "micro": micro,
        "macro_bias": macro_bias,
        "micro_bias": micro_bias,
        "guidance": guidance,
    }


def log_message(message):
    """记录日志并发送到 WebSocket"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    trading_status["logs"].append(log_entry)
    # 只保留最近 100 条日志
    if len(trading_status["logs"]) > 100:
        trading_status["logs"] = trading_status["logs"][-100:]
    trading_status["last_update"] = timestamp
    socketio.emit('log', {'message': log_entry})
    print(log_entry)


def append_decision(entry: dict):
    """记录策略决策并通知前端"""
    trading_status.setdefault("decisions", []).append(entry)
    # 只保留最近 50 条决策
    trading_status["decisions"] = trading_status["decisions"][-50:]
    socketio.emit('decision', entry)


def position_guard_loop(binance, risk_cfg):
    """
    持仓守护线程：
    - 确保每个持仓在建仓后都存在移动止盈止损（TRAILING_STOP_MARKET）
    - 定期巡检未完成委托，缺失时自动补齐并给出现有仓位无保护的日志原因
    - 检查持仓时间，到期自动平仓
    """
    global _position_holding_times
    log_message("持仓守护线程已启动（移动止盈 / 自适应止损 / 持仓时间管理）")

    while trading_status["running"]:
        try:
            positions = binance.get_positions()
            open_orders_by_symbol = _group_open_orders_by_symbol(binance)
            current_time = datetime.now()

            # 清理无持仓但仍存在的止损/追踪委托，避免堆积占用名额
            position_symbols = {p.symbol for p in positions}
            for symbol, orders in list(open_orders_by_symbol.items()):
                if symbol in position_symbols:
                    continue
                for order in orders:
                    if order.get("type") in PROTECTIVE_ORDER_TYPES:
                        order_id = order.get("orderId")
                        if not order_id:
                            continue
                        try:
                            binance.cancel_order(symbol=symbol, order_id=int(order_id))
                            log_message(f"{symbol}: 检测到无持仓但仍存在保护单（{order.get('type')} #{order_id}），已自动取消。")
                        except Exception as e:
                            log_message(f"{symbol}: 自动取消无用保护单失败（{order.get('type')} #{order_id}）：{e}")

            # 检查持仓时间，到期自动平仓
            for symbol, holding_info in list(_position_holding_times.items()):
                entry_time = holding_info.get("entry_time")
                holding_time_minutes = holding_info.get("holding_time_minutes")

                if entry_time and holding_time_minutes:
                    elapsed_minutes = (current_time - entry_time).total_seconds() / 60
                    if elapsed_minutes >= holding_time_minutes:
                        try:
                            pos = next((p for p in positions if p.symbol == symbol), None)
                            if pos and pos.position_amt != 0:
                                qty = abs(pos.position_amt)
                                close_side = "sell" if pos.position_amt > 0 else "buy"

                                log_message(f"{symbol}: 持仓时间已到期（{holding_time_minutes} 分钟），自动平仓")
                                binance.place_market_order(side=close_side, quantity=qty, symbol=symbol)
                                log_message(f"{symbol}: 持仓时间到期平仓成功")
                                _position_holding_times.pop(symbol, None)
                                trading_status.setdefault("trailing_stops", {}).pop(f"{symbol}:long", None)
                                trading_status.setdefault("trailing_stops", {}).pop(f"{symbol}:short", None)
                        except Exception as e:
                            log_message(f"{symbol}: 持仓时间到期平仓失败: {e}")

            trailing_registry = trading_status.setdefault("trailing_stops", {})
            for p in positions:
                symbol = p.symbol
                amt = p.position_amt
                if amt == 0:
                    _position_holding_times.pop(symbol, None)
                    trailing_registry.pop(f"{symbol}:long", None)
                    trailing_registry.pop(f"{symbol}:short", None)
                    continue

                position_side = "long" if amt > 0 else "short"
                guard_key = f"{symbol}:{position_side}"
                closing_side = "SELL" if position_side == "long" else "BUY"
                qty = abs(amt)
                entry_price = p.entry_price or 0.0
                if entry_price <= 0:
                    try:
                        entry_price = binance.get_symbol_price(symbol)
                    except Exception:
                        entry_price = 0.0

                symbol_orders = open_orders_by_symbol.get(symbol, [])

                # 订单清理逻辑（已移除数量限制）
                _cleanup_orders_for_symbol(
                    binance=binance,
                    symbol=symbol,
                    position_side=position_side,
                    symbol_orders=symbol_orders,
                    current_position_qty=qty,
                )

                # 检查是否有足够的委托单（至少应该有移动止损）
                has_trailing = _has_trailing_stop(symbol_orders, closing_side)
                # 统计当前委托单数量
                current_order_count = len(symbol_orders)
                
                if not has_trailing:
                    if not _has_any_protective_order(symbol_orders, closing_side):
                        log_message(f"{symbol}: 检测到现有持仓缺少止盈/止损委托，自动补齐移动止损以避免裸奔。")
                    ensure_trailing_stop_for_position(
                        binance=binance,
                        symbol=symbol,
                        position_side=position_side,
                        quantity=qty,
                        entry_price=entry_price or binance.get_symbol_price(symbol),
                        source="持仓巡检（仅移动止损）",
                    )
                elif current_order_count < 3:
                    # 如果委托单数量少于3个，说明可能缺少做T的限价单
                    log_message(f"{symbol}: 当前委托单数量 {current_order_count} 个")
                else:
                    trailing_registry[guard_key] = trailing_registry.get(guard_key, "exchange")

            # 间隔一段时间再检查，避免频繁调用接口
            for _ in range(10):
                if not trading_status["running"]:
                    break
                time.sleep(1)
        except Exception as e:
            log_message(f"持仓守护线程异常: {e}")
            time.sleep(10)

    log_message("持仓守护线程已停止")


def maybe_run_learning_review(deepseek_client, binance_trade_history: Optional[List[Dict[str, Any]]] = None):
    """
    每累积 10 条决策，就将最近 10 条决策与收益摘要发送给 DeepSeek，请求自学习反馈，
    并根据反馈自动调整策略模式。同时提取知识并更新知识库。
    """
    history = trading_status.get("decisions", [])
    total = len(history)
    if total < 10 or total % 10 != 0:
        return
    if trading_status.get("last_learning_count") == total:
        return

    recent = history[-10:]
    start_equity = recent[0].get("equity") or trading_status.get("equity")
    end_equity = recent[-1].get("equity") or trading_status.get("equity")
    pnl = (end_equity or 0.0) - (start_equity or 0.0)

    metrics = {
        "total_decisions": total,
        "window_size": 10,
        "start_equity": start_equity,
        "end_equity": end_equity,
        "pnl": pnl,
        "wins": sum(1 for d in recent if d.get("status") == "ordered"),
        "skips": sum(1 for d in recent if d.get("status") == "skipped"),
        "fails": sum(1 for d in recent if d.get("status") == "failed"),
    }

    try:
        feedback = deepseek_client.get_learning_feedback(recent, metrics)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feedback_entry = {
            "timestamp": timestamp,
            "decision_count": total,
            "summary": feedback.get("summary", "暂无总结"),
            "risk_assessment": feedback.get("risk_assessment", ""),
            "recommended_strategy": feedback.get("recommended_strategy", ""),
            "action_items": feedback.get("action_items", []),
            "pnl": pnl,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "wins": metrics.get("wins", 0),
            "skips": metrics.get("skips", 0),
            "fails": metrics.get("fails", 0),
        }
        
        trading_status["strategy_feedback"] = feedback_entry
        trading_status["last_learning_count"] = total
        
        # 持久化保存学习总结
        save_learning_feedback(feedback_entry)

        socketio.emit("learning_feedback", feedback_entry)
        log_message(f"🤖 自学习总结: {feedback.get('summary', '无')}（已保存，重启后不丢失）")

        # 提取知识并更新知识库
        try:
            knowledge_base = load_knowledge_base()
            extracted_knowledge = deepseek_client.extract_knowledge_from_feedback(
                feedback=feedback,
                decisions=recent,
                metrics=metrics,
                binance_trade_history=binance_trade_history,
            )
            
            # 合并新提取的知识到知识库（去重，只保留最近50条）
            if extracted_knowledge:
                for key in ["successful_patterns", "failed_patterns", "best_practices", 
                           "timing_experience", "stop_loss_take_profit_experience"]:
                    new_items = extracted_knowledge.get(key, [])
                    existing_items = knowledge_base.get(key, [])
                    # 合并并去重（简单的字符串去重）
                    combined = existing_items + new_items
                    # 去重：保留第一次出现的
                    seen = set()
                    unique_items = []
                    for item in combined:
                        item_str = str(item) if not isinstance(item, str) else item
                        if item_str not in seen:
                            seen.add(item_str)
                            unique_items.append(item)
                    # 只保留最近50条
                    knowledge_base[key] = unique_items[-50:]
                
                # 对于market_conditions，需要特殊处理（是字典列表）
                new_conditions = extracted_knowledge.get("market_conditions", [])
                existing_conditions = knowledge_base.get("market_conditions", [])
                combined_conditions = existing_conditions + new_conditions
                # 简单去重：基于condition和strategy的组合
                seen_conditions = set()
                unique_conditions = []
                for cond in combined_conditions:
                    key_str = f"{cond.get('condition', '')}:{cond.get('strategy', '')}"
                    if key_str not in seen_conditions:
                        seen_conditions.add(key_str)
                        unique_conditions.append(cond)
                knowledge_base["market_conditions"] = unique_conditions[-50:]
                
                # 保存更新后的知识库
                save_knowledge_base(knowledge_base)
                log_message(f"📚 已提取并更新知识库（成功模式: {len(extracted_knowledge.get('successful_patterns', []))}条，失败模式: {len(extracted_knowledge.get('failed_patterns', []))}条）")
        except Exception as e:
            log_message(f"⚠️  知识提取失败: {type(e).__name__}: {e}")

        # 如果 DeepSeek 建议切换策略模式，则立即更新
        rec_mode = feedback.get("recommended_strategy")
        if rec_mode and rec_mode != trading_status.get("strategy_mode"):
            trading_status["strategy_mode"] = rec_mode
            log_message(f"策略模式已根据自学习反馈自动调整为：{rec_mode}")
    except Exception as e:
        log_message(f"自学习反馈调用失败: {type(e).__name__}: {e}")
        import traceback
        log_message(f"详细错误: {traceback.format_exc()[:200]}")


def parse_interval_seconds(interval_str: str) -> int:
    """
    将 INTERVAL（如 '1m', '3m', '5m', '15m'）转换为秒数。
    支持：
      - Xm : 分钟
      - Xs : 秒
      - Xh : 小时
    默认返回 60 秒。
    """
    if not interval_str:
        return 60
    s = interval_str.strip().lower()
    try:
        if s.endswith("m"):
            minutes = float(s[:-1])
            return int(minutes * 60)
        if s.endswith("s"):
            seconds = float(s[:-1])
            return int(seconds)
        if s.endswith("h"):
            hours = float(s[:-1])
            return int(hours * 3600)
        # 如果是纯数字，按秒处理
        return int(float(s))
    except Exception:
        # 解析失败，退回默认 60 秒
        log_message(f"警告：无法解析 INTERVAL='{interval_str}'，使用默认 60 秒周期")
        return 60


def run_trading_bot():
    """运行交易机器人（在主线程中）"""
    global trading_status
    
    try:
        from binance_client import BinanceWrapper
        from config import get_binance_config, get_deepseek_config, get_risk_config
        from deepseek_client import DeepSeekClient
        from risk import OrderPlan
        from strategy import generate_order_plan
        
        binance_cfg = get_binance_config()
        deepseek_cfg = get_deepseek_config()
        risk_cfg = get_risk_config()
        
        if not binance_cfg.api_key or not binance_cfg.api_secret:
            log_message("错误：未配置 BINANCE_API_KEY / BINANCE_API_SECRET")
            trading_status["running"] = False
            return
        
        binance = BinanceWrapper(binance_cfg)
        deepseek = DeepSeekClient(deepseek_cfg)
        
        initial_equity = binance.get_account_equity()
        trading_status["equity"] = initial_equity
        log_message(f"启动时合约账户资金: {initial_equity:.4f} {binance_cfg.base_asset}")
        
        # 获取所有历史交易记录（用于DeepSeek学习）
        log_message("正在获取币安账户所有历史交易记录...")
        try:
            all_trade_history = binance.get_all_trade_history()
            log_message(f"✅ 已获取 {len(all_trade_history)} 条历史交易记录，将用于DeepSeek学习")
        except Exception as e:
            log_message(f"⚠️  获取历史交易记录失败: {e}，将使用空记录")
            all_trade_history = []
        
        # 加载知识库（积累的学习经验）
        knowledge_base = load_knowledge_base()
        log_message(f"📚 已加载知识库（成功模式: {len(knowledge_base.get('successful_patterns', []))}条，失败模式: {len(knowledge_base.get('failed_patterns', []))}条）")
        
        # 仅交易 ETH
        symbols = ["ETHUSDT"]
        
        if not symbols:
            log_message("错误：未能获取可交易合约")
            trading_status["running"] = False
            return
        
        trading_status["interval"] = binance_cfg.interval
        # 初始策略模式（可通过 /api/config 更新）
        trading_status["strategy_mode"] = os.getenv("STRATEGY_MODE", "balanced")
        log_message(
            f"自动识别 {len(symbols)} 个交易对: {symbols}, 周期: {binance_cfg.interval}, "
            f"策略模式: {trading_status['strategy_mode']}"
        )

        # 启动持仓守护线程（移动止盈 / 自适应止损）
        guard_thread = threading.Thread(
            target=position_guard_loop,
            args=(binance, risk_cfg),
            daemon=True,
        )
        guard_thread.start()
        
        while trading_status["running"]:
            try:
                # 每个大循环开始时，根据当前配置动态计算周期秒数
                current_interval = trading_status.get("interval") or binance_cfg.interval
                current_strategy_mode = trading_status.get("strategy_mode") or os.getenv(
                    "STRATEGY_MODE", "balanced"
                )
                interval_seconds = parse_interval_seconds(current_interval)

                # 更新合约账户资金和回撤
                current_equity = binance.get_account_equity()
                trading_status["equity"] = current_equity
                if initial_equity > 0:
                    drawdown = (initial_equity - current_equity) / initial_equity
                    trading_status["drawdown"] = drawdown
                    
                    if drawdown >= risk_cfg.max_drawdown:
                        log_message(f"⚠️ 合约账户资金回撤 {drawdown:.2%} 已触及上限 {risk_cfg.max_drawdown:.2%}，执行紧急平仓并停止交易。")
                        if emergency_close_all_positions(binance):
                            log_message("✅ 紧急平仓完成，已停止交易以保护资金。")
                        else:
                            log_message("❌ 紧急平仓过程中部分交易对处理失败，请人工检查。")
                        trading_status["running"] = False
                        break
                    elif drawdown >= risk_cfg.max_drawdown * 0.8:
                        log_message(f"⚠️ 提示：当前回撤 {drawdown:.2%}，接近最大允许回撤 {risk_cfg.max_drawdown:.2%}")
                
                # 更新持仓信息
                positions = binance.get_positions()
                position_lookup = {p.symbol: p for p in positions}
                trading_status["positions"] = [
                    {
                        "symbol": p.symbol,
                        "amount": p.position_amt,
                        "entry_price": p.entry_price
                    }
                    for p in positions
                ]

                # 当前已有持仓的币种集合，用于控制“最多持仓 N 个币种”
                active_symbols = {
                    p.symbol for p in positions if getattr(p, "position_amt", 0) != 0
                }
                max_open_symbols = getattr(risk_cfg, "max_open_symbols", 0) or 0
                
                # 处理每个交易对
                for symbol in symbols:
                    if not trading_status["running"]:
                        break

                    # 确保每个交易对使用统一的 10x 杠杆（若设置失败，仅记录日志，不中断循环）
                    try:
                        binance.set_leverage(symbol, leverage=10)
                    except Exception as e:
                        log_message(f"{symbol}: 设置杠杆 10x 失败: {e}")

                    log_message(f"=== 处理交易对 {symbol} （策略: {current_strategy_mode}） ===")
                    klines = binance.get_k_lines(symbol=symbol, interval=current_interval, limit=200)
                    
                    # 获取实时市场数据（通过 CCXT）
                    market_data = {}
                    try:
                        market_data = deepseek._get_realtime_market_data(symbol)
                        if market_data:
                            ticker_info = ""
                            if 'ticker' in market_data:
                                t = market_data['ticker']
                                ticker_info = f"价格: {t.get('last', 0):.2f}, 24h涨跌: {t.get('change', 0):.2f}%"
                            if 'orderbook' in market_data:
                                ob = market_data['orderbook']
                                imbalance = ob.get('imbalance', 0)
                                imbalance_text = f"买卖盘不平衡: {imbalance:.2f}%"
                                if ticker_info:
                                    ticker_info += f", {imbalance_text}"
                                else:
                                    ticker_info = imbalance_text
                            if ticker_info:
                                log_message(f"{symbol}: 实时行情 - {ticker_info}")
                    except Exception as e:
                        log_message(f"{symbol}: 获取实时行情数据失败: {e}")

                    # 获取多周期 K 线摘要
                    multi_interval_summary = {}
                    try:
                        multi_interval_summary = binance.get_multi_interval_summary(symbol)
                    except Exception as e:
                        log_message(f"{symbol}: 获取多周期行情摘要失败: {e}")
                    
                    trend_context = analyze_trend_layers(multi_interval_summary)
                    
                    # 获取当前币种的实时持仓，用来判断是否允许开新仓
                    current_position = position_lookup.get(symbol)

                    # 若已达到最多持仓币种数，且当前无持仓，则本周期直接观望，避免再增加新币种持仓
                    if (
                        max_open_symbols > 0
                        and symbol not in active_symbols
                        and len(active_symbols) >= max_open_symbols
                    ):
                        reason = (
                            f"当前已持有 {len(active_symbols)} 个币种仓位，"
                            f"达到最多持仓 {max_open_symbols} 个币种的限制，本周期不再新开此币种仓位"
                        )
                        log_message(f"{symbol}: {reason}")
                        decision_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": symbol,
                            "strategy": current_strategy_mode,
                            "interval": current_interval,
                            "action": "flat",
                            "quantity": 0.0,
                            "entry": None,
                            "stop_loss": None,
                            "take_profit": None,
                            "status": "skipped",
                            "message": reason,
                            "equity": trading_status.get("equity"),
                        }
                        append_decision(decision_entry)
                        # 不取实时行情/多周期数据，直接跳过 DeepSeek 调用，减少开新仓尝试
                        continue

                    decision_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": symbol,
                        "strategy": current_strategy_mode,
                        "interval": current_interval,
                        "action": "pending",
                        "quantity": 0.0,
                        "entry": None,
                        "stop_loss": None,
                        "take_profit": None,
                        "status": "pending",
                        "message": "",
                        "equity": trading_status.get("equity"),
                        "market_data": market_data,  # 添加实时市场数据
                        "multi_timeframes": multi_interval_summary,
                        "trend_context": trend_context,
                    }
                    current_price = None
                    if market_data and market_data.get("ticker"):
                        current_price = market_data["ticker"].get("last")
                    if not current_price:
                        try:
                            current_price = binance.get_symbol_price(symbol)
                        except Exception:
                            current_price = None

                    plan: OrderPlan | None = generate_order_plan(
                        binance=binance,
                        deepseek=deepseek,
                        risk_cfg=risk_cfg,
                        symbol=symbol,
                        klines=klines,
                        strategy_mode=current_strategy_mode,
                        multi_timeframes=multi_interval_summary,
                        trend_context=trend_context,
                        recent_context=trading_status.get("recent_market_states", []),
                        current_position=current_position,
                        current_price=current_price,
                        binance_trade_history=all_trade_history,
                    )
                    
                    # 无法生成计划或计划数量为 0，都视为“本周期不下单”，但要把原因写到日志里
                    if not plan or plan.quantity <= 0:
                        reason = ""
                        if plan and getattr(plan, "reason", None):
                            reason = plan.reason
                        msg = (
                            f"{symbol}: 本周期观望或风控拒绝开仓"
                            + (f"，原因：{reason}" if reason else "")
                        )
                        log_message(msg)
                        
                        # 即使不下单，如果有持仓且DeepSeek返回了orders配置，也要创建委托单
                        if plan and current_position and abs(current_position.position_amt) > 0:
                            orders_config = getattr(plan, "orders", None)
                            if orders_config:
                                log_message(f"{symbol}: 持仓中，DeepSeek返回了 {len(orders_config)} 个委托单配置")
                            if orders_config and len(orders_config) > 0:
                                position_side = "long" if current_position.position_amt > 0 else "short"
                                position_qty = abs(current_position.position_amt)
                                
                                try:
                                    order_results = create_orders_from_deepseek_plan(
                                        binance=binance,
                                        symbol=symbol,
                                        position_side=position_side,
                                        position_quantity=position_qty,
                                        orders_config=orders_config,
                                    )
                                    log_message(f"{symbol}: 持仓中，已根据DeepSeek决策创建 {len(orders_config)} 个委托单（移动止损/止盈 + 限价做T单）")
                                except Exception as e:
                                    log_message(f"{symbol}: 为持仓创建委托单失败：{e}")
                        
                        decision_entry.update({
                            "action": "flat",
                            "status": "skipped",
                            "message": reason or "本周期观望或风控拒绝开仓",
                        })
                        append_decision(decision_entry)
                        record_market_state(decision_entry, market_data)
                        maybe_run_learning_review(deepseek, all_trade_history)
                    else:
                        formatted_qty = binance._format_quantity(plan.quantity, symbol)
                        decision_entry.update({
                            "action": plan.side,
                            "quantity": float(formatted_qty),
                            "entry": float(plan.entry_price or 0.0),
                            "stop_loss": float(plan.stop_loss or 0.0),
                            "take_profit": float(plan.take_profit or 0.0),
                            "status": "planned",
                            "message": getattr(plan, "reason", "准备下单"),
                        })
                        log_message(
                            f"{symbol}: 准备下单: side={plan.side}, qty={formatted_qty:.6f}, "
                            f"entry={plan.entry_price:.4f}, sl={plan.stop_loss:.4f}, tp={plan.take_profit:.4f}"
                        )
                        # 把本轮决策的详细“交易原因”也直接写进日志
                        if getattr(plan, "reason", None):
                            log_message(f"{symbol}: 决策说明：{plan.reason}")
                        try:
                            order = binance.place_market_order(
                                side=plan.side,
                                quantity=plan.quantity,
                                symbol=symbol,
                            )
                            log_message(f"{symbol}: 订单结果: {order.get('orderId', 'N/A')}")
                            
                            # 获取当前持仓数量（用于创建委托单）
                            position_side = "long" if plan.side.lower() == "buy" else "short"
                            entry_reference = plan.entry_price or binance.get_symbol_price(symbol)
                            
                            # 如果有DeepSeek返回的orders配置，创建委托单
                            orders_config = getattr(plan, "orders", None)
                            if orders_config:
                                log_message(f"{symbol}: DeepSeek返回了 {len(orders_config)} 个委托单配置")
                            if orders_config and len(orders_config) > 0:
                                # 获取当前持仓数量
                                current_position = position_lookup.get(symbol)
                                position_qty = abs(current_position.position_amt) if current_position else plan.quantity
                                
                                # 创建委托单
                                order_results = create_orders_from_deepseek_plan(
                                    binance=binance,
                                    symbol=symbol,
                                    position_side=position_side,
                                    position_quantity=position_qty,
                                    orders_config=orders_config,
                                )
                                log_message(f"{symbol}: 已根据DeepSeek决策创建 {len(orders_config)} 个委托单（移动止损/止盈 + 限价做T单）")
                            else:
                                # 如果DeepSeek没有返回orders配置，只创建基本的移动止损作为保护
                                # 不再使用旧的固定止损/止盈逻辑
                                log_message(f"{symbol}: DeepSeek未返回orders配置，仅创建基本移动止损保护")
                                ensure_trailing_stop_for_position(
                                    binance=binance,
                                    symbol=symbol,
                                    position_side=position_side,
                                    quantity=plan.quantity,
                                    entry_price=entry_reference,
                                    source="新开仓（无orders配置）",
                                )
                            
                            # 记录持仓时间（如果 DeepSeek 提供了）
                            holding_time = getattr(plan, "holding_time_minutes", None)
                            if holding_time and holding_time > 0:
                                log_message(f"{symbol}: 建议持仓时间 {holding_time} 分钟（由 DeepSeek 根据行情决定）")
                                # 将持仓时间信息存储到决策记录中
                                decision_entry["holding_time_minutes"] = holding_time
                                # 记录到持仓时间跟踪字典
                                _position_holding_times[symbol] = {
                                    "entry_time": datetime.now(),
                                    "holding_time_minutes": holding_time,
                                    "order_id": order.get('orderId', 'N/A'),
                                }
                            
                            decision_entry.update({
                                "status": "ordered",
                                "order_id": order.get('orderId', 'N/A'),
                                "message": (getattr(plan, "reason", "") + "；" if getattr(plan, "reason", None) else "") + "市价单及止盈止损已提交",
                            })
                        except Exception as e:
                            log_message(f"{symbol}: 下单失败: {e}")
                            decision_entry.update({
                                "status": "failed",
                                "message": f"下单失败: {e}"
                            })
                        append_decision(decision_entry)
                        record_market_state(decision_entry, market_data)
                        maybe_run_learning_review(deepseek, all_trade_history)
                
                # 发送状态更新
                socketio.emit('status_update', {
                    'equity': trading_status["equity"],
                    'drawdown': trading_status["drawdown"],
                    'positions': trading_status["positions"],
                    'last_update': trading_status["last_update"],
                    'interval': trading_status.get("interval"),
                    'strategy_mode': trading_status.get("strategy_mode"),
                    'decisions': trading_status.get("decisions", [])[-20:],
                    'strategy_feedback': trading_status.get("strategy_feedback"),
                    'strategy_feedback_history': trading_status.get("strategy_feedback_history", [])[-20:],  # 最近20条历史
                    'recent_market_states': trading_status.get("recent_market_states", [])[-30:],
                    'logs': trading_status.get("logs", [])[-50:],
                })
                
                # 等待下一个周期
                for _ in range(interval_seconds):
                    if not trading_status["running"]:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                log_message(f"运行出错: {e}")
                time.sleep(60)
        
        log_message("交易程序已停止")
        
    except Exception as e:
        log_message(f"程序启动失败: {e}")
        trading_status["running"] = False


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """获取交易状态"""
    return jsonify(trading_status)


@app.route('/api/learning_history')
def get_learning_history():
    """获取历史学习总结"""
    history = trading_status.get("strategy_feedback_history", [])
    return jsonify({
        "total": len(history),
        "history": history[-50:],  # 返回最近50条
    })


@app.route('/api/start', methods=['POST'])
def start_trading():
    """启动交易"""
    global trading_thread, trading_status
    
    if trading_status["running"]:
        return jsonify({"success": False, "message": "交易已在运行中"})
    
    trading_status["running"] = True
    trading_status["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trading_status["logs"] = []
    
    trading_thread = threading.Thread(target=run_trading_bot, daemon=True)
    trading_thread.start()
    
    log_message("交易程序已启动")
    return jsonify({"success": True, "message": "交易已启动"})


@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """停止交易"""
    global trading_status
    
    if not trading_status["running"]:
        return jsonify({"success": False, "message": "交易未在运行"})
    
    trading_status["running"] = False
    log_message("正在停止交易程序...")
    return jsonify({"success": True, "message": "交易已停止"})


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """获取或更新配置"""
    env_file = Path(__file__).parent / ".env"
    
    if request.method == 'GET':
        # 读取配置
        config_data = {}
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config_data[key.strip()] = value.strip()
        return jsonify(config_data)
    
    else:  # POST
        # 更新配置
        data = request.json
        if not env_file.exists():
            return jsonify({"success": False, "message": ".env 文件不存在"})
        
        # 读取现有配置
        lines = []
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        
        # 更新配置
        updated_keys = set()
        new_lines = []
        for line in lines:
            if '=' in line and not line.strip().startswith('#'):
                key = line.split('=', 1)[0].strip()
                if key in data:
                    new_lines.append(f"{key}={data[key]}\n")
                    updated_keys.add(key)
                else:
                    if not line.endswith('\n'):
                        line = line + '\n'
                    new_lines.append(line)
            else:
                if line and not line.endswith('\n'):
                    line = line + '\n'
                new_lines.append(line)
        
        # 添加新配置
        for key, value in data.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}\n")
        
        # 写入文件
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        # 如果前端更新了 INTERVAL，同步到内存状态，下一轮循环立即生效
        if "INTERVAL" in data:
            trading_status["interval"] = data["INTERVAL"]
            log_message(f"运行周期已更新为 {data['INTERVAL']}（将从下一个周期开始生效）")

        if "STRATEGY_MODE" in data:
            trading_status["strategy_mode"] = data["STRATEGY_MODE"]
            log_message(f"策略模式已更新为 {data['STRATEGY_MODE']}（将从下一个周期开始生效）")

        return jsonify({"success": True, "message": "配置已更新"})


@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    emit('status_update', {
        'equity': trading_status["equity"],
        'drawdown': trading_status["drawdown"],
        'positions': trading_status["positions"],
        'last_update': trading_status["last_update"],
        'interval': trading_status.get("interval"),
        'strategy_mode': trading_status.get("strategy_mode"),
        'decisions': trading_status.get("decisions", [])[-20:],
        'strategy_feedback': trading_status.get("strategy_feedback"),
        'strategy_feedback_history': trading_status.get("strategy_feedback_history", [])[-20:],  # 最近20条历史
        'recent_market_states': trading_status.get("recent_market_states", [])[-30:],
        'logs': trading_status.get("logs", [])[-50:],
    })


if __name__ == '__main__':
    # 启动时加载历史学习总结
    load_learning_feedback()
    
    # 获取端口（环境变量优先，默认 5000）
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("Binance 智能交易系统 - Web 界面")
    print("=" * 60)
    print(f"访问地址: http://0.0.0.0:{port}")
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    # 生产环境建议使用 Gunicorn，这里保留开发模式
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)

