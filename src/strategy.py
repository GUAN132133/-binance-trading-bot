from __future__ import annotations

from typing import Any, Dict, List, Optional

from binance_client import BinanceWrapper, PositionInfo
from config import RiskConfig
from deepseek_client import DeepSeekClient
from risk import OrderPlan, calc_order_quantity_with_risk


def generate_order_plan(
    binance: BinanceWrapper,
    deepseek: DeepSeekClient,
    risk_cfg: RiskConfig,
    symbol: str,
    klines: List[List[Any]],
    strategy_mode: str = "balanced",
    multi_timeframes: Optional[Dict[str, Any]] = None,
    trend_context: Optional[Dict[str, Any]] = None,
    recent_context: Optional[List[Dict[str, Any]]] = None,
    current_position: Optional[PositionInfo] = None,
    current_price: Optional[float] = None,
    binance_trade_history: Optional[List[Dict[str, Any]]] = None,
    knowledge_base: Optional[Dict[str, Any]] = None,
) -> Optional[OrderPlan]:
    """综合 DeepSeek 信号与风险控制，返回最终可执行下单计划。"""
    position_payload: Optional[Dict[str, Any]] = None
    if current_position:
        direction = "long" if current_position.position_amt > 0 else "short"
        qty = abs(current_position.position_amt)
        entry_price = current_position.entry_price
        pnl_pct = 0.0
        if entry_price and current_price:
            pnl_pct = ((current_price - entry_price) / entry_price) * (1 if direction == "long" else -1)

        position_payload = {
            "direction": direction,
            "quantity": qty,
            "entry_price": entry_price,
            "unrealized_pnl_pct": pnl_pct * 100,
        }

    signal: Dict[str, Any] = deepseek.get_trade_signal(
        klines,
        symbol,
        strategy_mode=strategy_mode,
        multi_timeframes=multi_timeframes,
        trend_context=trend_context,
        recent_context=recent_context,
        position_state=position_payload,
        binance_trade_history=binance_trade_history,
        knowledge_base=knowledge_base,
    )

    # 从 DeepSeek 信号中抽取动作、持仓操作与文字说明
    action = str(signal.get("action", "flat")).lower()
    position_action = str(signal.get("position_action", "open")).lower()
    comment = str(signal.get("comment", "") or "").strip()
    try:
        size_pct = float(signal.get("size_pct") or 0.0)
    except (TypeError, ValueError):
        size_pct = 0.0
    if size_pct < 0:
        size_pct = 0.0
    if size_pct > 1:
        size_pct = 1.0

    # 如果 DeepSeek 建议观望（flat），也返回一个“数量为 0 的计划”，
    # 这样上层可以把观望理由展示在网页日志 / 决策列表中。
    if action not in {"long", "short"}:
        entry_price = float(
            signal.get("entry_price") or binance.get_symbol_price(symbol)
        )
        stop_loss = float(signal.get("stop_loss") or 0.0)
        take_profit = float(signal.get("take_profit") or 0.0)

        reason = "DeepSeek 建议本周期观望"
        if comment:
            reason += f"：{comment}"

        return OrderPlan(
            side="buy",  # 占位，不会实际下单
            quantity=0.0,
            reason=reason,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    entry_price = float(
        signal.get("entry_price") or binance.get_symbol_price(symbol)
    )
    stop_loss = float(signal.get("stop_loss") or 0.0)
    # 止盈价格完全使用 DeepSeek 给出的 take_profit（如果不给，则视为无固定止盈，只用止损保护）
    take_profit = float(signal.get("take_profit") or 0.0)

    # DeepSeek 建议的单标的仓位上限（如果不给，就用风控配置）
    try:
        max_position_pct = float(signal.get("max_position_pct") or 0.0)
    except (TypeError, ValueError):
        max_position_pct = 0.0
    if max_position_pct <= 0:
        max_position_pct = 0.05  # 默认 5% 资金
    max_position_pct = min(max_position_pct, 1.0)

    side = "buy" if action == "long" else "sell"

    # 处理已有持仓时的加仓/减仓/平仓/反手/保持
    if current_position and abs(current_position.position_amt) > 0:
        current_side = "long" if current_position.position_amt > 0 else "short"
        current_qty = abs(current_position.position_amt)

        # 保持仓位不动：仅记录理由
        if position_action == "hold":
            reason = "保持当前持仓不动"
            if comment:
                reason = f"策略理由：{comment}；{reason}"
            return OrderPlan(
                side=side,
                quantity=0.0,
                reason=reason,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # 全部平仓或反手：本周期只执行“平仓”部分，反手交给后续周期
        if position_action in {"close", "reverse"}:
            close_side = "sell" if current_side == "long" else "buy"
            reason = "平掉当前全部持仓"
            if position_action == "reverse":
                reason += "（DeepSeek 建议反手，系统将先平仓，后续周期再考虑反向开仓）"
            if comment:
                reason = f"策略理由：{comment}；{reason}"
            return OrderPlan(
                side=close_side,
                quantity=current_qty,
                reason=reason,
                entry_price=current_price or entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # 减仓：在当前方向上部分平仓
        if position_action == "reduce" and size_pct > 0:
            close_side = "sell" if current_side == "long" else "buy"
            reduce_qty = current_qty * size_pct
            if reduce_qty <= 0:
                reduce_qty = 0.0
            reason = f"部分减仓 {size_pct:.0%} 以锁定收益或控制风险"
            if comment:
                reason = f"策略理由：{comment}；{reason}"
            return OrderPlan(
                side=close_side,
                quantity=reduce_qty,
                reason=reason,
                entry_price=current_price or entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # 加仓：在当前方向上按风控上限加一部分仓位
        if position_action == "add" and size_pct > 0:
            # 先用风险引擎计算“这次最多还能加多少”
            add_plan = calc_order_quantity_with_risk(
                client=binance,
                risk_cfg=risk_cfg,
                side=current_side,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                symbol=symbol,
                max_position_pct=max_position_pct,
            )
            add_qty = add_plan.quantity * size_pct
            if add_qty <= 0:
                return OrderPlan(
                    side=side,
                    quantity=0.0,
                    reason="由于风险约束，加仓数量被压缩为 0",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

            # 将 DeepSeek 的理由与风控说明合并
            reason = add_plan.reason
            if comment:
                reason = f"策略理由：{comment}；风控说明：{reason}"

            return OrderPlan(
                side="buy" if current_side == "long" else "sell",
                quantity=add_qty,
                reason=reason,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

    # 没有持仓（或 DeepSeek 未指定 position_action）：按原有逻辑开新仓
    plan = calc_order_quantity_with_risk(
        client=binance,
        risk_cfg=risk_cfg,
        side=side,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        symbol=symbol,
        max_position_pct=max_position_pct,
    )

    # 把止损和止盈价格写回计划，供下单和止损止盈单使用
    plan.take_profit = take_profit

    # 提取持仓时间（分钟），由 DeepSeek 决定
    holding_time = signal.get("holding_time_minutes")
    if holding_time is not None:
        plan.holding_time_minutes = int(holding_time)

    # 提取DeepSeek返回的orders配置
    orders_config = signal.get("orders", [])
    if orders_config:
        plan.orders = orders_config
        # 使用print输出到标准输出，会被systemd捕获到journalctl
        print(f"[DEBUG] {symbol}: DeepSeek返回了 {len(orders_config)} 个委托单配置")
        print(f"[DEBUG] {symbol}: orders配置详情: {orders_config}")
    else:
        signal_keys = list(signal.keys())
        print(f"[DEBUG] {symbol}: DeepSeek未返回orders配置，signal keys: {signal_keys}")
        # 检查是否有orders字段但值为None或空
        if "orders" in signal:
            print(f"[DEBUG] {symbol}: signal中有orders字段，但值为: {signal.get('orders')}")

    # 将 DeepSeek 的"交易理由"合并到风控理由中，方便在网页日志中展示完整原因
    if comment:
        plan.reason = f"策略理由：{comment}；风控说明：{plan.reason}"

    return plan


