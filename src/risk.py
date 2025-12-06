from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from binance_client import BinanceWrapper
from config import RiskConfig


@dataclass
class OrderPlan:
    side: str  # "buy" or "sell"
    quantity: float
    reason: str
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    holding_time_minutes: Optional[int] = None  # 持仓时间（分钟），由 DeepSeek 决定
    orders: List[Dict[str, Any]] = field(default_factory=list)  # DeepSeek返回的委托单配置


def calc_order_quantity_with_risk(
    client: BinanceWrapper,
    risk_cfg: RiskConfig,
    side: str,
    entry_price: float,
    stop_loss_price: float,
    symbol: str,
    max_position_pct: float,
) -> OrderPlan:
    """
    根据合约账户资金和 DeepSeek 给出的仓位比例计算下单数量。
    已移除所有限制条件，完全尊重 DeepSeek 的仓位决策。
    """
    target_pct = float(max_position_pct or 0.0)
    if target_pct <= 0:
        target_pct = 0.05  # 默认使用 5% 的仓位比例
    target_pct = min(target_pct, 1.0)  # 不允许超过账户资金 100%

    equity = client.get_account_equity()
    leverage = 11.0

    # 目标名义（来自 DeepSeek 的 max_position_pct），按杠杆放大
    target_notional = equity * target_pct * leverage
    plan_reason = "根据 DeepSeek 指定的仓位比例自动计算的目标仓位"

    qty = target_notional / entry_price if entry_price > 0 else 0.0

    if qty <= 0:
        return OrderPlan(
            side=side,
            quantity=0.0,
            reason="DeepSeek 给出的仓位比例导致下单数量为 0，请检查信号",
            entry_price=entry_price,
            stop_loss=stop_loss_price,
        )

    # 格式化数量，确保符合 Binance 步长要求
    qty = client._format_quantity(qty, symbol)

    return OrderPlan(
        side=side,
        quantity=qty,
        reason=plan_reason,
        entry_price=entry_price,
        stop_loss=stop_loss_price,
    )


