from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from config import DeepSeekConfig


class DeepSeekClient:
    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg
        self.ccxt_exchange: Optional[Any] = None
        if CCXT_AVAILABLE:
            try:
                # 初始化 CCXT Binance 交易所实例（用于获取实时行情数据）
                self.ccxt_exchange = ccxt.binance({
                    'apiKey': '',  # CCXT 仅用于公共数据，不需要 API key
                    'secret': '',
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',  # 使用合约市场
                    }
                })
            except Exception as e:
                print(f"警告: CCXT 初始化失败，将仅使用 K 线数据: {e}")
                self.ccxt_exchange = None

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def _post_chat(self, prompt: str, timeout: float = 30.0, max_retries: int = 3) -> Dict[str, Any]:
        if not self.cfg.api_key:
            raise RuntimeError("missing DEEPSEEK api key")

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": "You are a professional quantitative trading assistant."},
                {"role": "user", "content": prompt},
            ],
        }

        url = f"{self.cfg.api_base.rstrip('/')}/chat/completions"
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    sleep_time = 3 * (attempt + 1)
                    print(f"DeepSeek 请求超时/失败，第 {attempt + 1} 次重试前等待 {sleep_time}s：{e}")
                    time.sleep(sleep_time)
                else:
                    raise

        if last_error:
            raise last_error

    def _get_realtime_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        使用 CCXT 获取实时市场数据（ticker、订单簿、深度等）
        如果 CCXT 不可用或获取失败，返回空字典
        """
        if not CCXT_AVAILABLE or not self.ccxt_exchange:
            return {}
        
        market_data = {}
        try:
            # 获取 ticker（24小时价格统计）
            ticker = self.ccxt_exchange.fetch_ticker(symbol)
            last = ticker.get('last') or 0
            bid = ticker.get('bid') or 0
            ask = ticker.get('ask') or 0
            # 计算买卖价差（避免 None 值导致的错误）
            bid_ask_spread = ((ask - bid) / last * 100) if (last and bid and ask) else 0
            market_data['ticker'] = {
                'last': last,
                'bid': bid,
                'ask': ask,
                'high': ticker.get('high') or 0,
                'low': ticker.get('low') or 0,
                'volume': ticker.get('quoteVolume') or 0,
                'change': ticker.get('percentage') or 0,
                'bid_ask_spread': bid_ask_spread,
            }
        except Exception as e:
            print(f"获取 ticker 失败: {e}")
        
        try:
            # 获取订单簿（前20档买卖盘）
            orderbook = self.ccxt_exchange.fetch_order_book(symbol, limit=20)
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            market_data['orderbook'] = {
                'bid_depth': len(bids),
                'ask_depth': len(asks),
                'top_bid': bids[0] if bids else None,
                'top_ask': asks[0] if asks else None,
                'bid_volume': sum([b[1] for b in bids[:5]]),  # 前5档买单总量
                'ask_volume': sum([a[1] for a in asks[:5]]),  # 前5档卖单总量
                'imbalance': (sum([b[1] for b in bids[:5]]) - sum([a[1] for a in asks[:5]])) / (sum([b[1] for b in bids[:5]]) + sum([a[1] for a in asks[:5]]) + 1e-10) * 100,  # 买卖盘不平衡度
            }
        except Exception as e:
            print(f"获取订单簿失败: {e}")
        
        try:
            # 获取最近成交记录（trades）
            trades = self.ccxt_exchange.fetch_trades(symbol, limit=50)
            if trades:
                buy_volume = sum([t['amount'] for t in trades if t['side'] == 'buy'])
                sell_volume = sum([t['amount'] for t in trades if t['side'] == 'sell'])
                market_data['recent_trades'] = {
                    'count': len(trades),
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'buy_sell_ratio': buy_volume / (sell_volume + 1e-10),
                    'avg_price': sum([t['price'] * t['amount'] for t in trades]) / sum([t['amount'] for t in trades]) if trades else None,
                }
        except Exception as e:
            print(f"获取成交记录失败: {e}")
        
        return market_data

    def get_trade_signal(
        self,
        klines: List[List[Any]],
        symbol: str,
        strategy_mode: str = "balanced",
        multi_timeframes: Optional[Dict[str, Any]] = None,
        trend_context: Optional[Dict[str, Any]] = None,
        recent_context: Optional[List[Dict[str, Any]]] = None,
        position_state: Optional[Dict[str, Any]] = None,
        binance_trade_history: Optional[List[Dict[str, Any]]] = None,
        knowledge_base: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        调用 DeepSeek，返回结构化交易信号，支持多种策略模式。
        现在会使用 CCXT 获取实时行情数据（ticker、订单簿、深度等）来辅助判断。
        strategy_mode: 'conservative' | 'balanced' | 'aggressive' | 'scalping' | 'sniper'
        binance_trade_history: 币安账户所有历史交易记录（成交记录），用于深度学习
        """
        mode = (strategy_mode or "balanced").lower()
        
        # 解析策略模式：支持 "conservative_ultrashort", "conservative_trend" 等格式
        is_ultrashort = False
        is_trend = False
        base_mode = mode
        
        if "_ultrashort" in mode:
            base_mode = mode.replace("_ultrashort", "")
            is_ultrashort = True
        elif "_trend" in mode:
            base_mode = mode.replace("_trend", "")
            is_trend = True
        
        if base_mode not in {
            "conservative",
            "balanced",
            "aggressive",
        }:
            base_mode = "balanced"

        mode_desc_map = {
            "conservative": (
                "策略模式：保守型（AI），风险等级：2-3分（1-10分制，分数越低风险越低）。\n"
                "目标：控制回撤、尽量避免亏损，宁可少做或不做，也不要激进追涨杀跌。\n"
                "要求：\n"
                "- 只在趋势非常明确、性价比高的位置进场\n"
                "- 止损要相对紧一些，优先保护本金\n"
                "- 建议仓位较轻（每次 3%-5% 左右）\n"
                "- 委托单规划：数量较少（2-4单），优先保护本金，止损较紧，做T频率低，分批止盈保守\n"
            ),
            "balanced": (
                "策略模式：均衡型（AI），风险等级：5-6分（1-10分制，分数越低风险越低）。\n"
                "目标：在风险可控的前提下获取稳定收益，兼顾胜率和盈亏比。\n"
                "要求：\n"
                "- 顺势交易为主，适当参与震荡区间套利\n"
                "- 止损和止盈设置中等，盈亏比建议 >= 1.5:1\n"
                "- 建议仓位中等（每次 5%-8% 左右）\n"
                "- 委托单规划：数量中等（4-7单），兼顾风险与收益，止损止盈适中，适度做T，分批止盈均衡\n"
            ),
            "aggressive": (
                "策略模式：激进型（AI），风险等级：8-9分（1-10分制，分数越低风险越低）。\n"
                "目标：在行情有较大波动和趋势时，积极博取更高收益，接受更大回撤。\n"
                "要求：\n"
                "- 优先捕捉强趋势突破、加速行情\n"
                "- 止损可以放宽一些，但必须设置，避免爆仓\n"
                "- 止盈目标可以更远，盈亏比建议 >= 2:1\n"
                "- 建议仓位偏重（每次 8%-12% 左右），但不要满仓\n"
                "- 委托单规划：数量较多，追求更高收益，止损可放宽，积极做T，分批止盈激进\n"
            ),
        }

        # 获取实时市场数据（通过 CCXT）
        market_data = self._get_realtime_market_data(symbol)
        
        base_prompt = (
            "你是一个专业量化交易模型，请根据给定的 K 线数据和实时行情数据，为币安合约给出交易建议。\n\n"
        )

        # 构建市场数据描述
        market_info = ""
        if market_data:
            market_info = "\n实时市场数据：\n"
            if 'ticker' in market_data:
                t = market_data['ticker']
                market_info += f"- 当前价格: {t.get('last')}\n"
                market_info += f"- 买一价: {t.get('bid')}, 卖一价: {t.get('ask')}\n"
                market_info += f"- 24h最高: {t.get('high')}, 24h最低: {t.get('low')}\n"
                market_info += f"- 24h成交量: {t.get('volume')}\n"
                market_info += f"- 24h涨跌幅: {t.get('change')}%\n"
                market_info += f"- 买卖价差: {t.get('bid_ask_spread', 0):.4f}%\n"
            
            if 'orderbook' in market_data:
                ob = market_data['orderbook']
                market_info += f"- 订单簿: 买盘深度 {ob.get('bid_depth')}档, 卖盘深度 {ob.get('ask_depth')}档\n"
                if ob.get('top_bid'):
                    market_info += f"- 买一: {ob['top_bid'][0]} (量: {ob['top_bid'][1]})\n"
                if ob.get('top_ask'):
                    market_info += f"- 卖一: {ob['top_ask'][0]} (量: {ob['top_ask'][1]})\n"
                market_info += f"- 前5档买卖盘不平衡度: {ob.get('imbalance', 0):.2f}% (正数表示买盘更强)\n"
            
            if 'recent_trades' in market_data:
                rt = market_data['recent_trades']
                market_info += f"- 最近成交: {rt.get('count')}笔, 买卖比: {rt.get('buy_sell_ratio', 1):.2f}\n"
                if rt.get('avg_price'):
                    market_info += f"- 平均成交价: {rt['avg_price']}\n"
        else:
            market_info = "\n注意: 实时市场数据获取失败，仅基于 K 线数据进行分析。\n"

        # 根据超短/趋势模式添加说明
        holding_time_instruction = ""
        if is_ultrashort:
            holding_time_instruction = (
                "\n持仓时间要求（超短模式）：\n"
                "- 根据当前行情波动、订单簿深度、成交量活跃度等实时数据，判断适合的超短持仓时间\n"
                "- 超短模式通常持仓时间较短（建议 5-30 分钟），适合快速获利了结\n"
                "- 如果订单簿深度较浅、波动较大，建议更短的持仓时间\n"
                "- 如果成交量活跃、趋势明确，可以适当延长持仓时间\n"
            )
        elif is_trend:
            holding_time_instruction = (
                "\n持仓时间要求（趋势模式）：\n"
                "- 根据当前趋势强度、成交量持续性、订单簿深度等实时数据，判断适合的趋势持仓时间\n"
                "- 趋势模式通常持仓时间较长（建议 30 分钟-数小时），适合跟随趋势获取更大利润\n"
                "- 如果趋势明确、成交量持续放大，建议更长的持仓时间\n"
                "- 如果趋势较弱、波动较大，可以适当缩短持仓时间\n"
            )
        
        # 构建多周期行情描述
        multi_tf_info = ""
        if multi_timeframes:
            multi_tf_info = "\n多周期行情摘要（覆盖最近 30 天）：\n"
            for interval, stats in multi_timeframes.items():
                ema12 = stats.get('ema_12')
                ema26 = stats.get('ema_26')
                rsi14 = stats.get('rsi_14')
                macd = stats.get('macd')
                macd_signal = stats.get('macd_signal')
                macd_hist = stats.get('macd_hist')
                extra_indicators = []
                if ema12 is not None and ema26 is not None:
                    extra_indicators.append(f"EMA12 {ema12:.4f} / EMA26 {ema26:.4f}")
                if rsi14 is not None:
                    extra_indicators.append(f"RSI14 {rsi14:.2f}")
                if macd is not None and macd_signal is not None:
                    hist_text = f"{macd_hist:.4f}" if macd_hist is not None else "--"
                    extra_indicators.append(f"MACD {macd:.4f} / Signal {macd_signal:.4f} / Hist {hist_text}")

                multi_tf_info += (
                    f"- {interval}: 收盘 {stats.get('latest_close', '--'):.4f}, "
                    f"30日变动 {stats.get('change_pct', 0):.2f}%, "
                    f"高/低 {stats.get('high', '--'):.4f}/{stats.get('low', '--'):.4f}, "
                    f"波动 {stats.get('volatility_pct', 0):.2f}%, "
                    f"成交量 {stats.get('total_volume', 0):.2f}, 趋势 {stats.get('trend', 'N/A')}"
                )
                if extra_indicators:
                    multi_tf_info += f"，指标：{'；'.join(extra_indicators)}\n"
                else:
                    multi_tf_info += "\n"

        trend_context_info = ""
        if trend_context:
            macro = trend_context.get("macro") or {}
            micro = trend_context.get("micro") or {}
            macro_bias = trend_context.get("macro_bias")
            micro_bias = trend_context.get("micro_bias")
            guidance = trend_context.get("guidance")

            trend_context_info = "\n多周期层级判断（宏观 vs 小周期）：\n"
            if macro:
                trend_context_info += f"- 宏观({', '.join(macro.get('intervals', []))})：{macro.get('summary', '')}\n"
            if micro:
                trend_context_info += f"- 小周期({', '.join(micro.get('intervals', []))})：{micro.get('summary', '')}\n"
            if macro_bias:
                trend_context_info += f"- 宏观方向偏好：{macro_bias}\n"
            if micro_bias:
                trend_context_info += f"- 小周期方向偏好：{micro_bias}\n"
            if guidance:
                trend_context_info += f"- 执行指引：{guidance}\n"

        recent_context_info = ""
        current_equity = None
        if recent_context:
            # 提取最新的账户权益信息
            for ctx in reversed(recent_context[-10:]):
                equity_val = ctx.get('equity')
                if equity_val is not None and equity_val != '--':
                    try:
                        current_equity = float(equity_val)
                        break
                    except (ValueError, TypeError):
                        pass
            
            recent_context_info = "\n历史交易记录与经验学习（最近 10 条，请仔细分析并提取经验教训）：\n"
            for idx, ctx in enumerate(recent_context[-10:], 1):
                status = ctx.get('status', '--')
                action = ctx.get('action', '--')
                strategy = ctx.get('strategy', '--')
                symbol_ctx = ctx.get('symbol', '--')
                price = ctx.get('price', '--')
                change_pct = ctx.get('change_pct', '--')
                imbalance = ctx.get('orderbook_imbalance', '--')
                equity_val = ctx.get('equity', '--')
                timestamp = ctx.get('timestamp', '--')
                
                # 判断交易结果（成功/失败/观望）
                result_marker = ""
                if status == "success" or status == "filled":
                    result_marker = "✓"
                elif status == "failed" or status == "rejected":
                    result_marker = "✗"
                elif action == "flat":
                    result_marker = "○"
                
                recent_context_info += (
                    f"{idx}. [{result_marker}] {timestamp} {symbol_ctx} | "
                    f"策略: {strategy} | 动作: {action} | 状态: {status} | "
                    f"价格: {price} | 24h变动: {change_pct}% | "
                    f"订单簿不平衡: {imbalance}% | 账户权益: {equity_val} USDT\n"
                )
            recent_context_info += (
                "\n请从上述历史记录中学习：\n"
                "- 分析成功交易（✓）的共同特征：入场时机、止损止盈设置、做T策略、市场条件等\n"
                "- 分析失败交易（✗）的原因：入场时机不当、止损过紧/过松、做T频率过高/过低、市场条件判断错误、保证金不足等\n"
                "- 分析观望决策（○）的合理性：是否错过了机会，还是正确规避了风险\n"
                "- 总结做T的最佳实践：什么情况下做T效果好，什么情况下应该减少做T频率\n"
                "- 将学习到的经验应用到当前决策中，避免重复错误，复制成功模式\n"
            )
        
        # 添加币安所有历史交易记录（成交记录）
        binance_history_info = ""
        if binance_trade_history and len(binance_trade_history) > 0:
            # 只显示当前交易对的历史记录，并按时间倒序排列（最新的在前）
            symbol_trades = [t for t in binance_trade_history if t.get("symbol") == symbol]
            if symbol_trades:
                # 按时间倒序排列（最新的在前）
                symbol_trades.sort(key=lambda x: int(x.get("time", 0)), reverse=True)
                # 限制显示最近100条，避免提示词过长
                display_trades = symbol_trades[:100]
                
                binance_history_info = f"\n币安账户历史交易记录（{symbol}，共{len(symbol_trades)}条，显示最近100条）：\n"
                binance_history_info += "格式：[时间] 方向 | 价格 | 数量 | 手续费 | 是否买方 | 盈亏\n"
                
                for idx, trade in enumerate(display_trades, 1):
                    trade_time = int(trade.get("time", 0))
                    time_str = datetime.fromtimestamp(trade_time / 1000).strftime("%Y-%m-%d %H:%M:%S") if trade_time else "--"
                    side = trade.get("side", "--")  # BUY or SELL
                    price = trade.get("price", "--")
                    qty = trade.get("qty", "--")
                    commission = trade.get("commission", "--")
                    commission_asset = trade.get("commissionAsset", "")
                    is_buyer = "买方" if trade.get("buyer", False) else "卖方"
                    realized_pnl = trade.get("realizedPnl", "--")
                    
                    binance_history_info += (
                        f"{idx}. [{time_str}] {side} | "
                        f"价格: {price} | 数量: {qty} | "
                        f"手续费: {commission} {commission_asset} | "
                        f"{is_buyer} | "
                        f"盈亏: {realized_pnl} USDT\n"
                    )
                
                binance_history_info += (
                    "\n请从币安历史交易记录中深度分析：\n"
                    "- 统计历史交易的胜率、平均盈亏比、最大单笔盈利/亏损\n"
                    "- 分析哪些价格区间、时间段、市场条件下的交易成功率更高\n"
                    "- 分析哪些交易方向（做多/做空）在当前市场环境下表现更好\n"
                    "- 分析手续费对收益的影响，优化交易频率\n"
                    "- 识别重复出现的错误模式，避免再次犯错\n"
                    "- 总结最佳入场时机、持仓时间、止盈止损设置的经验\n"
                    "- 将历史交易经验与当前市场条件结合，做出更优决策\n"
                )
        
        # 添加积累的知识库信息
        knowledge_info = ""
        if knowledge_base:
            knowledge_info = "\n积累的交易知识与经验（重要，请参考这些经验做决策）：\n"
            
            successful_patterns = knowledge_base.get("successful_patterns", [])
            if successful_patterns:
                knowledge_info += "\n成功交易模式（请优先采用）：\n"
                for idx, pattern in enumerate(successful_patterns[-10:], 1):  # 只显示最近10条
                    knowledge_info += f"{idx}. {pattern}\n"
            
            failed_patterns = knowledge_base.get("failed_patterns", [])
            if failed_patterns:
                knowledge_info += "\n失败交易模式（请避免）：\n"
                for idx, pattern in enumerate(failed_patterns[-10:], 1):  # 只显示最近10条
                    knowledge_info += f"{idx}. {pattern}\n"
            
            best_practices = knowledge_base.get("best_practices", [])
            if best_practices:
                knowledge_info += "\n最佳实践：\n"
                for idx, practice in enumerate(best_practices[-10:], 1):  # 只显示最近10条
                    knowledge_info += f"{idx}. {practice}\n"
            
            market_conditions = knowledge_base.get("market_conditions", [])
            if market_conditions:
                knowledge_info += "\n市场条件与策略匹配：\n"
                for idx, condition in enumerate(market_conditions[-10:], 1):  # 只显示最近10条
                    cond_desc = condition.get("condition", "")
                    strategy = condition.get("strategy", "")
                    reason = condition.get("reason", "")
                    knowledge_info += f"{idx}. 条件：{cond_desc} → 策略：{strategy}（原因：{reason}）\n"
            
            timing_experience = knowledge_base.get("timing_experience", [])
            if timing_experience:
                knowledge_info += "\n交易时机经验：\n"
                for idx, timing in enumerate(timing_experience[-10:], 1):  # 只显示最近10条
                    knowledge_info += f"{idx}. {timing}\n"
            
            stop_loss_take_profit_experience = knowledge_base.get("stop_loss_take_profit_experience", [])
            if stop_loss_take_profit_experience:
                knowledge_info += "\n止损止盈经验：\n"
                for idx, experience in enumerate(stop_loss_take_profit_experience[-10:], 1):  # 只显示最近10条
                    knowledge_info += f"{idx}. {experience}\n"
            
            knowledge_info += "\n请将这些积累的知识与当前市场条件结合，做出更优的决策。\n"
        
        # 添加账户权益信息到提示
        account_info = ""
        if current_equity:
            account_info = f"\n当前账户权益：约 {current_equity:.2f} USDT。请注意：已有持仓会占用保证金，设置委托单时请确保所有订单的总保证金不超过可用资金，避免出现“保证金不足”错误。\n"

        position_state_info = ""
        if position_state:
            direction = "多单" if position_state.get("direction") == "long" else "空单"
            position_state_info = (
                "\n币安实时持仓（请结合此仓位做决策）：\n"
                f"- 仓位方向: {direction}\n"
                f"- 持仓数量: {position_state.get('quantity', 0)}\n"
                f"- 入场均价: {position_state.get('entry_price', 0)}\n"
                f"- 未实现收益率: {position_state.get('unrealized_pnl_pct', 0):.4f}%\n"
                "- 如果建议加仓/减仓/反手，请明确说明理由，系统不会再限制仓位，请自行控制风险\n"
            )
        else:
            position_state_info = (
                "\n币安实时持仓：当前该交易对没有持仓，你可以自由选择观望或开仓，仓位由你完全决定。\n"
            )

        prompt = "".join([
            str(base_prompt),
            str(mode_desc_map[base_mode]),
            str(multi_tf_info),
            str(trend_context_info),
            str(recent_context_info),
            str(binance_history_info),
            str(knowledge_info),
            str(account_info),
            str(position_state_info),
            str(holding_time_instruction),
            str(market_info),
            "\n通用要求：\n",
            "1. 明确给出做多(long)、做空(short)或观望(flat) 的决策\n",
            "2. 严格给出止损价和止盈价\n",
            "3. 合理给出建议仓位比例，确保整体风险可控，总体持仓不得超过合约账户资金的 50%\n",
            "4. 结合实时市场数据（订单簿不平衡度、买卖盘力量、成交活跃度等）进行综合判断\n",
            "5. 根据当前行情和策略模式，给出建议的持仓时间（分钟）\n",
            "6. 历史经验学习（重要）：仔细分析上述历史交易记录，提取成功/失败模式，并将经验应用到当前决策：\n",
            "   - 如果历史记录显示类似市场条件下做T效果好，当前可以适当增加做T频率和委托单数量\n",
            "   - 如果历史记录显示类似市场条件下频繁做T导致亏损，当前应该减少做T频率，更谨慎\n",
            "   - 如果历史记录显示止损设置过紧导致频繁被扫止损，当前可以适当放宽止损\n",
            "   - 如果历史记录显示止损设置过松导致大亏损，当前应该收紧止损\n",
            "   - 如果历史记录显示某些入场时机（如订单簿不平衡度、价格位置等）成功率较高，当前优先采用类似时机\n",
            "   - 如果历史记录显示某些入场时机成功率较低，当前应该避免类似时机\n",
            "   - 总结做T的最佳实践：在什么市场条件下做T效果好，什么条件下应该减少做T\n",
            "7. 结合 trend_context 中的宏观/小周期判断，宏观看多时优先多头并在小周期回调中低买高卖；宏观看空时优先空头并在小周期反弹中高卖低买。\n",
            "8. 做T与利润保护：目标是\"扩大利润、减少回撤、守护已有利润\"，具体点位、数量、回调比例全部由你根据实时行情/订单簿/成交量/支撑阻力/波动性以及历史经验自主决定，系统只保留订单数量上限约束，不再提供固定阈值或分级规则。\n",
            "9. 仓位管理原则（重要）：不要一味加仓，要主动减仓以降低成本：\n",
            "   - 当持仓盈利时，在合适位置（如达到第一目标位、遇到阻力位、订单簿显示卖压增大等）主动减仓部分仓位，锁定利润并降低持仓成本\n",
            "   - 即使没有盈利，在合适位置也要主动减仓：当价格接近阻力位/支撑位、订单簿压力增大、市场条件变化、持仓时间过长等情况下，即使当前未盈利，也应该主动减仓部分仓位，降低风险暴露和持仓成本\n",
            "   - 减仓策略：多单在价格上涨至阻力位或订单簿卖压增大时减仓，空单在价格下跌至支撑位或订单簿买压增大时减仓\n",
            "   - 避免频繁加仓：不要在每个小波动都加仓，应该等待更明确的信号或更好的价格位置再加仓\n",
            "   - 分批减仓：可以设置多个减仓点位（如30%、50%、70%），逐步降低仓位和成本\n",
            "   - 在 orders 中合理配置 limit_buy（多单减仓）或 limit_sell（空单减仓）订单，用于在合适位置主动减仓\n",
            "10. 资金约束：\n",
            "   - 仅交易 ETHUSDT。\n",
            "   - 重要：设置委托单时，请确保所有订单的总保证金不超过当前账户可用资金。如果已有持仓占用了保证金，请相应减少新订单的数量或减少委托单数量，避免出现\"保证金不足\"错误。\n\n",
            "请只返回 JSON，不要任何多余文字，不要使用 markdown 代码块。\n",
            "直接返回纯 JSON 对象，格式如下：\n",
            "JSON 字段：\n",
            "- action: 'long'|'short'|'flat' （本周期做多/做空/观望）\n",
            "- entry_price: 建议入场价格（如果 action 是 flat，可设为当前价格）\n",
            "- stop_loss: 止损价格\n",
            "- take_profit: 止盈价格（建议盈亏比不低于 1.2:1）\n",
            "- max_position_pct: 本标的目标最大仓位占合约账户资金比例（0-0.50，例如 0.1 表示使用 10% 资金），请确保所有标的合计不超过 0.50\n",
            "- holding_time_minutes: 建议持仓时间（分钟），根据当前行情和运行周期决定（超短模式建议 5-30 分钟，趋势模式建议 30-180 分钟）\n",
            "- risk_level: 当前策略模式的风险等级评分（1-10分，1分最低风险，10分最高风险）。保守型约2-3分，均衡型约5-6分，激进型约8-9分。\n",
            "- comment: 简短中文理由（说明为什么选择这个方向，或为什么观望，可提及订单簿、成交量等实时数据）\n",
            "- orders: 委托单配置数组，根据当前策略模式（保守型/均衡型/激进型）自行规划委托单结构：\n",
            "  * 必须包含：3个移动止盈（type=trailing_take_profit）和3个移动止损（type=trailing_stop_loss），每个周期都要设置这6个移动止盈止损单。\n",
            "  * 移动止盈止损的设置：\n",
            "    - 3个移动止盈：可以设置不同的激活价格（activation_price）和回调率（callback_rate），用于分批锁定利润。例如：第一个在价格达到第一目标位时激活，第二个在更高位置激活，第三个在最高位置激活。\n",
            "    - 3个移动止损：可以设置不同的激活价格和回调率，用于分批保护本金和利润。例如：第一个在价格接近成本价时激活（保护本金），第二个在价格达到一定盈利时激活（保护利润），第三个在价格达到更高盈利时激活（保护更多利润）。\n",
            "    - 每个移动止盈/止损的 quantity_pct 可以不同，你可以根据策略需要分配数量占比（总和不超过1.0）。\n",
            "  * 根据策略模式规划其余委托单（限价单等）：\n",
            "    - 保守型（conservative）：委托单数量较少，优先保护本金，止损较紧，做T频率低，分批止盈保守。\n",
            "    - 均衡型（balanced）：委托单数量中等，兼顾风险与收益，止损止盈适中，适度做T，分批止盈均衡。\n",
            "    - 激进型（aggressive）：委托单数量较多，追求更高收益，止损可放宽，积极做T，分批止盈激进。\n",
            "  * 可用的委托类型：limit_buy（买入限价）、limit_sell（卖出限价）、trailing_stop_loss（移动止损）、trailing_take_profit（移动止盈）等。\n",
            "  * 你自主决定委托类型/数量/点位/回调比例/数量占比，目标是\"扩大利润、减少回撤、保护利润\"，但必须符合当前策略模式的风险偏好。\n",
            "  * 重要：不要一味设置加仓单，要主动设置减仓单以降低成本。当持仓盈利时，在合适位置（阻力位、支撑位、订单簿压力位等）设置减仓单（多单用 limit_sell 减仓，空单用 limit_buy 减仓），逐步降低仓位和成本。\n",
            "  * 示例：\n",
            "    - trailing_stop_loss: {\"type\":\"trailing_stop_loss\", \"callback_rate\":..., \"activation_price\":..., \"quantity_pct\":1.0}\n",
            "    - limit_buy / limit_sell: {\"type\":\"limit_buy\", \"price\":..., \"quantity_pct\":..., \"purpose\":...}\n",
            "    - trailing_take_profit: {\"type\":\"trailing_take_profit\", \"callback_rate\":..., \"activation_price\":..., \"quantity_pct\":...}\n",
            "  * 你可以自由决定价格、回调比例、数量占比、目的说明。\n\n",
            f"策略模式: {base_mode}\n",
            f"运行周期: {mode}\n",
            f"交易标的: {symbol}\n",
            f"最近 K 线数据: {klines[-30:]}\n",
            "请综合分析 K 线趋势、支撑阻力、成交量、实时订单簿深度、买卖盘力量等，给出符合上述策略模式的交易建议。\n",
            "重要：请直接返回 JSON 对象，不要添加任何说明文字、markdown 代码块或其他格式。"
        ])

        # 如果没有配置 DeepSeek key，直接返回观望，避免卡死
        if not self.cfg.api_key:
            return {
                "action": "flat",
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "max_position_pct": 0.0,
                "comment": "未配置 DEEPSEEK_API_KEY，保持观望",
            }

        try:
            content = self._post_chat(prompt)
        except Exception as e:
            # 网络错误、超时等情况，一律返回观望，避免整个程序崩溃
            return {
                "action": "flat",
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "max_position_pct": 0.0,
                "comment": f"调用 DeepSeek 失败({type(e).__name__})，保持观望",
            }

        # 尝试解析 JSON，增强容错能力
        signal = None
        try:
            # 方法 1: 直接解析
            signal = json.loads(content)
        except json.JSONDecodeError:
            # 方法 2: 尝试提取 JSON（可能被 markdown 代码块包裹）
            try:
                # 移除 markdown 代码块标记
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    # 移除开头的 ```json 或 ```
                    lines = cleaned.split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned = '\n'.join(lines)
                elif cleaned.startswith("```json"):
                    cleaned = cleaned[7:].strip()
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3].strip()
                
                # 尝试找到 JSON 对象（可能前后有文字）
                json_start = cleaned.find('{')
                json_end = cleaned.rfind('}')
                if json_start >= 0 and json_end > json_start:
                    json_str = cleaned[json_start:json_end+1]
                    signal = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("未找到 JSON 对象", cleaned, 0)
            except (json.JSONDecodeError, ValueError) as e:
                # 方法 3: 记录原始内容用于调试，然后返回观望
                print(f"⚠️  DeepSeek 返回内容无法解析为 JSON:")
                print(f"   原始内容前 200 字符: {content[:200]}")
                print(f"   错误: {e}")
                signal = {
                    "action": "flat",
                    "entry_price": 0.0,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "max_position_pct": 0.0,
                    "comment": f"解析失败（已尝试多种方法），原始内容: {content[:50]}...",
                }
        
        # 验证解析后的信号是否包含必要字段
        if signal and not isinstance(signal, dict):
            signal = {
                "action": "flat",
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "max_position_pct": 0.0,
                "comment": "解析结果格式错误，观望",
            }
        
        return signal

    def get_learning_feedback(
        self,
        decisions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        将最近 N 次决策及收益指标发送给 DeepSeek，请求自学习反馈，返回总结、风险评估和推荐策略模式。
        """
        if not self.cfg.api_key:
            return {
                "summary": "未配置 DEEPSEEK_API_KEY，无法自学习。",
                "risk_assessment": "",
                "recommended_strategy": "",
                "action_items": [],
            }

        trimmed = [
            {
                "timestamp": d.get("timestamp"),
                "symbol": d.get("symbol"),
                "strategy": d.get("strategy"),
                "action": d.get("action"),
                "status": d.get("status"),
                "message": d.get("message"),
                "equity": d.get("equity"),
            }
            for d in decisions
        ]

        prompt = (
            "你是一个负责自学习与复盘的量化交易教练。\n"
            "以下是最近若干次（最多 10 次）自动交易决策及合约账户资金变化，请分析策略表现，"
            "总结表现亮点和问题，并给出下一阶段策略模式（在 conservative, balanced, aggressive, scalping, sniper, momentum_guard 中选择其一）以及 1-3 条具体优化建议。\n"
            "请用 JSON 格式回答，字段：\n"
            "- summary: 对最近一次窗口的整体表现进行一句话总结\n"
            "- risk_assessment: 当前风控状况的评价与提醒\n"
            "- recommended_strategy: 以上六个模式之一\n"
            "- action_items: 数组，列出 1-3 条具体改进建议\n\n"
            f"最近决策: {json.dumps(trimmed, ensure_ascii=False)}\n"
            f"绩效指标: {json.dumps(metrics, ensure_ascii=False)}\n"
            "请直接返回 JSON。"
        )

        try:
            content = self._post_chat(prompt)
            # 尝试解析 JSON，增强容错能力
            try:
                feedback = json.loads(content)
            except json.JSONDecodeError:
                # 尝试提取 JSON（可能被 markdown 代码块包裹）
                try:
                    cleaned = content.strip()
                    if cleaned.startswith("```"):
                        if cleaned.startswith("```json"):
                            cleaned = cleaned[7:].strip()
                        else:
                            cleaned = cleaned[3:].strip()
                        if cleaned.endswith("```"):
                            cleaned = cleaned[:-3].strip()
                    json_start = cleaned.find('{')
                    json_end = cleaned.rfind('}')
                    if json_start >= 0 and json_end > json_start:
                        json_str = cleaned[json_start:json_end+1]
                        feedback = json.loads(json_str)
                    else:
                        raise ValueError("无法提取 JSON 对象")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⚠️  自学习反馈 JSON 解析失败:")
                    print(f"   原始内容前 300 字符: {content[:300]}")
                    print(f"   错误: {e}")
                    return {
                        "summary": f"自学习反馈解析失败: {str(e)[:50]}",
                        "risk_assessment": "",
                        "recommended_strategy": "",
                        "action_items": [],
                    }
            return feedback
        except Exception as e:
            print(f"⚠️  自学习反馈调用失败: {type(e).__name__}: {e}")
            return {
                "summary": f"自学习反馈调用失败: {type(e).__name__}",
                "risk_assessment": "",
                "recommended_strategy": "",
                "action_items": [],
            }

    def extract_knowledge_from_feedback(
        self,
        feedback: Dict[str, Any],
        decisions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        binance_trade_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        从学习反馈和历史交易记录中提取知识，用于积累经验。
        返回结构化的知识对象。
        """
        if not self.cfg.api_key:
            return {}
        
        # 分析成功和失败的交易
        successful_decisions = [d for d in decisions if d.get("status") in ["ordered", "success", "filled"]]
        failed_decisions = [d for d in decisions if d.get("status") in ["failed", "rejected"]]
        
        # 构建提示词，让DeepSeek提取知识
        prompt = (
            "你是一个量化交易知识提取专家。请从以下学习反馈和历史交易记录中提取可复用的知识。\n\n"
            f"学习反馈总结: {feedback.get('summary', '')}\n"
            f"风险评估: {feedback.get('risk_assessment', '')}\n"
            f"改进建议: {json.dumps(feedback.get('action_items', []), ensure_ascii=False)}\n"
            f"绩效指标: {json.dumps(metrics, ensure_ascii=False)}\n\n"
            f"成功交易数量: {len(successful_decisions)}\n"
            f"失败交易数量: {len(failed_decisions)}\n"
        )
        
        if successful_decisions:
            prompt += f"成功交易示例: {json.dumps(successful_decisions[:3], ensure_ascii=False)}\n"
        
        if failed_decisions:
            prompt += f"失败交易示例: {json.dumps(failed_decisions[:3], ensure_ascii=False)}\n"
        
        if binance_trade_history:
            # 分析币安历史交易记录中的盈利和亏损交易
            profitable_trades = [t for t in binance_trade_history if float(t.get("realizedPnl", 0)) > 0]
            losing_trades = [t for t in binance_trade_history if float(t.get("realizedPnl", 0)) < 0]
            prompt += (
                f"\n币安历史交易统计:\n"
                f"- 盈利交易: {len(profitable_trades)} 笔\n"
                f"- 亏损交易: {len(losing_trades)} 笔\n"
            )
            if profitable_trades:
                prompt += f"盈利交易示例: {json.dumps(profitable_trades[:2], ensure_ascii=False)}\n"
            if losing_trades:
                prompt += f"亏损交易示例: {json.dumps(losing_trades[:2], ensure_ascii=False)}\n"
        
        prompt += (
            "\n请用 JSON 格式返回提取的知识，字段：\n"
            "- successful_patterns: 数组，列出成功交易的模式特征（如：入场时机、市场条件、策略选择等），每条不超过50字\n"
            "- failed_patterns: 数组，列出失败交易的模式特征（如：常见错误、应避免的情况等），每条不超过50字\n"
            "- best_practices: 数组，列出最佳实践建议，每条不超过50字\n"
            "- market_conditions: 数组，列出不同市场条件下适合的策略，格式：{\"condition\": \"描述\", \"strategy\": \"策略名\", \"reason\": \"原因\"}\n"
            "- timing_experience: 数组，列出交易时机的经验，每条不超过50字\n"
            "- stop_loss_take_profit_experience: 数组，列出止损止盈设置的经验，每条不超过50字\n"
            "请直接返回 JSON，不要包含其他文字。"
        )
        
        try:
            content = self._post_chat(prompt)
            # 尝试解析 JSON
            try:
                knowledge = json.loads(content)
            except json.JSONDecodeError:
                # 尝试提取 JSON（可能被 markdown 代码块包裹）
                try:
                    cleaned = content.strip()
                    if cleaned.startswith("```"):
                        if cleaned.startswith("```json"):
                            cleaned = cleaned[7:].strip()
                        else:
                            cleaned = cleaned[3:].strip()
                        if cleaned.endswith("```"):
                            cleaned = cleaned[:-3].strip()
                    json_start = cleaned.find('{')
                    json_end = cleaned.rfind('}')
                    if json_start >= 0 and json_end > json_start:
                        json_str = cleaned[json_start:json_end+1]
                        knowledge = json.loads(json_str)
                    else:
                        return {}
                except Exception:
                    return {}
            
            # 确保所有字段都存在
            return {
                "successful_patterns": knowledge.get("successful_patterns", []),
                "failed_patterns": knowledge.get("failed_patterns", []),
                "best_practices": knowledge.get("best_practices", []),
                "market_conditions": knowledge.get("market_conditions", []),
                "timing_experience": knowledge.get("timing_experience", []),
                "stop_loss_take_profit_experience": knowledge.get("stop_loss_take_profit_experience", []),
            }
        except Exception as e:
            print(f"⚠️  知识提取失败: {type(e).__name__}: {e}")
            return {}


