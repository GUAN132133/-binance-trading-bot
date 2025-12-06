import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 打包后，.env 文件应该在 exe 同目录
# PyInstaller 打包后，sys.executable 指向 exe 文件路径
if getattr(sys, 'frozen', False):
    # 打包后的情况：exe 文件所在目录
    exe_dir = Path(sys.executable).parent
    env_path = exe_dir / ".env"
else:
    # 开发环境：项目根目录
    env_path = Path(__file__).parent.parent / ".env"

# 尝试加载 .env 文件
if env_path.exists():
    load_dotenv(env_path)
else:
    # 如果找不到，尝试默认位置
    load_dotenv()


@dataclass
class BinanceConfig:
    api_key: str
    api_secret: str
    use_testnet: bool
    symbol: str
    interval: str
    base_asset: str
    max_symbols: int


@dataclass
class RiskConfig:
    max_drawdown: float = 0.30  # 最大允许回撤比例（相对于启动时合约账户资金）
    max_open_symbols: int = 4   # 同时持仓的最大币种数量


@dataclass
class DeepSeekConfig:
    api_key: str
    api_base: str
    model: str = "deepseek-trader"


def get_binance_config() -> BinanceConfig:
    return BinanceConfig(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
        use_testnet=os.getenv("BINANCE_USE_TESTNET", "true").lower() == "true",
        symbol=os.getenv("SYMBOL", "BTCUSDT"),
        interval=os.getenv("INTERVAL", "1m"),
        base_asset=os.getenv("BASE_ASSET", "USDT"),
        max_symbols=int(os.getenv("MAX_SYMBOLS", "5")),
    )


def get_risk_config() -> RiskConfig:
    return RiskConfig(
        max_drawdown=float(os.getenv("MAX_DRAWDOWN", "0.30")),
        max_open_symbols=int(os.getenv("MAX_OPEN_SYMBOLS", "4")),
    )


def get_deepseek_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    )


