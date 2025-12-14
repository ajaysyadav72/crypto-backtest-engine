"""
Configuration Management Module

Defines all dataclasses for type-safe configuration management.
Supports environment variables, YAML, and CLI overrides.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime
from enum import Enum


class OptimizationProfile(str, Enum):
    """Optimization profile selection"""
    FAST = "FAST"              # ±10%, 50 trials
    BALANCED = "BALANCED"      # ±15%, 150 trials (recommended)
    THOROUGH = "THOROUGH"      # ±25%, 300 trials


class AuthType(str, Enum):
    """Exchange authentication types"""
    HMAC = "hmac"
    OAUTH2 = "oauth2"
    TOKEN = "token"


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    base_url: str
    auth_type: AuthType
    rate_limit: int = 1200  # requests per minute
    timeout: int = 30       # seconds
    
    def __post_init__(self):
        if isinstance(self.auth_type, str):
            self.auth_type = AuthType(self.auth_type)


@dataclass
class UserInputConfig:
    """User input configuration from Stage 3"""
    mode: str  # "SINGLE" or "MULTI"
    exchanges: List[str]
    symbols: List[str]
    order_qty: List[float]
    backtest_start: datetime
    backtest_end: datetime
    timeframes: List[str]
    
    def validate(self) -> bool:
        """Validate user inputs"""
        if self.mode not in ["SINGLE", "MULTI"]:
            return False
        if self.backtest_end <= self.backtest_start:
            return False
        if any(qty <= 0 for qty in self.order_qty):
            return False
        return True


@dataclass
class StrategyConfig:
    """Strategy configuration from Stage 4"""
    # EMA Settings
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    
    # RSI Settings
    rsi_period: int = 14
    rsi_long_threshold: int = 35
    rsi_short_threshold: int = 65
    
    # Risk Management
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    trailing_sl_threshold: float = 0.02
    trailing_tp_threshold: float = 0.03
    
    # Hybrid Entry Engine
    enable_hybrid_tier1: bool = True
    enable_hybrid_tier2: bool = True
    enable_hybrid_tier3: bool = True
    enable_hybrid_tier4: bool = True
    ema_touch_tolerance: float = 0.0015
    hybrid_tier1_timeout: int = 5
    hybrid_tier2_timeout: int = 3
    hybrid_tier3_timeout: int = 4
    hybrid_tier4_timeout: int = 1
    hybrid_tier5_timeout: int = 0
    
    # Filter Toggles
    enable_trend_filter: bool = True
    enable_rsi_filter: bool = True
    enable_hybrid_engine: bool = True
    trailing_sl_enabled: bool = True
    trailing_tp_enabled: bool = True
    
    # Feature Toggles
    enable_optimization: bool = True
    enable_walk_forward: bool = True
    enable_monte_carlo: bool = True
    enable_dashboard: bool = True
    
    def validate(self) -> bool:
        """Validate strategy parameters"""
        if self.ema_fast >= self.ema_slow:
            return False
        if self.ema_slow >= self.ema_trend:
            return False
        if not (20 <= self.rsi_long_threshold <= 50):
            return False
        if not (50 <= self.rsi_short_threshold <= 80):
            return False
        if not (0.5 <= self.atr_sl_multiplier <= 5.0):
            return False
        if not (1.0 <= self.atr_tp_multiplier <= 5.0):
            return False
        return True


@dataclass
class OptimizationConfig:
    """Optimization configuration (Stage 7)"""
    profile: OptimizationProfile = OptimizationProfile.BALANCED
    optuna_sampler: str = "TPE"
    num_trials: int = 150
    multi_fidelity: bool = True
    fidelity_levels: Tuple[float, float, float] = (0.33, 0.66, 1.0)
    parallelization_workers: int = 4
    random_state: int = 42
    
    @property
    def grid_range_percentage(self) -> float:
        """Get grid range percentage based on profile"""
        ranges = {
            OptimizationProfile.FAST: 0.10,
            OptimizationProfile.BALANCED: 0.15,
            OptimizationProfile.THOROUGH: 0.25
        }
        return ranges.get(self.profile, 0.15)


@dataclass
class WalkForwardConfig:
    """Walk-forward configuration (Stage 8)"""
    is_length_days: int = 60
    oos_length_days: int = 20
    stride_days: int = 20
    min_trades_per_window: int = 5
    rebalance_interval: Optional[int] = None


@dataclass
class BacktestMetrics:
    """Backtest results metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    recovery_factor: float = 0.0
    profit_factor: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: List[dict] = field(default_factory=list)


@dataclass
class HardwareProfile:
    """Hardware profiling results (Stage 6)"""
    cpu_speed_score: float = 0.0  # ops/sec
    gpu_available: bool = False
    gpu_device: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_framework: Optional[str] = None
    timestamp: str = ""


# Supported exchanges configuration
EXCHANGES = {
    "binance": ExchangeConfig(
        name="Binance",
        base_url="https://api.binance.com",
        auth_type=AuthType.HMAC,
        rate_limit=1200
    ),
    "delta": ExchangeConfig(
        name="Delta Exchange India",
        base_url="https://api.india.delta.exchange",
        auth_type=AuthType.HMAC,
        rate_limit=6000
    ),
    "zerodha": ExchangeConfig(
        name="Zerodha (Kite)",
        base_url="https://api.kite.trade",
        auth_type=AuthType.OAUTH2,
        rate_limit=6000
    ),
    "dhan": ExchangeConfig(
        name="Dhan",
        base_url="https://api.dhan.co/v2",
        auth_type=AuthType.TOKEN,
        rate_limit=6000
    )
}
