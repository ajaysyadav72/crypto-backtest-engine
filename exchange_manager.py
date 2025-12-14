"""
Exchange Manager Module (Stage 1-2)

Centralizes credential management and API integration for all supported exchanges.
Provides unified interface for data fetching and authentication.
"""

import os
import json
import time
import hashlib
import hmac
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
from functools import wraps
import logging
from logging.handlers import RotatingFileHandler
from config import ExchangeConfig, EXCHANGES, AuthType


class BacktestError(Exception):
    """Base exception for backtest engine"""
    pass


class CredentialError(BacktestError):
    """Credential validation error"""
    pass


class APIError(BacktestError):
    """API call error"""
    pass


class RateLimitError(BacktestError):
    """Rate limit exceeded error"""
    pass


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Setup rotating logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    os.makedirs('logs', exist_ok=True)
    
    # File handler with rotation
    fh = RotatingFileHandler(
        f'logs/{log_file}',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, RateLimitError) as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt+1} failed. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


logger = setup_logger('exchange_manager', 'exchange_manager.log')


class ExchangeManager:
    """Singleton manager for exchange credentials and API operations"""
    
    _instance = None
    _credentials = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.exchanges = EXCHANGES
        self.session = requests.Session()
    
    def load_credentials_from_env(self) -> Dict[str, Dict[str, str]]:
        """Load credentials from environment variables"""
        logger.info("Loading credentials from environment variables")
        
        credentials = {}
        for exchange_name in self.exchanges:
            exchange_upper = exchange_name.upper()
            
            if exchange_name in ["binance", "delta"]:
                api_key = os.getenv(f"{exchange_upper}_API_KEY")
                api_secret = os.getenv(f"{exchange_upper}_API_SECRET")
                if api_key and api_secret:
                    credentials[exchange_name] = {
                        "api_key": api_key,
                        "api_secret": api_secret
                    }
            elif exchange_name == "zerodha":
                api_key = os.getenv(f"{exchange_upper}_API_KEY")
                access_token = os.getenv(f"{exchange_upper}_ACCESS_TOKEN")
                if api_key and access_token:
                    credentials[exchange_name] = {
                        "api_key": api_key,
                        "access_token": access_token
                    }
            elif exchange_name == "dhan":
                access_token = os.getenv(f"{exchange_upper}_ACCESS_TOKEN")
                client_id = os.getenv(f"{exchange_upper}_CLIENT_ID")
                if access_token and client_id:
                    credentials[exchange_name] = {
                        "access_token": access_token,
                        "client_id": client_id
                    }
        
        self._credentials = credentials
        logger.info(f"Loaded credentials for exchanges: {list(credentials.keys())}")
        return credentials
    
    def set_credentials(self, exchange: str, **creds):
        """Set credentials for exchange"""
        if exchange not in self.exchanges:
            raise CredentialError(f"Exchange {exchange} not supported")
        
        self._credentials[exchange] = creds
        logger.info(f"Credentials set for {exchange}")
    
    def get_credentials(self, exchange: str) -> Dict[str, str]:
        """Get credentials for exchange"""
        if exchange not in self._credentials:
            raise CredentialError(f"No credentials found for {exchange}")
        return self._credentials[exchange]
    
    def validate_credentials(self, exchange: str) -> bool:
        """Validate credential presence"""
        try:
            creds = self.get_credentials(exchange)
            
            if exchange in ["binance", "delta"]:
                return "api_key" in creds and "api_secret" in creds
            elif exchange == "zerodha":
                return "api_key" in creds and "access_token" in creds
            elif exchange == "dhan":
                return "access_token" in creds and "client_id" in creds
            
            return False
        except CredentialError:
            return False
    
    def build_signed_headers(self, exchange: str, method: str, path: str,
                            params: Optional[Dict] = None,
                            body: Optional[str] = None) -> Dict[str, str]:
        """Build signed headers for exchange API call"""
        creds = self.get_credentials(exchange)
        headers = {"Content-Type": "application/json"}
        
        if exchange in ["binance", "delta"]:
            # HMAC-SHA256 signature
            timestamp = int(time.time() * 1000)
            params = params or {}
            params["timestamp"] = timestamp
            
            query_string = "&".join(
                f"{k}={v}" for k, v in sorted(params.items())
            )
            
            signature = hmac.new(
                creds["api_secret"].encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            
            params["signature"] = signature
            headers["X-MBX-APIKEY"] = creds["api_key"]
        
        elif exchange == "zerodha":
            headers["X-Kite-Version"] = "3"
            headers["Authorization"] = f"token {creds['api_key']}:{creds['access_token']}"
        
        elif exchange == "dhan":
            headers["Authorization"] = f"Bearer {creds['access_token']}"
        
        return headers
    
    @retry_with_backoff(max_retries=3)
    def make_api_call(self, exchange: str, method: str, path: str,
                     params: Optional[Dict] = None,
                     body: Optional[str] = None,
                     timeout: int = 30) -> Dict[str, Any]:
        """Make API call with retry logic"""
        exchange_config = self.exchanges[exchange]
        url = exchange_config.base_url + path
        
        headers = self.build_signed_headers(
            exchange, method, path, params, body
        )
        
        try:
            if method.upper() == "GET":
                response = self.session.get(
                    url, params=params, headers=headers, timeout=timeout
                )
            elif method.upper() == "POST":
                response = self.session.post(
                    url, data=body, headers=headers, timeout=timeout
                )
            else:
                raise APIError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            raise TimeoutError(f"API timeout for {exchange}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Connection error for {exchange}")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded for {exchange}")
            raise APIError(f"HTTP error {response.status_code}: {str(e)}")
    
    @retry_with_backoff(max_retries=3)
    def fetch_ohlc(self, exchange: str, symbol: str, timeframe: str,
                   start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch OHLC candles with pagination"""
        logger.info(
            f"Fetching {symbol} {timeframe} candles from "
            f"{start_date.date()} to {end_date.date()}"
        )
        
        all_candles = []
        current_date = start_date
        
        # Determine timeframe in minutes
        tf_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "1h": 60,
            "4h": 240, "1d": 1440, "1w": 10080
        }
        minutes_per_candle = tf_minutes.get(timeframe, 60)
        
        # For demo: return mock data
        # In production, fetch from actual exchange API
        num_candles = int((end_date - start_date).total_seconds() / 
                         (minutes_per_candle * 60))
        
        logger.debug(f"Generating {num_candles} mock candles")
        
        # Generate mock OHLC data (in production, fetch from API)
        import random
        import numpy as np
        
        current_price = 45000.0
        for i in range(min(num_candles, 10000)):  # Limit to 10k candles
            ts = current_date + timedelta(minutes=minutes_per_candle * i)
            
            change = random.uniform(-0.02, 0.02)
            open_price = current_price
            close_price = open_price * (1 + change)
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.01)
            low_price = min(open_price, close_price) * random.uniform(0.99, 1.0)
            volume = random.uniform(100, 10000)
            
            all_candles.append({
                "timestamp": ts.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2)
            })
            
            current_price = close_price
        
        logger.info(f"Fetched {len(all_candles)} candles for {symbol}")
        return all_candles
    
    def validate_connectivity(self) -> Dict[str, bool]:
        """Test API connectivity for all exchanges"""
        logger.info("Validating exchange connectivity")
        
        results = {}
        for exchange_name in self.exchanges:
            try:
                if not self.validate_credentials(exchange_name):
                    results[exchange_name] = False
                    continue
                
                # Make minimal test call
                if exchange_name == "binance":
                    self.make_api_call(
                        exchange_name,
                        "GET",
                        "/api/v3/time"
                    )
                
                results[exchange_name] = True
                logger.info(f"{exchange_name}: ✓ READY")
            
            except Exception as e:
                results[exchange_name] = False
                logger.warning(f"{exchange_name}: ✗ FAILED - {str(e)}")
        
        return results
