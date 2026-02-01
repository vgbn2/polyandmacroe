"""
Crypto Price Tracker - Edge Detection for Polymarket
Self-contained version for polymandmacroe_sim
"""

import os
import sys
import time
import json
import requests
import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# ==========================================
# ⚙️ CONFIG
# ==========================================
class Config:
    API_BASE_URL = "https://gamma-api.polymarket.com/events"
    API_HEADERS = {'User-Agent': 'Mozilla/5.0'}
    REFRESH_SECONDS = 600
    BANKROLL = 1000.0
    
    # Price Data Sources
    COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"
    BINANCE_API = "https://api.binance.com/api/v3/ticker/price"
    
    # Z-Transform Parameters
    STABILITY_WINDOW = 24  # hours of price history
    POLE_THRESHOLD = 0.95  # |z| > this signals reversal risk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CryptoTracker")


# ==========================================
# 💰 CLUMPED ARBITRAGE CALCULATOR
# ==========================================
class ClumpedArbitrage:
    """Calculates optimal share sizing for payout equalization across buckets."""
    
    @staticmethod
    def calculate_dutch_sizing(buckets: List[Dict], total_bet: float) -> List[Dict]:
        """Dutch betting: Size positions inversely proportional to ask price."""
        if not buckets:
            return []
        
        inverse_sum = sum(1.0 / b['ask'] for b in buckets if b['ask'] > 0)
        
        if inverse_sum == 0:
            return buckets
        
        result = []
        for b in buckets:
            if b['ask'] > 0:
                weight = (1.0 / b['ask']) / inverse_sum
                allocation = total_bet * weight
                shares = allocation / (b['ask'] / 100.0)
                
                result.append({
                    **b,
                    'allocation': allocation,
                    'shares': shares,
                    'payout': shares * 1.0
                })
        
        return result
    
    @staticmethod
    def calculate_roi(buckets: List[Dict], total_cost: float) -> float:
        """ROI for Dutch bet on clumped buckets."""
        if not buckets or total_cost == 0:
            return 0
        
        if buckets[0].get('allocation') and buckets[0]['ask'] > 0:
            expected_payout = buckets[0]['allocation'] / (buckets[0]['ask'] / 100.0)
        else:
            avg_price = sum(b['ask'] for b in buckets) / len(buckets)
            expected_payout = total_cost / (avg_price / 100.0)
        
        return (expected_payout - total_cost) / total_cost * 100


# ==========================================
# 🌐 API CLIENT
# ==========================================
class CryptoMarketAPI:
    BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
    
    @staticmethod
    def get_btc_price() -> Optional[float]:
        """Fetch current BTC price from Binance."""
        try:
            resp = requests.get(f"{Config.BINANCE_API}?symbol=BTCUSDT", timeout=5)
            data = resp.json()
            return float(data['price'])
        except Exception as e:
            logger.warning(f"Binance API error: {e}")
            
        try:
            resp = requests.get(f"{Config.COINGECKO_API}?ids=bitcoin&vs_currencies=usd", timeout=5)
            data = resp.json()
            return data['bitcoin']['usd']
        except Exception as e:
            logger.error(f"All price APIs failed: {e}")
            return None
    
    @staticmethod
    def get_coin_price(coin_id: str) -> Optional[float]:
        """Fetch price for any coin from CoinGecko."""
        try:
            resp = requests.get(f"{Config.COINGECKO_API}?ids={coin_id}&vs_currencies=usd", timeout=5)
            data = resp.json()
            return data.get(coin_id, {}).get('usd')
        except Exception as e:
            logger.warning(f"Price fetch failed for {coin_id}: {e}")
            return None
    
    @staticmethod
    def get_crypto_markets(coins: List[str] = None) -> List[Dict]:
        """Fetch Polymarket crypto price prediction markets."""
        if coins is None:
            coins = ["bitcoin", "ethereum", "solana", "xrp"]
        
        events = []
        seen_ids = set()
        
        import datetime as dt
        current_month = dt.datetime.now().strftime("%B").lower()
        
        for coin in coins:
            coin_slugs = [
                f"what-price-will-{coin}-hit-january-26-1",
                f"what-price-will-{coin}-hit-february",
                f"what-price-will-{coin}-hit-in-february",
                f"{coin}-price-february",
            ]
            
            for slug in coin_slugs:
                try:
                    resp = requests.get(f"{Config.API_BASE_URL}?slug={slug}",
                                       headers=Config.API_HEADERS, timeout=5)
                    data = resp.json()
                    if data and isinstance(data, list):
                        for event in data:
                            if event.get('closed') is False and event['id'] not in seen_ids:
                                events.append(event)
                                seen_ids.add(event['id'])
                except:
                    pass
        
        # Also search by query
        for coin in coins:
            queries = [f"What price will {coin} hit", f"{coin} above", f"{coin} price"]
            for query in queries:
                try:
                    params = {"limit": 20, "closed": "false", "q": query}
                    resp = requests.get(Config.API_BASE_URL, params=params,
                                       headers=Config.API_HEADERS, timeout=5)
                    data = resp.json()
                    
                    for event in data:
                        title = event.get('title', '').lower()
                        if coin in title and ('price' in title or 'above' in title or 'hit' in title):
                            if event.get('closed') is False and event['id'] not in seen_ids:
                                events.append(event)
                                seen_ids.add(event['id'])
                except:
                    pass
        
        return events


# ==========================================
# 🎲 BETTING STRATEGY
# ==========================================
class BettingStrategy:
    @staticmethod
    def calculate_kelly(prob_percent: float, price_cents: float, bankroll: float) -> Tuple[float, float, str]:
        """Calculates Quarter Kelly bet size."""
        if prob_percent <= 0 or price_cents <= 0 or price_cents >= 100:
            return 0.0, 0.0, "-"
            
        p = prob_percent / 100.0
        q = 1.0 - p
        b = (100.0 / price_cents) - 1.0
        
        if b <= 0: return 0.0, 0.0, "NegOdds"
        
        f_star = (b * p - q) / b
        
        if f_star <= 0: return 0.0, 0.0, "NegEV"
        
        f = f_star * 0.25
        f = min(f, 0.10)
        
        amt = bankroll * f
        return f, amt, "OK"


# ==========================================
# 📊 MARKET PARSER
# ==========================================
class MarketParser:
    @staticmethod
    def parse_markets(markets: List, current_price: float = None) -> List[Dict]:
        """Parse Polymarket bucket data and calculate edge."""
        buckets = []
        
        for m in markets:
            try:
                name = m.get('groupItemTitle', 'Unknown')
                prices = json.loads(m.get('outcomePrices', '["0", "0"]'))
                ask = float(prices[0]) * 100
                
                low, high = 0, float('inf')
                is_target_market = False
                is_below = False
                
                try:
                    if '↓' in name or '<' in name:
                        is_below = True
                        
                    clean = name.replace('$', '').replace(',', '').replace('k', '000')
                    clean = clean.replace('↑', '').replace('↓', '').replace('>', '').replace('<', '').strip()
                    
                    if '-' in clean and len(clean.split('-')) == 2:
                        parts = clean.split('-')
                        low = float(parts[0].strip())
                        high = float(parts[1].strip())
                    elif clean.replace('.','').isdigit():
                        val = float(clean)
                        low = val
                        high = float('inf')
                        is_target_market = True
                except:
                    pass
                
                contains_current = False
                is_resolved = False
                
                if is_target_market:
                    if is_below:
                         if current_price and current_price <= low: is_resolved = True
                    else:
                         if current_price and current_price >= low: is_resolved = True
                    
                    if current_price:
                        distance_pct = abs(current_price - low) / current_price * 100
                        contains_current = distance_pct < 2.0
                else:
                    contains_current = current_price and low <= current_price <= high
                
                market_prob = ask
                
                if is_resolved:
                    model_prob = 0.0
                elif current_price and low > 0:
                    distance_pct = (low - current_price) / current_price * 100
                    weekly_vol = 5.0
                    
                    import math
                    z_score = abs(distance_pct) / weekly_vol
                    model_prob = max(1, min(95, 50 * math.exp(-0.3 * z_score)))
                else:
                    model_prob = market_prob
                
                edge = model_prob - market_prob
                
                buckets.append({
                    'name': name,
                    'low': low,
                    'high': high,
                    'ask': ask,
                    'is_resolved': is_resolved,
                    'contains_current': contains_current,
                    'prob': market_prob,
                    'model_prob': model_prob,
                    'edge': edge
                })
            except Exception:
                pass
                
        return buckets


# Backwards compatibility aliases
BTCMarketAPI = CryptoMarketAPI
Dashboard = type('Dashboard', (), {'_parse_markets': staticmethod(MarketParser.parse_markets)})
