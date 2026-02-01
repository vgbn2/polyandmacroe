# Self-contained library modules for polymandmacroe_sim
from lib.macro_models import Country, Indicator, Observation, CompositeScore, GradingThreshold, Base
from lib.crypto_tracker import CryptoMarketAPI, BettingStrategy, MarketParser, ClumpedArbitrage, Config as CryptoConfig
from lib.elonmusktweet import ElonTracker, TweetAnalyzer, PolymarketAPI as ElonPolymarketAPI, Config as ElonConfig, Dashboard as ElonDashboard

# Backwards compatibility
BTCMarketAPI = CryptoMarketAPI
