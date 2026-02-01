from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .database import Base

class PortfolioPosition(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    market_slug = Column(String, index=True) # e.g. "btc-hit-90k-feb1"
    market_name = Column(String)
    outcome = Column(String) # YES / NO
    shares = Column(Float, default=0.0)
    avg_price = Column(Float, default=0.0) # Cents (0-100)
    current_value = Column(Float, default=0.0) # For caching display value

class TradeHistory(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    market_slug = Column(String)
    market_name = Column(String)
    outcome = Column(String)
    side = Column(String) # BUY / SELL
    price = Column(Float)
    shares = Column(Float)
    total_cost = Column(Float)
    
class UserAccount(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    balance = Column(Float, default=1000.0) # Starting simulated cash
