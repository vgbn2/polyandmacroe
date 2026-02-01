from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import UserAccount, PortfolioPosition, TradeHistory

router = APIRouter()

class TradeRequest(BaseModel):
    market_slug: str
    outcome: str
    side: str # BUY / SELL
    price: float # Cents
    shares: float

@router.post("/trade")
def execute_trade(trade: TradeRequest, db: Session = Depends(get_db)):
    user = db.query(UserAccount).first()
    if not user:
        user = UserAccount(balance=1000.0)
        db.add(user)
    
    cost = (trade.price / 100.0) * trade.shares
    
    if trade.side == "BUY":
        if user.balance < cost:
            return {"success": False, "message": "Insufficient Balance"}
        
        user.balance -= cost
        
        # Update Position
        pos = db.query(PortfolioPosition).filter_by(
            market_slug=trade.market_slug, 
            outcome=trade.outcome
        ).first()
        
        if pos:
            # Average Price calculation
            total_shares = pos.shares + trade.shares
            total_cost = (pos.shares * (pos.avg_price/100)) + cost
            pos.avg_price = (total_cost / total_shares) * 100
            pos.shares = total_shares
        else:
            pos = PortfolioPosition(
                market_slug=trade.market_slug,
                market_name=trade.market_slug, # Simplify
                outcome=trade.outcome,
                shares=trade.shares,
                avg_price=trade.price,
                current_value=trade.price # Init
            )
            db.add(pos)
            
    elif trade.side == "SELL":
        # Check position
        pos = db.query(PortfolioPosition).filter_by(
            market_slug=trade.market_slug, 
            outcome=trade.outcome
        ).first()
        
        if not pos or pos.shares < trade.shares:
            return {"success": False, "message": "Not enough shares"}
            
        proceeds = (trade.price / 100.0) * trade.shares
        user.balance += proceeds
        pos.shares -= trade.shares
        
        if pos.shares < 0.01:
            db.delete(pos)
            
    # Record Trade
    history = TradeHistory(
        market_slug=trade.market_slug,
        market_name=trade.market_slug,
        outcome=trade.outcome,
        side=trade.side,
        price=trade.price,
        shares=trade.shares,
        total_cost=cost
    )
    db.add(history)
    db.commit()
    
    return {"success": True, "balance": user.balance}
