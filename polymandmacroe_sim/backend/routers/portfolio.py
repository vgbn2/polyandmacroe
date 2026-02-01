from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import UserAccount, PortfolioPosition, TradeHistory

router = APIRouter()

@router.get("/portfolio")
def get_portfolio(db: Session = Depends(get_db)):
    # Get User Balance
    user = db.query(UserAccount).first()
    if not user:
        user = UserAccount(balance=1000.0)
        db.add(user)
        db.commit()
    
    # Get Positions
    positions = db.query(PortfolioPosition).all()
    
    # Mock History for Chart (Placeholder)
    history = [1000 + i*10 for i in range(10)] 
    
    return {
        "balance": user.balance,
        "positions": positions,
        "history": history
    }
