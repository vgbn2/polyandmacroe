from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any

# Create a dedicated connection to the Macro DB
from backend.config import settings

# Manual setup for Macro DB Connection since it's external to this app's main DB
macro_engine = create_engine(settings.MACRO_DB_URL, connect_args={"check_same_thread": False})
MacroSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=macro_engine)

def get_macro_db():
    db = MacroSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Import Models from local lib
from lib.macro_models import Country, CompositeScore, Observation, Indicator

router = APIRouter(
    prefix="/macro",
    tags=["macro"],
    responses={404: {"description": "Not found"}},
)

@router.get("/heatmap")
def get_economic_heatmap(db: Session = Depends(get_macro_db)):
    """
    Returns the latest Composite Score + Grade for all countries.
    Used for the World Map visualization.
    """
    # Get latest date
    subquery = db.query(
        CompositeScore.country_id, 
        func.max(CompositeScore.date).label('max_date')
    ).group_by(CompositeScore.country_id).subquery()
    
    results = db.query(CompositeScore, Country).join(
        subquery, 
        (CompositeScore.country_id == subquery.c.country_id) & 
        (CompositeScore.date == subquery.c.max_date)
    ).join(Country).all()
    
    data = []
    for score, country in results:
        data.append({
            "code": country.code,
            "name": country.name,
            "grade": score.health_grade, # A, B, C...
            "score": score.bias_score,
            "date": score.date.isoformat()
        })
    return data

@router.get("/indicators/{country_code}")
def get_country_indicators(country_code: str, db: Session = Depends(get_macro_db)):
    """
    Returns Z-Scores for all indicators for a specific country (History).
    """
    country = db.query(Country).filter(Country.code == country_code).first()
    if not country:
        raise HTTPException(status_code=404, message="Country not found")
        
    # Get last 12 months of observations
    # For now, just Limit 100
    obs = db.query(Observation, Indicator).join(Indicator).filter(
        Observation.country_id == country.id
    ).order_by(desc(Observation.date)).limit(100).all()
    
    data = []
    for o, i in obs:
        data.append({
            "indicator": i.name,
            "code": i.code,
            "value": o.value,
            "z_score": o.z_score_60m,
            "date": o.date.isoformat()
        })
    return data
