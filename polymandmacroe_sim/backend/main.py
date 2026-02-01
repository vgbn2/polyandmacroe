import os
import sys
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Add project root (CODEPTIT) to path for imports
# main.py -> backend -> polym_sim -> polymarket -> CODEPTIT
# For Docker: working_dir is /app/polym_sim, so we add /app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.database import engine, Base, get_db
from backend import models
from backend.routers import portfolio, markets, trade, macro
from backend.config import settings

# Initialize Database
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

# Include Routers
app.include_router(portfolio.router, prefix="/api")
app.include_router(markets.router, prefix="/api")
app.include_router(trade.router, prefix="/api")
app.include_router(macro.router, prefix="/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (Frontend)
static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)
    
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "PolySimulator Backend Online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
