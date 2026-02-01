import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Unified Edge Terminal"
    PROJECT_VERSION: str = "1.0.0"
    
    # DATABASE
    # Uses the same DB as the ingested data
    # Path relative to where run_app.py is run (CODEPTIT/polymarket/polymandmacroe_sim)
    # We need to point to: CODEPTIT/macroe/macro_data.db
    # This path hack assumes we run from polymandmacroe_sim directory
    MACRO_DB_URL: str = os.getenv("MACRO_DB_URL", "sqlite:///../../macroe/macro_data.db")
    POLY_DB_URL: str = os.getenv("POLY_DB_URL", "sqlite:///./polysim.db")
    
    # SECURITY
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super_secret_key_change_me")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # POLYMARKET API
    POLY_API_KEY: str = os.getenv("POLY_API_KEY", "")
    POLY_PROXY_ADDRESS: str = os.getenv("POLY_PROXY_ADDRESS", "")

settings = Settings()
