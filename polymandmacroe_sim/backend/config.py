import os
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directory for the app (polymandmacroe_sim folder)
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    PROJECT_NAME: str = "Unified Edge Terminal"
    PROJECT_VERSION: str = "1.0.0"
    
    # DATABASE
    # In Docker: use /app/data paths
    # Locally: use relative paths from polymandmacroe_sim
    MACRO_DB_URL: str = os.getenv("MACRO_DB_URL", f"sqlite:///{BASE_DIR.parent / 'macroe' / 'macro_data.db'}")
    POLY_DB_URL: str = os.getenv("POLY_DB_URL", f"sqlite:///{BASE_DIR / 'polysim.db'}")
    
    # SECURITY
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super_secret_key_change_me")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # POLYMARKET API
    POLY_API_KEY: str = os.getenv("POLY_API_KEY", "")
    POLY_PROXY_ADDRESS: str = os.getenv("POLY_PROXY_ADDRESS", "")

settings = Settings()
