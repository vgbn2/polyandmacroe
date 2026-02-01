
import os
import sys
import subprocess

# Windows UTF-8 fix
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Define the path to the main application
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "main.py")

# Ensure dependencies are installed
def check_dependencies():
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "sqlalchemy", "python-multipart"])

if __name__ == "__main__":
    check_dependencies()
    print(f"🚀 Starting PolySimulator at http://localhost:8000")
    print(f"📂 App Path: {APP_PATH}")
    
    # Run Uvicorn
    # format: backend.main:app
    # ensure we are in the polym_sim directory context
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])
