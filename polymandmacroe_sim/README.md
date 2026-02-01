# Unified Edge Terminal

A full-stack web application combining **Polymarket Edge Finder** and **Macro Economic Grading** into a unified trading intelligence platform.

## Features

- **Edge Detection** – Real-time arbitrage and +EV opportunities from Polymarket
-  **Macro Heatmap** – World economic health visualization with country grades
- 📊 **Portfolio Simulator** – Paper trading with Kelly sizing
- 📈 **Indicator Tracking** – Z-scores for economic indicators by country

## Tech Stack

| Layer      | Technology                     |
| ---------- | ------------------------------ |
| Frontend   | React 18 + Vite + Tailwind CSS |
| Backend    | FastAPI + SQLAlchemy           |
| Database   | SQLite                         |
| Deployment | Docker                         |

## Quick Start

### Local Development

```bash
# Install backend dependencies
pip install -r requirements.txt

# Run backend
python run_app.py

# In a separate terminal, run frontend
cd frontend
npm install
npm run dev
```

### Docker Deployment

```bash
docker-compose up --build
```

Access at: `http://localhost:8000`

## Project Structure

```
polymandmacroe_sim/
├── backend/
│   ├── main.py           # FastAPI app
│   ├── config.py         # Settings & env vars
│   ├── database.py       # DB connection
│   ├── models.py         # ORM models
│   └── routers/
│       ├── markets.py    # Edge detection API
│       ├── macro.py      # Macro data API
│       ├── portfolio.py  # Portfolio API
│       └── trade.py      # Trading API
├── lib/                  # Self-contained logic modules
│   ├── crypto_tracker.py # Edge detection logic
│   └── macro_models.py   # Macro database models
├── frontend/
│   └── src/
│       ├── pages/        # Home, Macro, Polymarket
│       └── components/   # Layout, shared UI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
MACRO_DB_URL=sqlite:///path/to/macro_data.db
POLY_DB_URL=sqlite:///path/to/polysim.db
SECRET_KEY=your_secret_key
```

## Customizing Logic

| Feature          | File                                                          |
| ---------------- | ------------------------------------------------------------- |
| Edge calculation | `lib/crypto_tracker.py` → `MarketParser.parse_markets()` |
| Macro scoring    | `lib/macro_models.py` → `CompositeScore` model           |
| API endpoints    | `backend/routers/*.py`                                      |

## API Endpoints

| Endpoint                             | Description             |
| ------------------------------------ | ----------------------- |
| `GET /api/health`                  | Health check            |
| `GET /api/markets`                 | Live edge opportunities |
| `GET /api/macro/heatmap`           | Country grades          |
| `GET /api/macro/indicators/{code}` | Country indicators      |
| `GET /api/portfolio`               | Portfolio status        |
| `POST /api/trade`                  | Execute trade           |

## Deployment

This project is self-contained and ready for deployment on:

- **Railway** – Connect repo, auto-deploys
- **Render** – Dockerfile detection
- **Fly.io** – `fly launch`
- **Any Docker host** – `docker-compose up`

## License

MIT
