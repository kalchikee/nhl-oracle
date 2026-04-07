# NHL Oracle v4.0

Machine learning NHL prediction engine. Runs fully automated on GitHub Actions — no computer required.

## What it does
- **6:00 AM CST daily** → Morning Discord briefing with game predictions, recommended bets, and season record
- **11:30 PM CST daily** → Evening Discord recap comparing predictions vs actual results
- **Every Monday** → Retrains the model on 5 years of historical data
- **Automatically stops** sending notifications when the NHL season ends (Stanley Cup awarded)
- **Automatically resumes** when the new season starts in October

## Setup (one-time, ~10 minutes)

### 1. Create a GitHub repo
Push this entire `nhl-oracle/` folder to a new GitHub repository.

```bash
cd nhl-oracle
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/nhl-oracle.git
git push -u origin main
```

### 2. Add GitHub Secrets
Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value |
|---|---|
| `DISCORD_WEBHOOK_URL` | Your Discord webhook URL |
| `ODDS_API_KEY` | *(Optional)* Free key from [the-odds-api.com](https://the-odds-api.com) — enables market edge detection |

### 3. Enable GitHub Actions
Go to your repo → **Actions tab** → Click "I understand my workflows, go ahead and enable them"

### 4. Trigger initial model training
Go to **Actions → Weekly Model Retrain → Run workflow**

This downloads 5 years of historical NHL data and trains the logistic regression model. Takes ~20-30 minutes on first run, then caches data for fast subsequent retrains.

### 5. That's it!
GitHub Actions will automatically run every morning at 6am CST and every evening at 11:30pm CST. You can also trigger any workflow manually from the Actions tab.

---

## Architecture

```
Data Sources
├── NHL API (api-web.nhle.com) — schedule, scores, standings, team stats [free]
├── MoneyPuck (moneypuck.com) — xG, GSAx, Corsi, shot quality [free CSVs]
└── The-Odds-API (optional) — Vegas moneyline odds [500 req/month free]

ML Pipeline
├── Feature Engineering (18 features per game)
│   ├── Elo rating difference
│   ├── Points percentage, goals per game, special teams
│   ├── Last-10 form, regulation win %
│   ├── xGF%, Corsi%, PDO (from MoneyPuck)
│   ├── Goalie GSAx difference
│   └── Rest days, back-to-back flags
├── Monte Carlo Simulation (10,000 Poisson iterations)
│   ├── Regulation win probability
│   ├── OT/SO win probability
│   ├── Projected score, total goals
│   └── Shutout probability
└── Logistic Regression + Platt Scaling (calibrated probabilities)

Confidence Tiers
├── 🔥 EXTREME CONVICTION — 68%+ calibrated probability
├── ⭐ HIGH CONVICTION — 63-68%
├── ✅ STRONG — 60-63%
├── 📊 LEAN — 55-60%
└── 🪙 COIN FLIP — 50-55%

Recommend Bet when: model_edge >= 5% AND pick_prob >= 58%
```

## Model Performance Targets
| Filter | Expected Accuracy |
|---|---|
| All games | 59–62% |
| 63%+ picks | 66–70% |
| 63%+ AND 6pt edge vs Vegas | 70–74% |

## Files
```
src/
├── nhl_api.py          — NHL API wrapper
├── moneypuck.py        — MoneyPuck data downloader
├── elo_system.py       — Elo rating system
├── features.py         — Feature engineering (used for both training + prediction)
├── monte_carlo.py      — 10,000-game Poisson simulation
├── train_model.py      — 5-year training pipeline
├── predictor.py        — Live prediction engine
├── discord_notifier.py — Discord message formatting + sending
├── morning_run.py      — Morning briefing entry point
└── evening_run.py      — Evening recap entry point

data/
├── elo_ratings.json         — Persisted Elo ratings (updated nightly)
├── prediction_history.json  — All predictions + results + season record
└── season_state.json        — Season active/inactive tracking

models/
├── model.pkl       — Trained calibrated logistic regression
├── scaler.pkl      — Feature StandardScaler
└── metadata.json   — Training metadata + CV results + feature importance
```
