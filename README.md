# GenoTek Autonomous Hiring Agent

A production-ready autonomous hiring pipeline covering Components 2–6 of the GenoTek challenge.

---

## Architecture

```
CSV / Internshala Export
        │
        ▼
┌───────────────┐        ┌──────────────────┐
│  scorer.py    │──────▶│ learning_system   │
│  (Component 2)│        │ (Component 5)    │
│               │        │                  │
│ - Skill score │        │ - SQLite KB      │
│ - Answer qual │        │ - Periodic LLM   │
│ - GitHub API  │        │   analysis       │
│ - AI phrases  │        │ - Weight updates │
│ - Tiers       │        │ - NL queries     │
└───────┬───────┘        └────────┬─────────┘
        │                         │
        ▼                         │
┌───────────────┐                 │
│ email_agent   │◀───────────────┘
│ (Component 3) │     (weights feed back into scoring)
│               │
│ - Gmail API   │
│ - Thread track│
│ - LLM replies │
│ - Multi-round │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ anti_cheat.py │
│ (Component 4) │
│               │
│ - AI phrases  │
│ - Embeddings  │
│ - Copy rings  │
│ - Timing      │
│ - Strike sys  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ orchestrator  │  ← ties everything together
│ (Component 6) │  ← 24/7 polling loop
│               │  ← error handling + recovery
└───────────────┘
```

## 🖥️ UI Dashboard — GenoTek Hiring Agent

A real-time interactive dashboard to monitor the entire AI hiring pipeline — from candidate ingestion to anti-cheat detection and analytics.

---

### 📊 Dashboard Overview

![Dashboard](./assets/dashboard.png)

- Total candidates processed in real-time
- Tier distribution (Fast-Track, Standard, Review, Reject)
- AI phrase fingerprint detection
- Scoring weight visualization

---

### ⚡ Pipeline Execution

![Pipeline](./assets/pipeline.png)

- Run full pipeline with one click
- Supports mock / CSV / ATS integrations
- Real-time progress tracking
- Live execution logs

---

### 📜 Live Logs & Anti-Cheat Detection

![Logs](./assets/logs.png)

- Streaming logs during execution
- AI fingerprint detection in candidate answers
- Strike system for suspicious responses
- Transparent decision-making system

---

### 👥 Candidate Management

![Candidates](./assets/candidates1.png)
![Candidates](./assets/candidates2.png)

- Ranked candidate list with scores
- Tier-based filtering (Fast-Track, Standard, etc.)
- Search by name/email
- Strike and flag tracking per candidate

---

### 🛡️ Anti-Cheat System

![AntiCheat](./assets/anticheat.png)

- Detect AI-generated answers in real-time
- Pattern-based + embedding-based detection
- Strike board for tracking suspicious candidates
- Human vs AI answer comparison testing

---

### 📈 Analytics & Insights

![Analytics](./assets/analytics.png)

- Top-performing candidates leaderboard
- Score distribution insights
- Natural language query system (LLM-powered)
- Hiring decision support

---

### 🤖 AI Query System (LLM Integration)

![LLM](./assets/llm.png)

- Ask questions like:
  - “Which candidates showed the most original thinking?”
  - “What % used AI-generated answers?”
- Powered by Gemini / LLM backend
- Works with structured hiring data

---

## 🧠 Key Highlights

- ⚡ Fully interactive frontend (React-based inside HTML)
- 🔄 Real-time pipeline execution with logs
- 🛡️ Built-in AI cheating detection system
- 📊 Data-driven hiring decisions
- 🤖 Optional LLM-powered analytics layer

---

## 📂 Folder Structure (UI)

frontend/
├── dashboard.html
├── assets/
│ ├── dashboard.png
│ ├── pipeline.png
│ ├── logs.png
│ ├── candidates.png
│ ├── anticheat.png
│ ├── analytics.png
│ └── llm.png

---

## Files

| File | Component | Description |
|------|-----------|-------------|
| `scorer.py` | 2 | Score & rank applicants from CSV |
| `anti_cheat.py` | 4 | AI/copy detection + strike system |
| `email_agent.py` | 3 | Gmail multi-round conversation manager |
| `learning_system.py` | 5 | SQLite KB + periodic LLM analysis |
| `orchestrator.py` | 6 | Main CLI + full pipeline integration |
| `requirements.txt` | — | Dependencies |

**Note on Component 1 (ACCESS):** Bypassing reCAPTCHA Enterprise or scraping Internshala
without authorization violates their Terms of Service. This codebase assumes you're working
with a CSV export (which Internshala supports for employer accounts) or data from a
platform that provides a proper API. The scoring, email, anti-cheat, and learning modules
are all data-source agnostic.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
export CHECK_GITHUB="true"           # optional — enables GitHub API calls

# 3. For email functionality, set up Gmail OAuth:
#    - Go to Google Cloud Console → APIs → Gmail API → Enable
#    - Create OAuth 2.0 credentials → Download as credentials.json
#    - Place credentials.json in this directory
#    - First run will open a browser for OAuth consent
```

---

## Usage

### Score a CSV of applicants
```bash
python orchestrator.py --score applicants.csv --output scored.csv
```

Expected CSV columns (flexible — edit column name args in scorer.py):
`name, email, github, skills, answer_1, answer_2`

### Send opening emails to top candidates
```bash
python orchestrator.py --engage scored.csv --tiers Fast-Track Standard
python orchestrator.py --engage scored.csv --dry-run  # preview only
```

### Start the 24/7 live loop
```bash
python orchestrator.py --run
```

Polls Gmail every 2 minutes. Replies to candidates, runs anti-cheat checks,
triggers analysis every 10 new candidates. Survives process restarts (state in SQLite).

### Run individual components in demo mode
```bash
python scorer.py          # Creates sample CSV and scores it
python anti_cheat.py      # Runs detection checks on sample answers
python email_agent.py     # Shows email generation (dry run)
python learning_system.py # Seeds KB with sample data + runs analysis
```

### Query the knowledge base
```bash
python orchestrator.py --query "Which 3 candidates showed the most original thinking?"
python orchestrator.py --query "What percentage of candidates mentioned Selenium first?"
python orchestrator.py --query "What are the most common AI-generated phrases?"
```

### View pipeline status
```bash
python orchestrator.py --status
```

---

## Scoring Algorithm

```
Final Score (0–100) =
  skill_score       × 1.5   (max 30: Python=3, Kubernetes=4, etc.)
+ answer_quality    × 2.0   (max 20: specificity, code, numbers, tools named)
+ github_score      × 2.0   (max 20: repos, commits, stars, account age)
+ completeness      × 1.0   (max 10: all fields filled)
- ai_penalty                (−5 per AI phrase, max −20)

All normalized to 0–100.
```

**Tiers:**
- Fast-Track: ≥ 75
- Standard: 55–74
- Review: 35–54
- Reject: < 35

Weights automatically update after every 10 candidates based on LLM analysis of
which early signals predicted strong performance.

---

## Anti-Cheat: Three Layers

1. **Phrase detection** — 25 known AI fingerprints (regex, instant, free)
2. **Embedding similarity** — Compare candidate answer to fresh LLM answer on same
   question using `all-MiniLM-L6-v2` cosine similarity. ≥ 80% = strike.
3. **Copy ring** — Pairwise O(n²) cosine similarity across all candidates per round.
   If 3+ share ≥ 75% similarity, all flagged.
4. **Timing** — Reply in < 2 minutes with > 80 words = suspicious.

**Strike system:** 3 strikes → automatic elimination (persisted to SQLite).

---

## Self-Learning Loop

Every 10 candidates, the system automatically:
1. Analyzes all interactions via Claude API
2. Extracts: success predictors, new AI phrases, which questions worked best
3. Updates scoring weights in the DB
4. New candidates are scored with updated weights

You can also query the knowledge base in plain English at any time.

---

## Error Handling

| Failure | Recovery |
|---------|----------|
| Gmail down | Polling loop retries after 30s, no candidates lost |
| Email bounces | Logged to DB, candidate status set to `bounced` |
| LLM API failure | Falls back to template reply, logs the failure |
| Unexpected CSV format | ValueError with clear message on missing columns |
| GitHub API rate limit | 0.5s sleep between requests; falls back to URL-only score |
| Process restart | All state in SQLite — resumes exactly where it left off |
| Candidate goes cold | Future feature: flag conversations with no reply in 72h |

---

## Deployment (24/7 on a server)

```bash
# Using systemd (recommended for VPS/EC2)
sudo nano /etc/systemd/system/hiring-agent.service

[Unit]
Description=GenoTek Hiring Agent
After=network.target

[Service]
WorkingDirectory=/opt/hiring-agent
ExecStart=/usr/bin/python3 orchestrator.py --run
Restart=always
RestartSec=10
Environment=ANTHROPIC_API_KEY=sk-ant-...

[Install]
WantedBy=multi-user.target

sudo systemctl enable hiring-agent
sudo systemctl start hiring-agent
sudo journalctl -u hiring-agent -f  # tail logs
```

All state persists in `hiring_agent.db` (SQLite). On restart, the agent
re-reads all active conversations and resumes polling — no candidate is
ever dropped.
