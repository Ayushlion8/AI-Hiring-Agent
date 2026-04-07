"""
api.py — Flask REST API for the GenoTek Hiring Agent UI

Run:
    pip install flask flask-cors
    python api.py

Then open dashboard.html in your browser.
Endpoints served at http://localhost:5000/api/...
"""

import os
import sys
import csv
import json
import time
import sqlite3
import threading
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS

# ── Add project dir to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from llm_client import llm_status, LLM_PROVIDER
from learning_system import KnowledgeBase
from anti_cheat import AntiCheatPipeline, check_ai_phrases
from access import ATSRouter

DB_PATH = os.getenv("DB_PATH", "hiring_agent.db")

app = Flask(__name__)
CORS(app)  # Allow dashboard.html to call the API

kb = KnowledgeBase(DB_PATH)
anti_cheat = AntiCheatPipeline(db_path=DB_PATH)

# ── SSE log queue for live streaming ────────────────────────────
import queue
_log_queue = queue.Queue()

class LogStream:
    """Capture print() output and push to SSE queue."""
    def __init__(self, original):
        self.original = original

    def write(self, text):
        self.original.write(text)
        if text.strip():
            _log_queue.put(text.rstrip())

    def flush(self):
        self.original.flush()

sys.stdout = LogStream(sys.stdout)


# ════════════════════════════════════════════════════════════════
# STATUS
# ════════════════════════════════════════════════════════════════
@app.route("/api/status")
def status():
    total = kb.get_candidate_count()
    weights = kb.get_current_weights()
    selenium_pct = kb.get_selenium_first_approach_pct()

    candidates = kb.get_all_candidates_summary()
    tier_counts = {}
    outcome_counts = {}
    for c in candidates:
        tier_counts[c["tier"]] = tier_counts.get(c["tier"], 0) + 1
        outcome_counts[c["final_outcome"]] = outcome_counts.get(c["final_outcome"], 0) + 1

    phrases = kb.get_top_patterns("ai_phrase", 10)

    # Conversation stats from email agent DB
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    convs = conn.execute("SELECT status, current_round FROM conversations").fetchall()
    conn.close()
    conv_stats = {"active": 0, "completed": 0, "total": len(convs)}
    round_dist = {}
    for c in convs:
        conv_stats[c["status"]] = conv_stats.get(c["status"], 0) + 1
        r = str(c["current_round"])
        round_dist[r] = round_dist.get(r, 0) + 1

    return jsonify({
        "total_candidates": total,
        "tier_breakdown": tier_counts,
        "outcome_breakdown": outcome_counts,
        "scoring_weights": weights,
        "selenium_first_pct": selenium_pct,
        "llm_provider": llm_status(),
        "llm_active": LLM_PROVIDER != "none",
        "top_ai_phrases": [{"phrase": p["value"], "count": p["frequency"]} for p in phrases],
        "conversations": conv_stats,
        "round_distribution": round_dist,
        "db_path": DB_PATH,
    })


# ════════════════════════════════════════════════════════════════
# CANDIDATES
# ════════════════════════════════════════════════════════════════
@app.route("/api/candidates")
def get_candidates():
    tier    = request.args.get("tier")
    search  = request.args.get("search", "").lower()
    page    = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    # Read from scored CSV if it exists, else from DB
    csv_path = Path("scored_candidates.csv")
    if csv_path.exists():
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if tier and row.get("tier") != tier:
                    continue
                if search and search not in row.get("name", "").lower() and search not in row.get("email", "").lower():
                    continue
                rows.append(row)
    else:
        # Fall back to DB
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        query = "SELECT id as email, name, initial_score as score, tier, final_outcome FROM candidates"
        params = []
        if tier:
            query += " WHERE tier = ?"
            params.append(tier)
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        conn.close()
        if search:
            rows = [r for r in rows if search in r.get("name","").lower() or search in r.get("email","").lower()]

    total = len(rows)
    start = (page - 1) * per_page
    page_rows = rows[start:start + per_page]

    # Enrich with strike info
    for row in page_rows:
        cid = row.get("email", "")
        strike_info = anti_cheat.strike_system.get_status(cid)
        row["strikes"] = strike_info["strikes"]
        row["eliminated"] = strike_info["eliminated"]

    return jsonify({
        "candidates": page_rows,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
    })


@app.route("/api/candidates/<path:candidate_id>")
def get_candidate(candidate_id):
    """Full profile: scores, flags, conversation history, strikes."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cand = conn.execute(
        "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
    ).fetchone()

    interactions = conn.execute(
        "SELECT * FROM interactions WHERE candidate_id = ? ORDER BY timestamp",
        (candidate_id,)
    ).fetchall()

    conv = conn.execute(
        "SELECT * FROM conversations WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()

    strikes_row = conn.execute(
        "SELECT strikes, eliminated, log FROM candidate_strikes WHERE candidate_id = ?",
        (candidate_id,)
    ).fetchone()

    conn.close()

    return jsonify({
        "candidate": dict(cand) if cand else None,
        "interactions": [dict(i) for i in interactions],
        "conversation": dict(conv) if conv else None,
        "strikes": {
            "count": strikes_row[0] if strikes_row else 0,
            "eliminated": bool(strikes_row[1]) if strikes_row else False,
            "log": json.loads(strikes_row[2]) if strikes_row else [],
        },
    })


# ════════════════════════════════════════════════════════════════
# PIPELINE ACTIONS
# ════════════════════════════════════════════════════════════════
_pipeline_running = False
_pipeline_status = {"step": "idle", "progress": 0, "message": ""}


def _run_pipeline_thread(source: str, count: int):
    global _pipeline_running, _pipeline_status
    try:
        _pipeline_status = {"step": "fetching", "progress": 10, "message": "Fetching candidates..."}

        router = ATSRouter(source=source, mock_count=count)
        router.fetch_and_save("candidates_raw.csv", db_path=DB_PATH)

        _pipeline_status = {"step": "scoring", "progress": 35, "message": "Scoring candidates..."}

        from scorer import score_candidates
        scored_df = score_candidates(
            "candidates_raw.csv", "scored_candidates.csv",
            check_github_api=False, verbose=True
        )

        _pipeline_status = {"step": "logging", "progress": 65, "message": "Logging to knowledge base..."}

        for _, row in scored_df.iterrows():
            cid = row["email"]
            kb.log_candidate(cid, row["name"], row["email"], row["score"], row["tier"])
            kb.log_interaction(cid, "scored",
                               f"Score: {row['score']}, Tier: {row['tier']}, Flags: {row.get('flags','')}",
                               round_number=0)

        _pipeline_status = {"step": "anti_cheat", "progress": 80, "message": "Running anti-cheat scan..."}

        csv_path = Path("candidates_raw.csv")
        flagged = 0
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("email", "")
                    answer = " ".join([row.get("answer_1",""), row.get("answer_2","")])
                    result = check_ai_phrases(cid, answer)
                    if result.is_flagged:
                        anti_cheat.strike_system.add_strike(cid, result.explanation, result.evidence)
                        kb.log_interaction(cid, "flagged", result.explanation)
                        flagged += 1

        tier_counts = scored_df["tier"].value_counts().to_dict()
        _pipeline_status = {
            "step": "done",
            "progress": 100,
            "message": f"Done. {len(scored_df)} scored. {flagged} flagged. {tier_counts}",
        }

    except Exception as e:
        _pipeline_status = {"step": "error", "progress": 0, "message": str(e)}
    finally:
        _pipeline_running = False


@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    global _pipeline_running
    if _pipeline_running:
        return jsonify({"error": "Pipeline already running"}), 409

    body   = request.get_json() or {}
    source = body.get("source", "mock")
    count  = int(body.get("count", 100))

    # Clear conversation state for fresh run
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM conversations")
    conn.execute("DELETE FROM processed_message_ids")
    conn.commit()
    conn.close()

    _pipeline_running = True
    thread = threading.Thread(target=_run_pipeline_thread, args=(source, count), daemon=True)
    thread.start()

    return jsonify({"started": True, "source": source, "count": count})


@app.route("/api/pipeline/status")
def pipeline_status():
    return jsonify({**_pipeline_status, "running": _pipeline_running})


@app.route("/api/pipeline/logs")
def pipeline_logs():
    """Server-Sent Events stream of live log output."""
    def generate():
        yield "data: {\"msg\": \"Log stream connected\"}\n\n"
        while True:
            try:
                msg = _log_queue.get(timeout=30)
                payload = json.dumps({"msg": msg})
                yield f"data: {payload}\n\n"
            except queue.Empty:
                yield "data: {\"msg\": \"ping\"}\n\n"
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ════════════════════════════════════════════════════════════════
# ANTI-CHEAT
# ════════════════════════════════════════════════════════════════
@app.route("/api/anticheat/check", methods=["POST"])
def anticheat_check():
    """Check a single answer for AI phrases on demand."""
    body = request.get_json() or {}
    answer = body.get("answer", "")
    candidate_id = body.get("candidate_id", "test_user")

    result = check_ai_phrases(candidate_id, answer)
    return jsonify({
        "flagged": result.is_flagged,
        "score": result.score,
        "explanation": result.explanation,
        "matched_patterns": result.evidence.get("matched_patterns", []),
    })


@app.route("/api/anticheat/strikes")
def get_strikes():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT candidate_id, strikes, eliminated, log FROM candidate_strikes ORDER BY strikes DESC"
    ).fetchall()
    conn.close()
    return jsonify([{
        "candidate_id": r["candidate_id"],
        "strikes": r["strikes"],
        "eliminated": bool(r["eliminated"]),
        "log": json.loads(r["log"]),
    } for r in rows])


# ════════════════════════════════════════════════════════════════
# ANALYTICS
# ════════════════════════════════════════════════════════════════
@app.route("/api/analytics/query", methods=["POST"])
def analytics_query():
    if LLM_PROVIDER == "none":
        return jsonify({"error": "No LLM configured. Set GEMINI_API_KEY."}), 400
    body = request.get_json() or {}
    question = body.get("question", "")
    if not question:
        return jsonify({"error": "question required"}), 400
    answer = kb.query(question, "")
    return jsonify({"answer": answer})


@app.route("/api/analytics/scores")
def score_distribution():
    csv_path = Path("scored_candidates.csv")
    if not csv_path.exists():
        return jsonify({"buckets": [], "tiers": {}})

    buckets = {f"{i}-{i+10}": 0 for i in range(0, 100, 10)}
    tiers = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row.get("score", 0))
                bucket = f"{int(score//10)*10}-{int(score//10)*10+10}"
                buckets[bucket] = buckets.get(bucket, 0) + 1
                tier = row.get("tier", "Unknown")
                tiers[tier] = tiers.get(tier, 0) + 1
            except:
                pass
    return jsonify({
        "buckets": [{"range": k, "count": v} for k, v in sorted(buckets.items())],
        "tiers": tiers,
    })


@app.route("/api/analytics/top_candidates")
def top_candidates():
    n = int(request.args.get("n", 10))
    csv_path = Path("scored_candidates.csv")
    if not csv_path.exists():
        return jsonify([])
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: float(r.get("score", 0)), reverse=True)
    return jsonify(rows[:n])


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("GenoTek Hiring Agent API")
    print(f"LLM: {llm_status()}")
    print(f"DB:  {DB_PATH}")
    print("Serving at http://localhost:5000")
    print("Open dashboard.html in your browser")
    print("=" * 55)
    app.run(debug=False, port=5000, threaded=True)
