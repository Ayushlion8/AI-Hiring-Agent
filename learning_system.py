"""
Component 5: SELF-LEARNING — The System Gets Smarter Over Time

- SQLite knowledge base: every email, score, decision, detection event
- Periodic analysis after every N candidates
- Produces insight reports that feed back into scoring weights
- Answers queries like "which 3 candidates showed most original thinking?"
"""

import json
import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

ANALYSIS_TRIGGER_EVERY_N = 10   # Run analysis after every 10 candidates


# ------------------------------------------------------------------ #
# Schema
# ------------------------------------------------------------------ #
SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    applied_at TEXT,
    initial_score REAL,
    tier TEXT,
    final_outcome TEXT,   -- 'hired' | 'rejected' | 'withdrawn' | 'pending'
    notes TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT,
    interaction_type TEXT,  -- 'email_sent' | 'email_received' | 'scored' | 'flagged' | 'advanced' | 'eliminated'
    round_number INTEGER DEFAULT 1,
    content TEXT,
    metadata TEXT,          -- JSON blob for extra fields
    timestamp TEXT,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS scoring_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER,
    weights TEXT,           -- JSON: {"skill": 1.5, "answer_quality": 2.0, ...}
    generated_at TEXT,
    rationale TEXT
);

CREATE TABLE IF NOT EXISTS analysis_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type TEXT,
    candidates_analyzed INTEGER,
    report TEXT,            -- Full LLM analysis text
    insights TEXT,          -- JSON: structured insights extracted
    generated_at TEXT
);

CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT,      -- 'ai_phrase' | 'success_signal' | 'rejection_signal' | 'copy_ring_phrase'
    value TEXT,
    frequency INTEGER DEFAULT 1,
    last_seen TEXT,
    notes TEXT
);
"""


class KnowledgeBase:
    def __init__(self, db_path: str = "hiring_agent.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            for stmt in SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.commit()

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def log_candidate(
        self,
        candidate_id: str,
        name: str,
        email: str,
        initial_score: float,
        tier: str,
        applied_at: Optional[str] = None,
    ):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO candidates
                (id, name, email, applied_at, initial_score, tier, final_outcome)
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """, (
                candidate_id, name, email,
                applied_at or datetime.now(timezone.utc).isoformat(),
                initial_score, tier,
            ))
            conn.commit()

    def log_interaction(
        self,
        candidate_id: str,
        interaction_type: str,
        content: str = "",
        round_number: int = 1,
        metadata: dict = None,
    ):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO interactions
                (candidate_id, interaction_type, round_number, content, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                candidate_id,
                interaction_type,
                round_number,
                content,
                json.dumps(metadata or {}),
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()

    def update_outcome(self, candidate_id: str, outcome: str, notes: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE candidates SET final_outcome = ?, notes = ? WHERE id = ?",
                (outcome, notes, candidate_id)
            )
            conn.commit()

    def log_pattern(self, pattern_type: str, value: str, notes: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id, frequency FROM patterns WHERE pattern_type = ? AND value = ?",
                (pattern_type, value)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE patterns SET frequency = ?, last_seen = ? WHERE id = ?",
                    (existing[1] + 1, datetime.now(timezone.utc).isoformat(), existing[0])
                )
            else:
                conn.execute("""
                    INSERT INTO patterns (pattern_type, value, frequency, last_seen, notes)
                    VALUES (?, ?, 1, ?, ?)
                """, (pattern_type, value, datetime.now(timezone.utc).isoformat(), notes))
            conn.commit()

    def save_weights(self, weights: dict, rationale: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            max_version = conn.execute("SELECT MAX(version) FROM scoring_weights").fetchone()[0] or 0
            conn.execute("""
                INSERT INTO scoring_weights (version, weights, generated_at, rationale)
                VALUES (?, ?, ?, ?)
            """, (
                max_version + 1,
                json.dumps(weights),
                datetime.now(timezone.utc).isoformat(),
                rationale,
            ))
            conn.commit()
        return max_version + 1

    def get_current_weights(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT weights FROM scoring_weights ORDER BY version DESC LIMIT 1"
            ).fetchone()
        if row:
            return json.loads(row[0])
        # Default weights
        return {
            "skill_score": 1.5,
            "answer_quality": 2.0,
            "github_score": 2.0,
            "completeness": 1.0,
        }

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_candidate_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    def get_all_candidates_summary(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT c.id, c.name, c.initial_score, c.tier, c.final_outcome,
                       COUNT(i.id) as interaction_count
                FROM candidates c
                LEFT JOIN interactions i ON c.id = i.candidate_id
                GROUP BY c.id
                ORDER BY c.initial_score DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def get_interactions_for_analysis(self, limit: int = 200) -> list[dict]:
        """Get recent interactions for LLM analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT c.name, c.initial_score, c.tier, c.final_outcome,
                       i.interaction_type, i.round_number, i.content, i.metadata, i.timestamp
                FROM interactions i
                JOIN candidates c ON c.id = i.candidate_id
                ORDER BY i.timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_top_candidates_by_originality(self, n: int = 3) -> list[dict]:
        """
        Identify candidates whose answers are most different from each other
        (high variance = original thinking).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Candidates who advanced furthest AND weren't flagged for AI
            rows = conn.execute("""
                SELECT c.id, c.name, c.initial_score, c.tier,
                       COUNT(CASE WHEN i.interaction_type = 'email_received' THEN 1 END) as rounds_completed,
                       COUNT(CASE WHEN i.interaction_type = 'flagged' THEN 1 END) as flags
                FROM candidates c
                LEFT JOIN interactions i ON c.id = i.candidate_id
                GROUP BY c.id
                HAVING flags = 0
                ORDER BY rounds_completed DESC, c.initial_score DESC
                LIMIT ?
            """, (n,)).fetchall()
        return [dict(r) for r in rows]

    def get_selenium_first_approach_pct(self) -> float:
        """What percentage of candidates mentioned Selenium as their first approach?"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE interaction_type = 'email_received' AND round_number = 1"
            ).fetchone()[0]

            selenium_count = conn.execute(
                """SELECT COUNT(*) FROM interactions
                   WHERE interaction_type = 'email_received'
                   AND round_number = 1
                   AND LOWER(content) LIKE '%selenium%'"""
            ).fetchone()[0]

        if total == 0:
            return 0.0
        return round((selenium_count / total) * 100, 1)

    def get_top_patterns(self, pattern_type: str, limit: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT value, frequency, notes FROM patterns
                WHERE pattern_type = ?
                ORDER BY frequency DESC
                LIMIT ?
            """, (pattern_type, limit)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def should_run_analysis(self) -> bool:
        count = self.get_candidate_count()
        with sqlite3.connect(self.db_path) as conn:
            last_report = conn.execute(
                "SELECT candidates_analyzed FROM analysis_reports ORDER BY id DESC LIMIT 1"
            ).fetchone()
        last_analyzed = last_report[0] if last_report else 0
        return (count - last_analyzed) >= ANALYSIS_TRIGGER_EVERY_N

    def run_periodic_analysis(self, api_key: str = "") -> Optional[dict]:
        """
        Every N candidates, analyze patterns and potentially update scoring weights.
        Returns the analysis result dict.
        """
        if not self.should_run_analysis():
            return None

        print(f"\nRunning periodic analysis (every {ANALYSIS_TRIGGER_EVERY_N} candidates)...")

        candidates = self.get_all_candidates_summary()
        interactions = self.get_interactions_for_analysis(limit=100)
        patterns = self.get_top_patterns("ai_phrase", limit=10)
        selenium_pct = self.get_selenium_first_approach_pct()
        top_original = self.get_top_candidates_by_originality(3)

        # Build a structured data summary for the LLM
        data_summary = {
            "total_candidates": len(candidates),
            "tier_breakdown": {},
            "outcome_breakdown": {},
            "selenium_first_approach_pct": selenium_pct,
            "top_ai_phrases_detected": [p["value"] for p in patterns[:5]],
            "top_original_candidates": [
                {"name": c["name"], "score": c["initial_score"], "rounds": c["rounds_completed"]}
                for c in top_original
            ],
            "sample_interactions": [
                {
                    "candidate": i["name"],
                    "type": i["interaction_type"],
                    "round": i["round_number"],
                    "excerpt": i["content"][:300] if i["content"] else "",
                }
                for i in interactions[:20]
            ],
        }

        for c in candidates:
            tier = c["tier"]
            outcome = c["final_outcome"]
            data_summary["tier_breakdown"][tier] = data_summary["tier_breakdown"].get(tier, 0) + 1
            data_summary["outcome_breakdown"][outcome] = data_summary["outcome_breakdown"].get(outcome, 0) + 1

        analysis_text = ""
        insights = {}

        if ANTHROPIC_AVAILABLE and api_key:
            client = anthropic.Anthropic(api_key=api_key)
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1500,
                    system="""You are an AI hiring system analyst. You analyze candidate interaction data 
to find patterns and improve the scoring algorithm. Be concrete and data-driven.
Respond in JSON format with keys:
- "summary": 2-sentence overview
- "key_findings": list of 3-5 specific findings with numbers
- "ai_generation_patterns": new phrases or patterns discovered
- "weight_adjustments": suggested changes to scoring weights (as JSON delta)
- "best_questions": which questions produced most differentiated responses
- "common_first_approach": what candidates try first
- "success_predictors": what patterns predicted strong Round 2 performance""",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Analyze this hiring data and return JSON insights:\n\n"
                            f"{json.dumps(data_summary, indent=2)}"
                        )
                    }]
                )
                raw = response.content[0].text
                # Strip markdown code fences if present
                raw = raw.replace("```json", "").replace("```", "").strip()
                try:
                    insights = json.loads(raw)
                    analysis_text = json.dumps(insights, indent=2)
                except json.JSONDecodeError:
                    analysis_text = raw
                    insights = {"raw": raw}

            except Exception as e:
                print(f"  LLM analysis failed: {e}")
                analysis_text = "Analysis failed — LLM unavailable"
                insights = {}
        else:
            # Fallback: rule-based summary
            hired = data_summary["outcome_breakdown"].get("hired", 0)
            rejected = data_summary["outcome_breakdown"].get("rejected", 0)
            analysis_text = (
                f"Candidates analyzed: {len(candidates)}. "
                f"Hired: {hired}, Rejected: {rejected}. "
                f"Selenium first-approach rate: {selenium_pct}%. "
                f"Top AI phrases: {', '.join(data_summary['top_ai_phrases_detected'][:3])}."
            )
            insights = {"summary": analysis_text}

        # Save report
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO analysis_reports
                (report_type, candidates_analyzed, report, insights, generated_at)
                VALUES ('periodic', ?, ?, ?, ?)
            """, (
                len(candidates),
                analysis_text,
                json.dumps(insights),
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()

        # Apply weight adjustments if suggested
        if "weight_adjustments" in insights:
            current_weights = self.get_current_weights()
            adjustments = insights["weight_adjustments"]
            if isinstance(adjustments, dict):
                new_weights = {**current_weights, **adjustments}
                version = self.save_weights(
                    new_weights,
                    rationale=f"Auto-updated after {len(candidates)} candidates — {insights.get('summary', '')}"
                )
                print(f"  Scoring weights updated to version {version}")
                print(f"  New weights: {new_weights}")

        # Register newly discovered AI phrases as patterns
        new_phrases = insights.get("ai_generation_patterns", [])
        for phrase in new_phrases:
            if isinstance(phrase, str):
                self.log_pattern("ai_phrase", phrase, "Auto-discovered by periodic analysis")

        print(f"  Analysis complete. Report saved.")
        return insights

    def query(self, natural_language_question: str, api_key: str) -> str:
        """
        Answer free-form questions about the candidate pool.
        Examples:
          "Which 3 candidates showed the most original thinking?"
          "What percentage of candidates mentioned Selenium first?"
          "What are the most common AI-generated phrases we've seen?"
        """
        if not ANTHROPIC_AVAILABLE or not api_key:
            return "LLM not available for natural language queries."

        # Gather context
        candidates = self.get_all_candidates_summary()
        patterns = self.get_top_patterns("ai_phrase")
        selenium_pct = self.get_selenium_first_approach_pct()
        top_original = self.get_top_candidates_by_originality(5)
        weights = self.get_current_weights()

        context = {
            "candidates": candidates[:50],
            "selenium_first_pct": selenium_pct,
            "top_ai_phrases": [p["value"] for p in patterns[:10]],
            "top_original_candidates": top_original,
            "current_scoring_weights": weights,
        }

        client = anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=(
                    "You are an analyst for an AI hiring system. "
                    "Answer questions about candidate data concisely and accurately. "
                    "Use specific names and numbers from the data provided."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Data context:\n{json.dumps(context, indent=2)}\n\n"
                        f"Question: {natural_language_question}"
                    )
                }]
            )
            return response.content[0].text
        except Exception as e:
            return f"Query failed: {e}"


# ------------------------------------------------------------------ #
# Demo
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    kb = KnowledgeBase("demo_learning.db")

    # Seed with sample data
    print("Seeding knowledge base with sample candidates...\n")

    sample_candidates = [
        ("alice@x.com", "Alice Sharma", 88.0, "Fast-Track", "hired"),
        ("bob@x.com", "Bob Kumar", 42.0, "Review", "rejected"),
        ("charlie@x.com", "Charlie Das", 71.0, "Standard", "pending"),
        ("diana@x.com", "Diana Mehta", 15.0, "Reject", "rejected"),
        ("eve@x.com", "Eve Patel", 91.0, "Fast-Track", "pending"),
        ("frank@x.com", "Frank Singh", 55.0, "Standard", "pending"),
        ("grace@x.com", "Grace Liu", 30.0, "Review", "rejected"),
        ("henry@x.com", "Henry Park", 78.0, "Fast-Track", "pending"),
        ("iris@x.com", "Iris Chen", 62.0, "Standard", "pending"),
        ("jack@x.com", "Jack Brown", 48.0, "Review", "pending"),
        ("karan@x.com", "Karan Mehta", 85.0, "Fast-Track", "pending"),
    ]

    for email, name, score, tier, outcome in sample_candidates:
        kb.log_candidate(email, name, email, score, tier)
        kb.update_outcome(email, outcome)

        # Sample interactions
        kb.log_interaction(email, "scored", f"Initial score: {score}", round_number=0)
        if score > 60:
            kb.log_interaction(
                email, "email_received",
                "I tried Playwright with stealth mode first, then switched to curl-impersonate "
                "when I hit the TLS fingerprint check. The reCAPTCHA v3 scores headless Chrome near 0."
                if score > 75 else
                "I would use Selenium to automate the login and then BeautifulSoup to parse the HTML.",
                round_number=1
            )

    # Log some AI phrase patterns
    kb.log_pattern("ai_phrase", "i'd be happy to help", "ChatGPT opener")
    kb.log_pattern("ai_phrase", "here's a comprehensive overview", "ChatGPT structure")
    kb.log_pattern("ai_phrase", "in today's rapidly evolving landscape", "ChatGPT filler")
    kb.log_pattern("ai_phrase", "it's worth noting", "Claude opener")
    kb.log_pattern("success_signal", "curl-impersonate", "Indicates hands-on experience")
    kb.log_pattern("success_signal", "tls fingerprint", "Shows deep knowledge")
    kb.log_pattern("rejection_signal", "i would use selenium", "Generic non-tested answer")

    print(f"Total candidates: {kb.get_candidate_count()}")
    print(f"Selenium first-approach rate: {kb.get_selenium_first_approach_pct()}%")
    print(f"\nTop 3 original candidates:")
    for c in kb.get_top_candidates_by_originality(3):
        print(f"  {c['name']}: score={c['initial_score']}, rounds={c['rounds_completed']}, flags={c['flags']}")

    print(f"\nCurrent scoring weights: {kb.get_current_weights()}")

    # Run analysis (needs 10+ candidates, which we have)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        print("\nRunning LLM-powered analysis...")
        insights = kb.run_periodic_analysis(api_key)
        if insights:
            print(f"\nKey findings: {insights.get('key_findings', 'N/A')}")

        print("\nQuerying knowledge base...")
        answer = kb.query(
            "Which 3 candidates showed the most original thinking, and why?",
            api_key
        )
        print(f"\nQuery result:\n{answer}")
    else:
        print("\nSet ANTHROPIC_API_KEY for LLM-powered analysis and queries.")
        print("Running rule-based analysis instead...")
        kb.run_periodic_analysis(api_key="")
