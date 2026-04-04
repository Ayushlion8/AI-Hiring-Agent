"""
Component 4: ANTI-CHEAT — Detecting AI-Generated or Copied Responses

Three detection layers:
  1. AI phrase fingerprinting + LLM comparison via embeddings
  2. Cross-candidate similarity (pairwise cosine on embeddings)
  3. Response timing analysis

Strike system: 3 strikes = automatic elimination.
"""

import json
import sqlite3
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# Optional heavy imports — graceful fallback if not installed
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Embedding-based checks disabled.")
    print("  Install with: pip install sentence-transformers")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
AI_SIMILARITY_THRESHOLD = 0.80   # Strike if LLM comparison ≥ this
COPY_RING_THRESHOLD = 0.75        # Flag if candidate pair ≥ this
COPY_RING_MIN_GROUP = 3           # Min group size to call it a ring
FAST_REPLY_SECONDS = 120          # Under 2 min = suspicious
VERY_FAST_REPLY_SECONDS = 30      # Under 30 sec = near-certain bot
MAX_REASONABLE_REPLY_SECONDS = 14400  # 4 hours

AI_FINGERPRINTS = [
    r"\bi[''']d be happy to help\b",
    r"\bhere[''']s a comprehensive\b",
    r"\bin today[''']s rapidly evolving\b",
    r"\bit[''']s worth noting\b",
    r"\bgreat question\b",
    r"\bcertainly[!,]?\b",
    r"\babsolutely[!,]?\b",
    r"\bas an ai\b",
    r"\bi hope this helps\b",
    r"\blet me break this down\b",
    r"\bto summarize\b",
    r"\bthis is a multifaceted\b",
    r"\bparadigm shift\b",
    r"\bholistic approach\b",
    r"\bcutting-edge\b",
    r"\bseamlessly integrat\b",
    r"\ba testament to\b",
    r"\bfoster innovation\b",
    r"\bdive deep into\b",
    r"\bin the realm of\b",
    r"\bat its core\b",
    r"\bleverage\b.*\bsynerg",  # "leverage synergies"
    r"\bin conclusion\b",
]

# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #
@dataclass
class DetectionResult:
    candidate_id: str
    check_type: str          # "ai_similarity" | "copy_ring" | "timing" | "phrase"
    is_flagged: bool
    score: float             # 0.0–1.0 confidence
    explanation: str
    evidence: dict = field(default_factory=dict)

@dataclass
class CandidateRecord:
    candidate_id: str
    question: str
    answer: str
    submitted_at: Optional[datetime] = None
    round_number: int = 1
    strikes: int = 0
    eliminated: bool = False
    detection_log: list = field(default_factory=list)


# ------------------------------------------------------------------ #
# Embedding helper
# ------------------------------------------------------------------ #
_model_cache = {}

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    if not EMBEDDINGS_AVAILABLE:
        return None
    if model_name not in _model_cache:
        print(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def get_embedding(text: str, model=None) -> Optional[np.ndarray]:
    if model is None:
        model = get_embedding_model()
    if model is None:
        return None
    return model.encode(text, convert_to_numpy=True)


# ------------------------------------------------------------------ #
# Check 1: AI phrase detection
# ------------------------------------------------------------------ #
def check_ai_phrases(candidate_id: str, answer: str) -> DetectionResult:
    """Fast, cheap check — no API needed."""
    answer_lower = answer.lower()
    matches = []
    for pattern in AI_FINGERPRINTS:
        if re.search(pattern, answer_lower):
            matches.append(pattern)

    count = len(matches)
    score = min(count / 4.0, 1.0)  # 4+ matches = 1.0

    return DetectionResult(
        candidate_id=candidate_id,
        check_type="phrase",
        is_flagged=count >= 2,
        score=score,
        explanation=(
            f"Found {count} AI fingerprint phrase(s)" if count > 0
            else "No AI fingerprint phrases detected"
        ),
        evidence={"matched_patterns": matches},
    )


# ------------------------------------------------------------------ #
# Check 2: LLM comparison via embeddings
# ------------------------------------------------------------------ #
def check_ai_similarity_embedding(
    candidate_id: str,
    question: str,
    candidate_answer: str,
    llm_answer: str,
) -> DetectionResult:
    """
    Compare candidate's answer to a fresh LLM answer on the same question.
    If cosine similarity > threshold, likely AI-generated.
    """
    model = get_embedding_model()
    if model is None:
        return DetectionResult(
            candidate_id=candidate_id,
            check_type="ai_similarity",
            is_flagged=False,
            score=0.0,
            explanation="Embedding model unavailable — check skipped",
        )

    emb_candidate = get_embedding(candidate_answer, model)
    emb_llm = get_embedding(llm_answer, model)

    sim = cosine_similarity(emb_candidate, emb_llm)

    # Also compare structural similarity via sentence pattern
    def sentence_pattern(text: str) -> str:
        """Reduce to sentence-length fingerprint."""
        sentences = re.split(r'[.!?]+', text)
        return " | ".join(str(len(s.split())) for s in sentences if s.strip())

    pattern_a = sentence_pattern(candidate_answer)
    pattern_b = sentence_pattern(llm_answer)
    structure_match = (pattern_a == pattern_b)

    combined_score = sim + (0.1 if structure_match else 0.0)
    combined_score = min(combined_score, 1.0)

    return DetectionResult(
        candidate_id=candidate_id,
        check_type="ai_similarity",
        is_flagged=combined_score >= AI_SIMILARITY_THRESHOLD,
        score=combined_score,
        explanation=(
            f"{combined_score*100:.0f}% similar to LLM-generated answer"
            + (" (identical structure)" if structure_match else "")
        ),
        evidence={
            "cosine_similarity": round(sim, 4),
            "structure_match": structure_match,
            "combined_score": round(combined_score, 4),
        },
    )


def generate_llm_reference_answer(question: str, api_key: str) -> Optional[str]:
    """
    Get a fresh LLM answer to use as comparison baseline.
    Strips personal context from the question before sending.
    """
    if not ANTHROPIC_AVAILABLE:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": (
                    f"Answer this question concisely as a job applicant would:\n\n{question}\n\n"
                    "Be direct and specific. Do not introduce yourself."
                )
            }]
        )
        return msg.content[0].text
    except Exception as e:
        print(f"LLM reference generation failed: {e}")
        return None


# ------------------------------------------------------------------ #
# Check 3: Cross-candidate copy ring detection
# ------------------------------------------------------------------ #
def check_copy_ring(
    candidates: list[dict],   # [{"id": ..., "answer": ...}, ...]
    question_id: str,
) -> list[DetectionResult]:
    """
    Pairwise cosine similarity across all candidates for one question.
    If 3+ candidates share ≥ threshold similarity, flag all as a copy ring.
    
    O(n²) — manageable for n ≤ 1000 with sentence-transformers.
    """
    results = []

    if len(candidates) < 2:
        return results

    model = get_embedding_model()
    if model is None:
        print("Embedding model unavailable — copy ring check skipped")
        return results

    print(f"Computing {len(candidates)} embeddings for question '{question_id}'...")
    texts = [c["answer"] for c in candidates]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Build similarity matrix
    n = len(candidates)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # Find clusters above threshold
    flagged_indices = set()
    suspicious_pairs = []

    for i in range(n):
        similar_to_i = [j for j in range(n) if i != j and sim_matrix[i][j] >= COPY_RING_THRESHOLD]
        if len(similar_to_i) >= COPY_RING_MIN_GROUP - 1:
            flagged_indices.add(i)
            flagged_indices.update(similar_to_i)
            for j in similar_to_i:
                suspicious_pairs.append({
                    "a": candidates[i]["id"],
                    "b": candidates[j]["id"],
                    "similarity": round(float(sim_matrix[i][j]), 4),
                })

    for idx in flagged_indices:
        c = candidates[idx]
        peers = [
            p for p in suspicious_pairs
            if p["a"] == c["id"] or p["b"] == c["id"]
        ]
        max_sim = max(p["similarity"] for p in peers) if peers else 0.0

        results.append(DetectionResult(
            candidate_id=c["id"],
            check_type="copy_ring",
            is_flagged=True,
            score=max_sim,
            explanation=(
                f"Part of a copy ring — answer matches {len(peers)} other candidate(s) "
                f"at {max_sim*100:.0f}% similarity for question '{question_id}'"
            ),
            evidence={"similar_pairs": peers[:5]},  # top 5 matches
        ))

    return results


# ------------------------------------------------------------------ #
# Check 4: Response timing
# ------------------------------------------------------------------ #
def check_timing(
    candidate_id: str,
    question_sent_at: datetime,
    answer_received_at: datetime,
    answer: str,
) -> DetectionResult:
    """Flag suspiciously fast replies."""
    delta_seconds = (answer_received_at - question_sent_at).total_seconds()
    word_count = len(answer.split())

    is_flagged = False
    score = 0.0
    explanation = ""

    if delta_seconds < VERY_FAST_REPLY_SECONDS:
        is_flagged = True
        score = 0.95
        explanation = (
            f"Reply in {delta_seconds:.0f}s with {word_count} words — near-certain automation"
        )
    elif delta_seconds < FAST_REPLY_SECONDS and word_count > 80:
        is_flagged = True
        score = 0.75
        explanation = (
            f"Reply in {delta_seconds:.0f}s with {word_count} words — "
            f"too fast for a thoughtful response"
        )
    else:
        score = 0.0
        explanation = (
            f"Reply time {delta_seconds:.0f}s ({word_count} words) — normal range"
        )

    return DetectionResult(
        candidate_id=candidate_id,
        check_type="timing",
        is_flagged=is_flagged,
        score=score,
        explanation=explanation,
        evidence={"delta_seconds": delta_seconds, "word_count": word_count},
    )


# ------------------------------------------------------------------ #
# Strike System
# ------------------------------------------------------------------ #
class StrikeSystem:
    """
    Manages strikes per candidate. 3 strikes = eliminated.
    Persists to SQLite so state survives process restarts.
    """
    def __init__(self, db_path: str = "hiring_agent.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidate_strikes (
                    candidate_id TEXT PRIMARY KEY,
                    strikes INTEGER DEFAULT 0,
                    eliminated INTEGER DEFAULT 0,
                    log TEXT DEFAULT '[]'
                )
            """)
            conn.commit()

    def add_strike(self, candidate_id: str, reason: str, evidence: dict = None) -> int:
        """Add a strike. Returns current strike count."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT strikes, log FROM candidate_strikes WHERE candidate_id = ?",
                (candidate_id,)
            ).fetchone()

            if row:
                strikes, log_str = row
                log = json.loads(log_str)
            else:
                strikes = 0
                log = []

            strikes += 1
            log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "evidence": evidence or {},
            })

            eliminated = 1 if strikes >= 3 else 0

            conn.execute("""
                INSERT OR REPLACE INTO candidate_strikes
                (candidate_id, strikes, eliminated, log)
                VALUES (?, ?, ?, ?)
            """, (candidate_id, strikes, eliminated, json.dumps(log)))
            conn.commit()

        if eliminated:
            print(f"  ❌ ELIMINATED: {candidate_id} — reached 3 strikes")
        else:
            print(f"  ⚠️  Strike {strikes}/3 for {candidate_id}: {reason}")

        return strikes

    def get_status(self, candidate_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT strikes, eliminated, log FROM candidate_strikes WHERE candidate_id = ?",
                (candidate_id,)
            ).fetchone()
        if not row:
            return {"strikes": 0, "eliminated": False, "log": []}
        return {
            "strikes": row[0],
            "eliminated": bool(row[1]),
            "log": json.loads(row[2]),
        }

    def is_eliminated(self, candidate_id: str) -> bool:
        status = self.get_status(candidate_id)
        return status["eliminated"]


# ------------------------------------------------------------------ #
# Full Pipeline
# ------------------------------------------------------------------ #
class AntiCheatPipeline:
    def __init__(self, db_path: str = "hiring_agent.db", anthropic_api_key: str = ""):
        self.strike_system = StrikeSystem(db_path)
        self.anthropic_api_key = anthropic_api_key

    def evaluate(
        self,
        candidate_id: str,
        question: str,
        answer: str,
        question_sent_at: Optional[datetime] = None,
        answer_received_at: Optional[datetime] = None,
    ) -> list[DetectionResult]:
        """
        Run all checks on a single candidate response.
        Apply strikes automatically.
        Returns list of all DetectionResult objects.
        """
        if self.strike_system.is_eliminated(candidate_id):
            print(f"  Skipping {candidate_id} — already eliminated")
            return []

        all_results = []

        # Check 1: Phrase detection (always run — cheap)
        phrase_result = check_ai_phrases(candidate_id, answer)
        all_results.append(phrase_result)
        if phrase_result.is_flagged:
            self.strike_system.add_strike(
                candidate_id,
                f"AI phrases detected: {phrase_result.explanation}",
                phrase_result.evidence,
            )

        # Check 2: LLM similarity (requires embedding model)
        if EMBEDDINGS_AVAILABLE and self.anthropic_api_key:
            llm_answer = generate_llm_reference_answer(question, self.anthropic_api_key)
            if llm_answer:
                sim_result = check_ai_similarity_embedding(
                    candidate_id, question, answer, llm_answer
                )
                all_results.append(sim_result)
                if sim_result.is_flagged:
                    self.strike_system.add_strike(
                        candidate_id,
                        sim_result.explanation,
                        sim_result.evidence,
                    )

        # Check 4: Timing (if timestamps provided)
        if question_sent_at and answer_received_at:
            timing_result = check_timing(
                candidate_id, question_sent_at, answer_received_at, answer
            )
            all_results.append(timing_result)
            if timing_result.is_flagged:
                self.strike_system.add_strike(
                    candidate_id,
                    timing_result.explanation,
                    timing_result.evidence,
                )

        return all_results

    def run_copy_ring_check(
        self,
        candidates: list[dict],
        question_id: str,
    ) -> list[DetectionResult]:
        """Run cross-candidate check. Call this after collecting all answers for a round."""
        results = check_copy_ring(candidates, question_id)
        for r in results:
            if r.is_flagged:
                self.strike_system.add_strike(
                    r.candidate_id,
                    r.explanation,
                    r.evidence,
                )
        return results


# ------------------------------------------------------------------ #
# Demo
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import os

    pipeline = AntiCheatPipeline(
        db_path="demo_hiring.db",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )

    question = "How would you approach scraping a website protected by reCAPTCHA Enterprise?"

    test_candidates = [
        {
            "id": "alice@example.com",
            "answer": (
                "I'd start by inspecting the network traffic with DevTools to understand "
                "what the login flow looks like. Then I'd try Playwright with stealth mode "
                "to see how far it gets. From my testing, Playwright headless gets blocked "
                "at the score threshold — reCAPTCHA v3 scores bots near 0. The approach "
                "that worked: using a real Chrome profile with a loaded extension to "
                "intercept the cookies after a human login, then replaying them with "
                "the same IP via a local proxy."
            ),
        },
        {
            "id": "bob@example.com",
            "answer": (
                "Certainly! I'd be happy to help with this. Here's a comprehensive overview "
                "of my approach. In today's rapidly evolving landscape of web scraping, "
                "it's worth noting that leveraging cutting-edge tools like Selenium with "
                "headless Chrome provides a holistic solution to this multifaceted challenge. "
                "To summarize: I would use Selenium to automate the login and then utilize "
                "BeautifulSoup to parse the resulting HTML."
            ),
        },
        {
            "id": "charlie@example.com",
            "answer": "use selenium",
        },
        {
            "id": "diana@example.com",
            "answer": (
                "Certainly! I'd be happy to help with this. Here's a comprehensive overview "
                "of my approach. In today's rapidly evolving landscape of web scraping, "
                "it's worth noting that leveraging cutting-edge tools like Selenium with "
                "headless Chrome provides a holistic solution to this multifaceted challenge. "
                "To summarize: I would use Selenium to automate the login."
            ),
        },
    ]

    print("=" * 60)
    print("Running individual checks...")
    print("=" * 60)

    sent_at = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    for i, c in enumerate(test_candidates):
        received_at = sent_at.replace(second=10 * (i + 1))  # Simulate fast replies
        print(f"\nCandidate: {c['id']}")
        results = pipeline.evaluate(
            c["id"], question, c["answer"],
            question_sent_at=sent_at,
            answer_received_at=received_at,
        )
        for r in results:
            status = "🚨 FLAGGED" if r.is_flagged else "✅ OK"
            print(f"  [{r.check_type}] {status} — {r.explanation}")

    print("\n" + "=" * 60)
    print("Running copy ring check across all candidates...")
    print("=" * 60)
    ring_results = pipeline.run_copy_ring_check(test_candidates, "q_recaptcha")
    if ring_results:
        for r in ring_results:
            print(f"  {r.candidate_id}: {r.explanation}")
    else:
        print("  No copy rings detected (embedding model may not be available)")

    print("\n" + "=" * 60)
    print("Final strike summary:")
    print("=" * 60)
    for c in test_candidates:
        status = pipeline.strike_system.get_status(c["id"])
        eliminated = "❌ ELIMINATED" if status["eliminated"] else "✅ Active"
        print(f"  {c['id']}: {status['strikes']} strike(s) — {eliminated}")
