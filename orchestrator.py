"""
Component 6: INTEGRATION — Full Pipeline Orchestrator

Full data flow (Component 1 → 2 → 3 → 4 → 5 → back to 2):

  ATS Source (Greenhouse / Lever / Workable / CSV / Mock)
       │
       ▼  access.py (Component 1)
  candidates_raw.csv
       │
       ▼  scorer.py (Component 2)
  scored_candidates.csv
       │
       ▼  email_agent.py (Component 3)
  Gmail multi-round conversations
       │
       ▼  anti_cheat.py (Component 4)
  Strike system + elimination
       │
       ▼  learning_system.py (Component 5)
  Periodic LLM analysis → updated scoring weights → feeds back into Component 2

Run modes:
  python orchestrator.py --fetch                  # Fetch candidates (Component 1)
  python orchestrator.py --score input.csv        # Score a batch (Component 2)
  python orchestrator.py --full-run               # Fetch + Score + Engage in one shot
  python orchestrator.py --run                    # Start the live email polling loop
  python orchestrator.py --analyze                # Run analysis report manually
  python orchestrator.py --query "..."            # Query the knowledge base
  python orchestrator.py --status                 # Show pipeline status
"""

import argparse
import os
import json
import time
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from access import ATSRouter
from scorer import score_candidates
from anti_cheat import AntiCheatPipeline
from email_agent import EmailAgent
from learning_system import KnowledgeBase

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
DB_PATH = os.getenv("DB_PATH", "hiring_agent.db")
from llm_client import llm_complete, LLM_PROVIDER, llm_status
# Keys are loaded inside llm_client from environment variables.
# Set GEMINI_API_KEY (preferred) or ANTHROPIC_API_KEY before running.
GMAIL_CREDENTIALS = os.getenv("GMAIL_CREDENTIALS", "credentials.json")
GMAIL_TOKEN = os.getenv("GMAIL_TOKEN", "token.json")
CHECK_GITHUB = os.getenv("CHECK_GITHUB", "false").lower() == "true"

OPENING_QUESTION = (
    "We have 1,140 applicants on Internshala. The employer portal has no public API "
    "and the login page uses reCAPTCHA Enterprise (invisible v3). Walk us through "
    "exactly how you would approach getting programmatic access to the applicant data. "
    "Be specific — what would you try first, what would you do when it fails, "
    "and what have you actually tried (not just what you'd theoretically do)?"
)


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #
class HiringPipeline:
    def __init__(self):
        self.kb = KnowledgeBase(DB_PATH)
        self.anti_cheat = AntiCheatPipeline(
            db_path=DB_PATH,
            )
        self.email_agent = EmailAgent(
            db_path=DB_PATH,
            gmail_credentials_path=GMAIL_CREDENTIALS,
            gmail_token_path=GMAIL_TOKEN,
        )
        self._setup_signal_handlers()

    # ------------------------------------------------------------------ #
    # Step 0 (Component 1): Fetch candidates from ATS / CSV / Mock
    # ------------------------------------------------------------------ #
    def fetch_candidates(self, source=None, output_csv="candidates_raw.csv", limit=None, **kwargs):
        sep = "=" * 60
        src = source or os.getenv("ATS_SOURCE", "mock")
        print("\n" + sep)
        print("COMPONENT 1: Fetching candidates (source: " + src + ")")
        print(sep)
        router = ATSRouter(source=src, **kwargs)
        return router.fetch_and_save(output_csv=output_csv, db_path=DB_PATH, limit=limit)

    def full_run(self, source="mock", count=100):
        sep = "=" * 60
        print("\n" + sep)
        print("FULL PIPELINE: source=" + source + ", count=" + str(count))
        print("LLM provider : " + llm_status())
        print(sep)
        # Clear conversation state so candidates aren't skipped on re-runs
        import sqlite3 as _sq
        conn = _sq.connect(DB_PATH)
        conn.execute("DELETE FROM conversations")
        conn.execute("DELETE FROM processed_message_ids")
        conn.commit()
        conn.close()
        print("  [Reset] Cleared previous conversation state")
        raw_csv = self.fetch_candidates(source=source, output_csv="candidates_raw.csv", mock_count=count)
        scored_csv = "scored_candidates.csv"
        self.score_batch(raw_csv, scored_csv)
        self.engage_top_candidates(scored_csv, dry_run=True)
        self.print_status()


    def _setup_signal_handlers(self):
        """Graceful shutdown on Ctrl+C or SIGTERM."""
        def handle_shutdown(sig, frame):
            print("\n\nShutdown signal received. Finishing current task...")
            sys.exit(0)
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

    # ------------------------------------------------------------------ #
    # Step 1: Score a batch of applicants from CSV
    # ------------------------------------------------------------------ #
    def score_batch(self, input_csv: str, output_csv: str = "scored_candidates.csv"):
        print(f"\n{'='*60}")
        print(f"STEP 1: Scoring applicants from {input_csv}")
        print(f"{'='*60}")

        # Load adaptive weights from learning system
        weights = self.kb.get_current_weights()
        print(f"Using scoring weights (version {self.kb.get_current_weights()}): {weights}")

        scored_df = score_candidates(
            input_csv=input_csv,
            output_csv=output_csv,
            check_github_api=CHECK_GITHUB,
        )

        # Log all candidates to knowledge base
        for _, row in scored_df.iterrows():
            candidate_id = row["email"]
            self.kb.log_candidate(
                candidate_id=candidate_id,
                name=row["name"],
                email=row["email"],
                initial_score=row["score"],
                tier=row["tier"],
            )
            self.kb.log_interaction(
                candidate_id=candidate_id,
                interaction_type="scored",
                content=f"Score: {row['score']}, Tier: {row['tier']}, Flags: {row.get('flags', '')}",
                round_number=0,
                metadata={
                    "score": row["score"],
                    "tier": row["tier"],
                    "flags": row.get("flags", ""),
                },
            )

        # Trigger analysis if enough new candidates
        if self.kb.should_run_analysis() and LLM_PROVIDER != "none":
            self.kb.run_periodic_analysis("")

        print(f"\nScored {len(scored_df)} candidates. Output: {output_csv}")
        return scored_df

    # ------------------------------------------------------------------ #
    # Step 2: Send opening emails to Fast-Track + Standard candidates
    # ------------------------------------------------------------------ #
    def engage_top_candidates(
        self,
        scored_csv: str = "scored_candidates.csv",
        tiers_to_contact: list = None,
        dry_run: bool = True,
    ):
        if tiers_to_contact is None:
            tiers_to_contact = ["Fast-Track", "Standard"]

        import pandas as pd
        df = pd.read_csv(scored_csv)
        targets = df[df["tier"].isin(tiers_to_contact)]

        print(f"\n{'='*60}")
        print(f"STEP 2: Engaging {len(targets)} candidates (tiers: {tiers_to_contact})")
        print(f"{'='*60}")

        if not dry_run:
            try:
                self.email_agent.connect_gmail()
            except Exception as e:
                print(f"Gmail connection failed: {e}")
                print("Falling back to dry run mode.")
                dry_run = True

        skipped_dup = 0
        skipped_elim = 0
        sent = 0
        for _, row in targets.iterrows():
            candidate_id = row["email"]

            # Skip eliminated candidates
            if self.anti_cheat.strike_system.is_eliminated(candidate_id):
                skipped_elim += 1
                continue

            # Check if already contacted
            existing = self.email_agent.db.get_conversation(candidate_id)
            if existing:
                skipped_dup += 1
                continue

            self.email_agent.start_conversation(
                candidate_id=candidate_id,
                candidate_name=row["name"],
                candidate_email=row["email"],
                opening_question=OPENING_QUESTION,
                score=row["score"],
                tier=row["tier"],
            )

            self.kb.log_interaction(
                candidate_id=candidate_id,
                interaction_type="email_sent",
                content=OPENING_QUESTION,
                round_number=1,
            )
            sent += 1

        # Clean summary
        print(f"  Sent/queued : {sent}")
        if sent:
            # Show first 5 as preview only
            preview = targets[targets["email"].apply(
                lambda e: not self.anti_cheat.strike_system.is_eliminated(e)
            )].head(5)
            for _, r in preview.iterrows():
                print(f"    ↳ {r['name']} <{r['email']}> | {r['tier']} {r['score']}")
            if sent > 5:
                print(f"    ... and {sent - 5} more")
        if skipped_dup:
            print(f"  Already contacted: {skipped_dup} (skipped)")
        if skipped_elim:
            print(f"  Eliminated  : {skipped_elim} (skipped)")

    # ------------------------------------------------------------------ #
    # Step 3: Process incoming replies (called in polling loop)
    # ------------------------------------------------------------------ #
    def process_replies(self):
        if not self.email_agent.service:
            return

        self.email_agent.process_new_replies()

        # After processing replies, run anti-cheat on any new answers
        # (Anti-cheat is also called inside email_agent for timing checks;
        # this handles the copy-ring check which needs ALL answers together)
        self._run_copy_ring_check_if_ready()

    def _run_copy_ring_check_if_ready(self):
        """Run copy-ring detection after each round when we have enough answers."""
        convs = self.email_agent.db.get_active_conversations()

        # Group by current round
        round_groups = {}
        for conv in convs:
            r = conv["current_round"]
            if r not in round_groups:
                round_groups[r] = []
            # Get their last received answer
            last_answer = ""
            for msg in reversed(conv["history"]):
                if msg["role"] == "candidate":
                    last_answer = msg["content"]
                    break
            if last_answer:
                round_groups[r].append({
                    "id": conv["candidate_id"],
                    "answer": last_answer,
                })

        for round_num, candidates in round_groups.items():
            if len(candidates) >= 3:  # Need at least 3 to detect a ring
                results = self.anti_cheat.run_copy_ring_check(
                    candidates, f"round_{round_num}"
                )
                for r in results:
                    if r.is_flagged:
                        self.kb.log_interaction(
                            candidate_id=r.candidate_id,
                            interaction_type="flagged",
                            content=r.explanation,
                            round_number=round_num,
                            metadata=r.evidence,
                        )

    # ------------------------------------------------------------------ #
    # Run forever
    # ------------------------------------------------------------------ #
    def run_live(self):
        """Start the live 24/7 polling loop."""
        print(f"\n{'='*60}")
        print("LIVE MODE: Hiring pipeline running 24/7")
        print(f"DB: {DB_PATH}")
        print(f"Gmail polling every 120s")
        print(f"{'='*60}\n")

        try:
            self.email_agent.connect_gmail()
        except Exception as e:
            print(f"Gmail setup failed: {e}")
            print("Ensure credentials.json is in place and you've run the OAuth flow.")
            sys.exit(1)

        while True:
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{now}] Polling for new replies...")
                self.process_replies()

                # Periodic analysis check
                if self.kb.should_run_analysis() and LLM_PROVIDER != "none":
                    print("  Triggering periodic analysis...")
                    self.kb.run_periodic_analysis("")

                self._print_live_status()
                time.sleep(120)

            except Exception as e:
                print(f"  Pipeline error (will retry): {e}")
                time.sleep(30)

    # ------------------------------------------------------------------ #
    # Status & reporting
    # ------------------------------------------------------------------ #
    def _print_live_status(self):
        convs = self.email_agent.db.get_active_conversations()
        count = self.kb.get_candidate_count()
        weights = self.kb.get_current_weights()

        print(f"\n  Status: {count} total candidates, {len(convs)} active conversations")
        print(f"  Current scoring weights: {weights}")

        if convs:
            by_round = {}
            for c in convs:
                r = c["current_round"]
                by_round[r] = by_round.get(r, 0) + 1
            print(f"  Conversations by round: {by_round}")

    def print_status(self):
        sep = "=" * 60
        print("")
        print(sep)
        print("PIPELINE STATUS")
        print(sep)

        total = self.kb.get_candidate_count()
        weights = self.kb.get_current_weights()
        selenium_pct = self.kb.get_selenium_first_approach_pct()

        print("Database         : " + DB_PATH)
        print("Total candidates : " + str(total))
        print("Scoring weights  :")
        for k, v in weights.items():
            print("  " + str(k) + " = " + str(v))
        print("Selenium 1st     : " + str(selenium_pct) + "%")

        candidates = self.kb.get_all_candidates_summary()
        tier_counts = {}
        outcome_counts = {}
        for c in candidates:
            tier_counts[c["tier"]] = tier_counts.get(c["tier"], 0) + 1
            outcome_counts[c["final_outcome"]] = outcome_counts.get(c["final_outcome"], 0) + 1

        print("")
        print("Tier breakdown   : " + str(tier_counts))
        print("Outcome breakdown: " + str(outcome_counts))

        top3 = self.kb.get_top_candidates_by_originality(3)
        if top3:
            print("")
            print("Top 3 by originality:")
            for c in top3:
                print("  " + c["name"] + " — score=" + str(c["initial_score"]) + " rounds=" + str(c["rounds_completed"]))

        phrases = self.kb.get_top_patterns("ai_phrase", 5)
        if phrases:
            print("")
            print("Top AI phrases detected:")
            for p in phrases:
                print("  [" + str(p["frequency"]) + "x] " + p["value"])
        print("")

# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="GenoTek Autonomous Hiring Agent")
    parser.add_argument("--fetch", action="store_true", help="Fetch candidates from ATS (set ATS_SOURCE env var)")
    parser.add_argument("--source", default=None, choices=["greenhouse","lever","workable","csv","mock"], help="ATS source for --fetch")
    parser.add_argument("--count", type=int, default=100, help="Mock candidate count (default: 100)")
    parser.add_argument("--raw-output", default="candidates_raw.csv", help="Output CSV from --fetch")
    parser.add_argument("--full-run", action="store_true", help="Full pipeline: fetch + score + engage")
    parser.add_argument("--score", metavar="CSV", help="Score applicants from a CSV file")
    parser.add_argument("--output", default="scored_candidates.csv", help="Output CSV path")
    parser.add_argument("--engage", metavar="CSV", help="Send opening emails to top candidates")
    parser.add_argument("--tiers", nargs="+", default=["Fast-Track", "Standard"],
                        help="Which tiers to engage (default: Fast-Track Standard)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually send emails")
    parser.add_argument("--run", action="store_true", help="Start live 24/7 polling loop")
    parser.add_argument("--analyze", action="store_true", help="Run analysis report now")
    parser.add_argument("--query", metavar="QUESTION", help="Query the knowledge base")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    pipeline = HiringPipeline()

    if args.fetch:
        pipeline.fetch_candidates(
            source=args.source,
            output_csv=args.raw_output,
            mock_count=args.count,
        )

    if args.full_run:
        pipeline.full_run(source=args.source or "mock", count=args.count)

    if args.score:
        if not Path(args.score).exists():
            print(f"Error: {args.score} not found")
            sys.exit(1)
        pipeline.score_batch(args.score, args.output)

    if args.engage:
        pipeline.engage_top_candidates(
            scored_csv=args.engage,
            tiers_to_contact=args.tiers,
            dry_run=args.dry_run,
        )

    if args.run:
        pipeline.run_live()

    if args.analyze:
        if LLM_PROVIDER == "none":
            print("Set GEMINI_API_KEY for LLM-powered analysis.")
        else:
            insights = pipeline.kb.run_periodic_analysis("")
            print(json.dumps(insights, indent=2) if insights else "No analysis triggered (not enough new candidates).")

    if args.query:
        if LLM_PROVIDER == "none":
            print("Set GEMINI_API_KEY to use natural language queries.")
        else:
            answer = pipeline.kb.query(args.query, "")
            print(f"\nAnswer: {answer}")

    if args.status:
        pipeline.print_status()


if __name__ == "__main__":
    main()
