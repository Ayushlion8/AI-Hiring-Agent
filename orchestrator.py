"""
Component 6: INTEGRATION — Full Pipeline Orchestrator

Data flow:
  CSV/Input → Scorer → EmailAgent → AntiCheat → KnowledgeBase (learning loop)

Run modes:
  python orchestrator.py --score input.csv        # Score a batch of applicants
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

from scorer import score_candidates
from anti_cheat import AntiCheatPipeline
from email_agent import EmailAgent
from learning_system import KnowledgeBase

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
DB_PATH = os.getenv("DB_PATH", "hiring_agent.db")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
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
            anthropic_api_key=ANTHROPIC_API_KEY,
        )
        self.email_agent = EmailAgent(
            db_path=DB_PATH,
            anthropic_api_key=ANTHROPIC_API_KEY,
            gmail_credentials_path=GMAIL_CREDENTIALS,
            gmail_token_path=GMAIL_TOKEN,
        )
        self._setup_signal_handlers()

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
        if self.kb.should_run_analysis() and ANTHROPIC_API_KEY:
            self.kb.run_periodic_analysis(ANTHROPIC_API_KEY)

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

        for _, row in targets.iterrows():
            candidate_id = row["email"]

            # Skip eliminated candidates
            if self.anti_cheat.strike_system.is_eliminated(candidate_id):
                print(f"  Skipping {row['name']} — eliminated")
                continue

            # Check if already contacted
            existing = self.email_agent.db.get_conversation(candidate_id)
            if existing:
                print(f"  Skipping {row['name']} — already in conversation (round {existing['current_round']})")
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
                if self.kb.should_run_analysis() and ANTHROPIC_API_KEY:
                    print("  Triggering periodic analysis...")
                    self.kb.run_periodic_analysis(ANTHROPIC_API_KEY)

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
        print(f"\n{'='*60}")
        print("PIPELINE STATUS")
        print(f"{'='*60}")
        print(f"Database: {DB_PATH}")
        print(f"Total candidates: {self.kb.get_candidate_count()}")
        print(f"Current weights: {json.dumps(self.kb.get_current_weights(), indent=2)}")
        print(f"Selenium first-approach rate: {self.kb.get_selenium_first_approach_pct()}%")

        candidates = self.kb.get_all_candidates_summary()
        tier_counts = {}
        outcome_counts = {}
        for c in candidates:
            tier_counts[c["tier"]] = tier_counts.get(c["tier"], 0) + 1
            outcome_counts[c["final_outcome"]] = outcome_counts.get(c["final_outcome"], 0) + 1

        print(f"\nTier breakdown: {tier_counts}")
        print(f"Outcome breakdown: {outcome_counts}")

        top3 = self.kb.get_top_candidates_by_originality(3)
        if top3:
            print(f"\nTop 3 original candidates:")
            for c in top3:
                print(f"  {c['name']} — score: {c['initial_score']}, rounds: {c['rounds_completed']}")

        print(f"\nTop AI phrases detected:")
        for p in self.kb.get_top_patterns("ai_phrase", 5):
            print(f"  '{p['value']}' — seen {p['frequency']}x")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="GenoTek Autonomous Hiring Agent")
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
        if not ANTHROPIC_API_KEY:
            print("Set ANTHROPIC_API_KEY for LLM-powered analysis.")
        else:
            insights = pipeline.kb.run_periodic_analysis(ANTHROPIC_API_KEY)
            print(json.dumps(insights, indent=2) if insights else "No analysis triggered (not enough new candidates).")

    if args.query:
        if not ANTHROPIC_API_KEY:
            print("Set ANTHROPIC_API_KEY to use natural language queries.")
        else:
            answer = pipeline.kb.query(args.query, ANTHROPIC_API_KEY)
            print(f"\nAnswer: {answer}")

    if args.status:
        pipeline.print_status()


if __name__ == "__main__":
    main()
