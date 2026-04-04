"""
Component 3: ENGAGEMENT — Multi-Round Email Conversations

- Monitors Gmail inbox for candidate replies
- Tracks threads per candidate
- Generates contextual replies using Claude
- Manages conversation state in SQLite
- Handles 50+ concurrent conversations without mixing threads
"""

import os
import json
import time
import base64
import sqlite3
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Gmail API
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    print("Warning: google-api-python-client not installed.")
    print("  Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. pip install anthropic")

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]

POLL_INTERVAL_SECONDS = 120   # Check Gmail every 2 minutes
MAX_ROUNDS = 3                 # Max email rounds per candidate
LABEL_NAME = "GenoTekHiring"   # Gmail label to track candidate emails


# ------------------------------------------------------------------ #
# Conversation prompts per round
# ------------------------------------------------------------------ #
ROUND_PROMPTS = {
    1: """You are a senior technical recruiter at GenoTek, a company building autonomous AI agents.
You are conducting round 1 of an async email interview.

Candidate: {candidate_name}
Role: AI Agent Developer
Their application score: {score}/100 (tier: {tier})

Their answer to our opening question:
---
{candidate_answer}
---

Write a concise reply email (150–250 words) that:
1. Opens with ONE specific observation about something they actually said (name it explicitly)
2. Asks ONE targeted follow-up question that probes deeper on their specific claim
3. If they mentioned a tool/approach, ask about a specific failure mode or edge case they'd need to handle
4. If their answer was vague or generic, ask them to be concrete (e.g., "You said you'd use machine learning — which model, trained on what data, evaluated how?")
5. Ends with the next question for Round 2

Do NOT:
- Use generic openers like "Thank you for your response"
- Repeat what they said back to them
- Use bullet points in the email body
- Sound like a template

Sign as: GenoTek Hiring Team
""",

    2: """You are a senior technical recruiter at GenoTek.
This is round 2 of an async email interview.

Candidate: {candidate_name}
Previous conversation summary: {conversation_summary}

Their Round 2 answer:
---
{candidate_answer}
---

Write a reply (150–200 words) that:
1. Acknowledges ONE specific technical detail they gave (or calls out a gap if they were still vague)
2. Gives them a concrete mini-problem to solve in Round 3:
   "We have a CSV with 1,140 applicants. Name, email, GitHub link (many empty), 
   and answers to 2 screening questions. Write Python code that scores them 0–100 
   and outputs a ranked CSV with a 'tier' column. Send us the code."
3. Makes clear this is a REAL task, not hypothetical

Do NOT be formal or verbose. This should sound like a real person.
Sign as: GenoTek Hiring Team
""",

    3: """You are a senior technical recruiter at GenoTek.
This is the final evaluation round.

Candidate: {candidate_name}

They submitted code/solution for our scoring challenge.
Their submission:
---
{candidate_answer}
---

Evaluate and reply (200–300 words):
1. Note ONE specific thing their code does well (be specific — line-level if possible)
2. Note ONE gap or bug (if any — be direct, not diplomatic)
3. If their code actually runs and solves the problem → move them forward:
   "We'd like to schedule a 30-minute live call. Reply with 3 time slots this week."
4. If their code is incomplete but shows real thinking → give ONE specific improvement request
5. If their submission is clearly ChatGPT-generated or doesn't work → decline professionally

Sign as: GenoTek Hiring Team
"""
}


# ------------------------------------------------------------------ #
# Database
# ------------------------------------------------------------------ #
class ConversationDB:
    def __init__(self, db_path: str = "hiring_agent.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    candidate_id TEXT PRIMARY KEY,
                    candidate_name TEXT,
                    email TEXT,
                    thread_id TEXT,
                    current_round INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    score REAL DEFAULT 0,
                    tier TEXT DEFAULT 'Review',
                    last_message_at TEXT,
                    history TEXT DEFAULT '[]',
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_message_ids (
                    message_id TEXT PRIMARY KEY,
                    processed_at TEXT
                )
            """)
            conn.commit()

    def upsert_conversation(self, candidate_id: str, **kwargs):
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT candidate_id FROM conversations WHERE candidate_id = ?",
                (candidate_id,)
            ).fetchone()

            if not existing:
                kwargs["created_at"] = datetime.now(timezone.utc).isoformat()
                kwargs["candidate_id"] = candidate_id
                if "history" not in kwargs:
                    kwargs["history"] = "[]"
                cols = ", ".join(kwargs.keys())
                placeholders = ", ".join("?" * len(kwargs))
                conn.execute(
                    f"INSERT INTO conversations ({cols}) VALUES ({placeholders})",
                    list(kwargs.values())
                )
            else:
                set_clause = ", ".join(f"{k} = ?" for k in kwargs)
                conn.execute(
                    f"UPDATE conversations SET {set_clause} WHERE candidate_id = ?",
                    list(kwargs.values()) + [candidate_id]
                )
            conn.commit()

    def get_conversation(self, candidate_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM conversations WHERE candidate_id = ?",
                (candidate_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["history"] = json.loads(d["history"])
        return d

    def get_by_thread_id(self, thread_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM conversations WHERE thread_id = ?",
                (thread_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["history"] = json.loads(d["history"])
        return d

    def append_message(self, candidate_id: str, role: str, content: str):
        conv = self.get_conversation(candidate_id)
        if not conv:
            return
        history = conv["history"]
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET history = ?, last_message_at = ? WHERE candidate_id = ?",
                (json.dumps(history), datetime.now(timezone.utc).isoformat(), candidate_id)
            )
            conn.commit()

    def mark_message_processed(self, message_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO processed_message_ids VALUES (?, ?)",
                (message_id, datetime.now(timezone.utc).isoformat())
            )
            conn.commit()

    def is_message_processed(self, message_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_message_ids WHERE message_id = ?",
                (message_id,)
            ).fetchone()
        return row is not None

    def get_active_conversations(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM conversations WHERE status = 'active'"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["history"] = json.loads(d["history"])
            result.append(d)
        return result


# ------------------------------------------------------------------ #
# Gmail helpers
# ------------------------------------------------------------------ #
def get_gmail_service(credentials_path: str = "credentials.json", token_path: str = "token.json"):
    """Authenticate and return a Gmail service object."""
    if not GMAIL_AVAILABLE:
        raise RuntimeError("google-api-python-client not installed")

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def extract_email_body(payload: dict) -> str:
    """Recursively extract plain text body from Gmail message payload."""
    body = ""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    elif "parts" in payload:
        for part in payload["parts"]:
            body += extract_email_body(part)
    return body


def get_header(headers: list, name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def strip_quoted_reply(text: str) -> str:
    """Remove quoted previous messages from email body."""
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        if line.startswith(">"):
            break
        if re.match(r"On .* wrote:", line):
            break
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def send_email(
    service,
    to: str,
    subject: str,
    body: str,
    thread_id: Optional[str] = None,
    in_reply_to: Optional[str] = None,
) -> dict:
    """Send an email, optionally as a reply in an existing thread."""
    msg = MIMEMultipart()
    msg["To"] = to
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    msg.attach(MIMEText(body, "plain"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    body_payload = {"raw": raw}
    if thread_id:
        body_payload["threadId"] = thread_id

    result = service.users().messages().send(userId="me", body=body_payload).execute()
    return result


# ------------------------------------------------------------------ #
# Response generation
# ------------------------------------------------------------------ #
def generate_reply(
    candidate_name: str,
    candidate_answer: str,
    current_round: int,
    conversation_history: list,
    score: float,
    tier: str,
    api_key: str,
) -> str:
    """Generate a contextual reply using Claude."""
    if not ANTHROPIC_AVAILABLE:
        return (
            f"Hi {candidate_name},\n\nThank you for your response. "
            "We'll be in touch shortly.\n\nGenoTek Hiring Team"
        )

    # Summarize conversation history for context
    conversation_summary = ""
    if conversation_history:
        summary_parts = []
        for msg in conversation_history[-4:]:  # Last 4 messages
            role = "Candidate" if msg["role"] == "candidate" else "GenoTek"
            summary_parts.append(f"{role}: {msg['content'][:300]}...")
        conversation_summary = "\n".join(summary_parts)

    prompt_template = ROUND_PROMPTS.get(current_round, ROUND_PROMPTS[3])
    system_prompt = prompt_template.format(
        candidate_name=candidate_name,
        candidate_answer=candidate_answer,
        conversation_summary=conversation_summary,
        score=score,
        tier=tier,
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=system_prompt,
            messages=[
                {"role": "user", "content": "Write the reply email now."}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"  LLM reply generation failed: {e}")
        return (
            f"Hi {candidate_name},\n\nThank you for your response — "
            "we'll follow up shortly.\n\nGenoTek Hiring Team"
        )


# ------------------------------------------------------------------ #
# Main polling loop
# ------------------------------------------------------------------ #
class EmailAgent:
    def __init__(
        self,
        db_path: str = "hiring_agent.db",
        anthropic_api_key: str = "",
        gmail_credentials_path: str = "credentials.json",
        gmail_token_path: str = "token.json",
    ):
        self.db = ConversationDB(db_path)
        self.anthropic_api_key = anthropic_api_key
        self.gmail_credentials_path = gmail_credentials_path
        self.gmail_token_path = gmail_token_path
        self.service = None

    def connect_gmail(self):
        self.service = get_gmail_service(
            self.gmail_credentials_path,
            self.gmail_token_path,
        )
        print("Gmail connected.")

    def start_conversation(
        self,
        candidate_id: str,
        candidate_name: str,
        candidate_email: str,
        opening_question: str,
        score: float = 0.0,
        tier: str = "Review",
    ):
        """Initiate the first email to a candidate."""
        self.db.upsert_conversation(
            candidate_id,
            candidate_name=candidate_name,
            email=candidate_email,
            current_round=1,
            score=score,
            tier=tier,
            status="active",
        )

        subject = "AI Agent Developer Role — Technical Question"
        body = (
            f"Hi {candidate_name},\n\n"
            f"{opening_question}\n\n"
            "Please reply directly to this email. Take the time you need — "
            "we value thoughtful answers over fast ones.\n\n"
            "GenoTek Hiring Team"
        )

        if self.service:
            result = send_email(self.service, candidate_email, subject, body)
            thread_id = result.get("threadId")
            self.db.upsert_conversation(candidate_id, thread_id=thread_id)
            self.db.append_message(candidate_id, "recruiter", body)
            print(f"  Sent opening email to {candidate_name} ({candidate_email})")
        else:
            print(f"  [DRY RUN] Would send opening email to {candidate_name}")
            print(f"  Subject: {subject}")
            print(f"  Body preview: {body[:200]}...")

    def process_new_replies(self):
        """Check inbox for new candidate replies and respond."""
        if not self.service:
            print("Gmail not connected. Call connect_gmail() first.")
            return

        # Fetch unread messages with our label (or from candidates we know)
        query = f"label:{LABEL_NAME} is:unread"
        results = self.service.users().messages().list(
            userId="me", q=query, maxResults=50
        ).execute()

        messages = results.get("messages", [])
        print(f"  Found {len(messages)} unread message(s) to process")

        for msg_ref in messages:
            msg_id = msg_ref["id"]

            if self.db.is_message_processed(msg_id):
                continue

            # Fetch full message
            msg = self.service.users().messages().get(
                userId="me", messageId=msg_id, format="full"
            ).execute()

            thread_id = msg.get("threadId")
            payload = msg.get("payload", {})
            headers = payload.get("headers", [])

            sender = get_header(headers, "From")
            subject = get_header(headers, "Subject")
            message_id_header = get_header(headers, "Message-ID")

            # Extract email address from "Name <email>" format
            email_match = re.search(r'<(.+?)>', sender)
            candidate_email = email_match.group(1) if email_match else sender

            body = extract_email_body(payload)
            body = strip_quoted_reply(body)

            if not body.strip():
                self.db.mark_message_processed(msg_id)
                continue

            # Look up conversation by thread
            conv = self.db.get_by_thread_id(thread_id)
            if not conv:
                print(f"  Unknown thread {thread_id} from {candidate_email} — skipping")
                self.db.mark_message_processed(msg_id)
                continue

            candidate_id = conv["candidate_id"]
            candidate_name = conv["candidate_name"]
            current_round = conv["current_round"]

            if conv["status"] != "active":
                print(f"  Conversation with {candidate_name} is {conv['status']} — skipping")
                self.db.mark_message_processed(msg_id)
                continue

            print(f"\n  Processing reply from {candidate_name} (Round {current_round})")

            # Log candidate message
            self.db.append_message(candidate_id, "candidate", body)

            if current_round >= MAX_ROUNDS:
                # Final round — evaluate and decide
                status = "completed"
                self.db.upsert_conversation(candidate_id, status=status)
                print(f"  {candidate_name} has completed all rounds")
            else:
                # Generate contextual reply
                next_round = current_round + 1
                reply_body = generate_reply(
                    candidate_name=candidate_name,
                    candidate_answer=body,
                    current_round=next_round,
                    conversation_history=conv["history"],
                    score=conv["score"],
                    tier=conv["tier"],
                    api_key=self.anthropic_api_key,
                )

                # Send reply in same thread
                send_email(
                    self.service,
                    to=candidate_email,
                    subject=f"Re: {subject}",
                    body=reply_body,
                    thread_id=thread_id,
                    in_reply_to=message_id_header,
                )

                # Update DB
                self.db.append_message(candidate_id, "recruiter", reply_body)
                self.db.upsert_conversation(
                    candidate_id,
                    current_round=next_round,
                    last_message_at=datetime.now(timezone.utc).isoformat(),
                )
                print(f"  Sent Round {next_round} question to {candidate_name}")

            # Mark as read and processed
            self.service.users().messages().modify(
                userId="me",
                id=msg_id,
                body={"removeLabelIds": ["UNREAD"]},
            ).execute()
            self.db.mark_message_processed(msg_id)

    def run_forever(self):
        """Main polling loop. Runs 24/7."""
        print(f"Email agent started. Polling every {POLL_INTERVAL_SECONDS}s...")
        while True:
            try:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking inbox...")
                self.process_new_replies()
            except KeyboardInterrupt:
                print("\nAgent stopped.")
                break
            except Exception as e:
                print(f"  Error in polling loop: {e}")
                # Don't crash — just wait and retry
            time.sleep(POLL_INTERVAL_SECONDS)


# ------------------------------------------------------------------ #
# Demo (no Gmail credentials needed)
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    agent = EmailAgent(
        db_path="demo_hiring.db",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )

    print("=== Email Agent Demo (dry run — no Gmail) ===\n")

    # Simulate starting conversations with Fast-Track candidates
    agent.start_conversation(
        candidate_id="alice@example.com",
        candidate_name="Alice Sharma",
        candidate_email="alice@example.com",
        opening_question=(
            "We have 1,140 applicants on Internshala. The employer portal has no public API "
            "and the login page uses reCAPTCHA Enterprise. Walk us through exactly how you'd "
            "approach getting programmatic access to the applicant data. "
            "Be specific — what would you try first, and what would you do when it fails?"
        ),
        score=82.5,
        tier="Fast-Track",
    )

    # Simulate generating a reply to a candidate response
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n=== Generating sample Round 2 reply ===\n")
        reply = generate_reply(
            candidate_name="Alice Sharma",
            candidate_answer=(
                "I'd start with Playwright stealth mode to see what reCAPTCHA v3 returns. "
                "From there, I'd intercept the cookie after a manual login using a browser "
                "extension and replay it on the same IP. The IP binding is the main constraint "
                "— I'd keep one egress IP per session using a residential proxy."
            ),
            current_round=2,
            conversation_history=[],
            score=82.5,
            tier="Fast-Track",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        print(reply)
    else:
        print("\nSet ANTHROPIC_API_KEY to see LLM-generated replies.")

    print("\n=== Conversation DB state ===")
    conv = agent.db.get_conversation("alice@example.com")
    if conv:
        print(f"Candidate: {conv['candidate_name']}")
        print(f"Round: {conv['current_round']}")
        print(f"Status: {conv['status']}")
        print(f"History entries: {len(conv['history'])}")
