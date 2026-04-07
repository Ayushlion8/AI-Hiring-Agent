"""
Component 1: ACCESS — Multi-Source ATS Adapter
===============================================

RESEARCH FINDINGS (what actually works legally):
─────────────────────────────────────────────────
PUBLIC (no auth, free):
  • Greenhouse Job Board API  → jobs only, no candidates
  • Lever Postings API        → jobs only, no candidates  
  • Workable Widget API       → jobs only, no candidates

PRIVATE (employer API key required — full candidate data):
  • Greenhouse Harvest API    → full candidate + answers + resume + score
  • Lever Full API            → full opportunity + custom questions + resume
  • Workable Full API         → full candidates + stages + answers

REALITY: Candidate data is NEVER publicly exposed.
You need the employer's API key for your own ATS account.
GenoTek IS the employer → they have the key → this is the correct approach.

This module provides:
  1. GreenhouseClient   — full Harvest API candidate pull
  2. LeverClient        — full opportunities + custom form answers
  3. WorkableClient     — full candidates + job stages
  4. MockATSClient      — realistic synthetic data for demo/dev
  5. ATSRouter          — auto-detect source, normalize to common schema
  6. CSVFallback        — load from Internshala/any CSV export

Output schema (feeds directly into scorer.py):
  name, email, github, skills, answer_1, answer_2,
  source_platform, applied_at, resume_url, raw_json
"""

import os
import csv
import json
import time
import random
import sqlite3
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Iterator
from pathlib import Path


# ─────────────────────────────────────────────
# Canonical Candidate Schema
# ─────────────────────────────────────────────
@dataclass
class Candidate:
    """Normalized candidate record. Every ATS source maps to this."""
    candidate_id: str
    name: str
    email: str
    github: str = ""
    linkedin: str = ""
    skills: str = ""
    answer_1: str = ""           # First screening question answer
    answer_2: str = ""           # Second screening question answer
    resume_url: str = ""
    applied_at: str = ""
    source_platform: str = ""    # "greenhouse" | "lever" | "workable" | "csv" | "mock"
    job_title: str = ""
    job_id: str = ""
    stage: str = ""              # "applied" | "screening" | "interview" | "offer"
    raw_json: str = "{}"         # Full raw API response for audit trail


# ─────────────────────────────────────────────
# Rate-limit-safe HTTP helper
# ─────────────────────────────────────────────
class RateLimitedSession:
    """Wraps requests.Session with retry + rate limit handling."""

    def __init__(self, requests_per_minute: int = 60):
        self.session = requests.Session()
        self.min_interval = 60.0 / requests_per_minute
        self._last_call = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        self._throttle()
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=15, **kwargs)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    print(f"  Rate limited. Sleeping {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                return resp
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed after 3 attempts: {url}")

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()


# ─────────────────────────────────────────────
# 1. Greenhouse Harvest API Client
# ─────────────────────────────────────────────
class GreenhouseClient:
    """
    Uses Greenhouse Harvest API (employer-side, requires API key).
    Docs: https://developers.greenhouse.io/harvest.html

    API key: Settings → Dev Center → API Credential Management
    Scope needed: candidates, applications, jobs (read-only is fine)

    Rate limit: 50 req/10s per API key (we stay well under this)
    Pagination: Link header with rel="next"
    """

    BASE = "https://harvest.greenhouse.io/v1"

    def __init__(self, api_key: str, job_id: Optional[str] = None):
        import base64
        self.api_key = api_key
        self.job_id = job_id
        encoded = base64.b64encode(f"{api_key}:".encode()).decode()
        self.http = RateLimitedSession(requests_per_minute=40)
        self.http.session.headers.update({
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        })

    def _get_paginated(self, endpoint: str, params: dict = None) -> Iterator[dict]:
        """Yield every item across all pages."""
        url = f"{self.BASE}/{endpoint}"
        params = params or {}
        params.setdefault("per_page", 100)

        while url:
            resp = self.http.get(url, params=params)
            resp.raise_for_status()
            items = resp.json()
            if isinstance(items, list):
                yield from items
            elif isinstance(items, dict):
                yield from items.get("applications", items.get("candidates", [items]))

            # Parse Link header for next page
            link_header = resp.headers.get("Link", "")
            next_url = None
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    next_url = part.split(";")[0].strip().strip("<>")
                    break
            url = next_url
            params = {}  # params are encoded in next URL

    def _get_application_answers(self, application_id: int) -> tuple[str, str]:
        """Extract screening question answers from an application."""
        resp = self.http.get(f"{self.BASE}/applications/{application_id}")
        if resp.status_code != 200:
            return "", ""
        app = resp.json()
        answers = app.get("answers", []) or []
        texts = [a.get("answer", "") or "" for a in answers if a.get("answer")]
        return (texts[0] if len(texts) > 0 else "",
                texts[1] if len(texts) > 1 else "")

    def fetch_candidates(self) -> Iterator[Candidate]:
        """
        Pull all candidates (with applications) for a job or account-wide.
        """
        params = {}
        if self.job_id:
            params["job_id"] = self.job_id

        print(f"Fetching candidates from Greenhouse"
              + (f" (job {self.job_id})" if self.job_id else " (all jobs)"))

        for app in self._get_paginated("applications", params):
            candidate_id = app.get("candidate_id") or app.get("id")
            cand_data = app.get("candidate") or {}

            # Extract GitHub and LinkedIn from candidate web links
            links = cand_data.get("website_addresses", []) or []
            github = next((l["value"] for l in links if "github" in (l.get("value") or "").lower()), "")
            linkedin = next((l["value"] for l in links if "linkedin" in (l.get("value") or "").lower()), "")

            # Extract screening answers
            a1, a2 = self._get_application_answers(app["id"])

            # Skills from tags
            tags = cand_data.get("tags", []) or []
            skills = ", ".join(tags)

            # Applied date
            applied_at = app.get("applied_at", "")

            yield Candidate(
                candidate_id=str(candidate_id),
                name=cand_data.get("first_name", "") + " " + cand_data.get("last_name", ""),
                email=(cand_data.get("email_addresses") or [{}])[0].get("value", ""),
                github=github,
                linkedin=linkedin,
                skills=skills,
                answer_1=a1,
                answer_2=a2,
                applied_at=applied_at,
                source_platform="greenhouse",
                job_title=app.get("jobs", [{}])[0].get("name", "") if app.get("jobs") else "",
                job_id=self.job_id or "",
                stage=app.get("status", ""),
                raw_json=json.dumps(app)[:2000],
            )
            time.sleep(0.3)  # Stay under rate limit


# ─────────────────────────────────────────────
# 2. Lever API Client
# ─────────────────────────────────────────────
class LeverClient:
    """
    Uses Lever Full API (employer-side, requires API key).
    Docs: https://hire.lever.co/developer/documentation

    API key: Settings → Integrations & API → API Credentials
    Objects: opportunities (= candidate + application merged)
    Custom questions: GET /opportunities/:id/applications for form answers
    Pagination: cursor-based (next offset token in response body)

    Rate limit: ~20 req/s
    """

    BASE = "https://api.lever.co/v1"

    def __init__(self, api_key: str, posting_id: Optional[str] = None):
        self.posting_id = posting_id
        self.http = RateLimitedSession(requests_per_minute=60)
        self.http.session.headers.update({
            "Authorization": f"Bearer {api_key}",
        })
        # Lever uses Basic auth with API key as username
        import base64
        encoded = base64.b64encode(f"{api_key}:".encode()).decode()
        self.http.session.headers["Authorization"] = f"Basic {encoded}"

    def _get_paginated(self, endpoint: str, params: dict = None) -> Iterator[dict]:
        url = f"{self.BASE}/{endpoint}"
        params = params or {}
        params["limit"] = 100

        while True:
            resp = self.http.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("data", [])
            yield from items

            next_cursor = data.get("next")
            if not next_cursor or not data.get("hasNext", False):
                break
            params = {"offset": next_cursor, "limit": 100}

    def _get_form_answers(self, opportunity_id: str) -> tuple[str, str]:
        """Get custom form (screening question) answers for an opportunity."""
        resp = self.http.get(
            f"{self.BASE}/opportunities/{opportunity_id}/applications"
        )
        if resp.status_code != 200:
            return "", ""
        apps = resp.json().get("data", [])
        if not apps:
            return "", ""

        custom_qs = apps[0].get("customQuestions", []) or []
        answers = []
        for q in custom_qs:
            for field_item in q.get("fields", []):
                val = field_item.get("value")
                if val and isinstance(val, str) and len(val) > 5:
                    answers.append(val)
        return (answers[0] if len(answers) > 0 else "",
                answers[1] if len(answers) > 1 else "")

    def fetch_candidates(self) -> Iterator[Candidate]:
        """Pull all opportunities (= candidates with applications)."""
        params = {"expand": "applications,stage"}
        if self.posting_id:
            params["posting_id"] = self.posting_id

        print(f"Fetching opportunities from Lever"
              + (f" (posting {self.posting_id})" if self.posting_id else " (all postings)"))

        for opp in self._get_paginated("opportunities", params):
            opp_id = opp.get("id", "")
            name = opp.get("name", "")
            email_list = opp.get("emails", []) or []
            email = email_list[0] if email_list else ""

            links = opp.get("links", []) or []
            github = next((l for l in links if "github" in l.lower()), "")
            linkedin = next((l for l in links if "linkedin" in l.lower()), "")

            tags = opp.get("tags", []) or []
            skills = ", ".join(tags)

            a1, a2 = self._get_form_answers(opp_id)

            created_at = opp.get("createdAt", 0)
            applied_at = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc).isoformat() if created_at else ""

            stage_data = opp.get("stage") or {}
            stage = stage_data.get("text", "") if isinstance(stage_data, dict) else ""

            yield Candidate(
                candidate_id=opp_id,
                name=name,
                email=email,
                github=github,
                linkedin=linkedin,
                skills=skills,
                answer_1=a1,
                answer_2=a2,
                applied_at=applied_at,
                source_platform="lever",
                job_title=opp.get("applications", [{}])[0].get("posting", {}).get("text", "") if opp.get("applications") else "",
                stage=stage,
                raw_json=json.dumps(opp)[:2000],
            )
            time.sleep(0.25)


# ─────────────────────────────────────────────
# 3. Workable API Client
# ─────────────────────────────────────────────
class WorkableClient:
    """
    Uses Workable Full API (employer-side, requires API key + subdomain).
    Docs: https://workable.readme.io/docs

    API key: Settings → Integrations → Access Token
    Subdomain: your Workable account subdomain (e.g., "genotek")
    Pagination: cursor in response body next_page token

    Rate limit: 10 req/s
    """

    def __init__(self, api_key: str, subdomain: str, job_shortcode: Optional[str] = None):
        self.job_shortcode = job_shortcode
        self.BASE = f"https://{subdomain}.workable.com/spi/v3"
        self.http = RateLimitedSession(requests_per_minute=300)
        self.http.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _get_paginated(self, endpoint: str, params: dict = None) -> Iterator[dict]:
        url = f"{self.BASE}/{endpoint}"
        params = params or {}
        params["limit"] = 100

        while True:
            resp = self.http.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            for key in ("candidates", "applications", "jobs"):
                if key in data:
                    yield from data[key]
                    break

            next_page = data.get("paging", {}).get("next")
            if not next_page:
                break
            params = {"next_page": next_page.split("next_page=")[-1]}

    def _get_candidate_answers(self, candidate_id: str, job_shortcode: str) -> tuple[str, str]:
        resp = self.http.get(
            f"{self.BASE}/jobs/{job_shortcode}/candidates/{candidate_id}"
        )
        if resp.status_code != 200:
            return "", ""
        cand = resp.json().get("candidate") or {}
        answers = cand.get("answers", []) or []
        texts = [a.get("body", "") for a in answers if a.get("body")]
        return (texts[0] if len(texts) > 0 else "",
                texts[1] if len(texts) > 1 else "")

    def fetch_candidates(self) -> Iterator[Candidate]:
        if self.job_shortcode:
            endpoints = [f"jobs/{self.job_shortcode}/candidates"]
        else:
            # Get all jobs first, then all candidates
            jobs = list(self._get_paginated("jobs", {"state": "published"}))
            endpoints = [f"jobs/{j['shortcode']}/candidates" for j in jobs]

        print(f"Fetching candidates from Workable ({len(endpoints)} job(s))...")

        for endpoint in endpoints:
            job_shortcode = endpoint.split("/")[1]
            for cand in self._get_paginated(endpoint, {"stage": "applied"}):
                cand_id = cand.get("id", "")
                a1, a2 = self._get_candidate_answers(cand_id, job_shortcode)

                links = cand.get("social_profiles", []) or []
                github = next((l.get("url", "") for l in links if l.get("type") == "github"), "")
                linkedin = next((l.get("url", "") for l in links if l.get("type") == "linkedin"), "")

                applied_at = cand.get("created_at", "")

                yield Candidate(
                    candidate_id=cand_id,
                    name=cand.get("name", ""),
                    email=cand.get("email", ""),
                    github=github,
                    linkedin=linkedin,
                    skills=", ".join(cand.get("tags", []) or []),
                    answer_1=a1,
                    answer_2=a2,
                    applied_at=applied_at,
                    source_platform="workable",
                    stage=cand.get("stage", {}).get("kind", ""),
                    raw_json=json.dumps(cand)[:2000],
                )
                time.sleep(0.12)


# ─────────────────────────────────────────────
# 4. CSV Fallback (Internshala export / any CSV)
# ─────────────────────────────────────────────
class CSVClient:
    """
    Load candidates from any CSV export (Internshala, Naukri, manual, etc.).
    Flexible column mapping — works with whatever headers the export gives you.
    """

    # Column aliases — maps possible CSV header names → canonical field
    COLUMN_MAP = {
        "name":      ["name", "full name", "candidate name", "applicant name"],
        "email":     ["email", "email address", "e-mail", "mail"],
        "github":    ["github", "github url", "github profile", "github link"],
        "linkedin":  ["linkedin", "linkedin url", "linkedin profile"],
        "skills":    ["skills", "skill set", "tech skills", "technologies", "expertise"],
        "answer_1":  ["answer_1", "answer 1", "q1", "question 1", "response 1", "screening answer 1"],
        "answer_2":  ["answer_2", "answer 2", "q2", "question 2", "response 2", "screening answer 2"],
        "applied_at":["applied at", "applied date", "date applied", "application date", "created at"],
        "job_title": ["job title", "position", "role", "job", "posting"],
    }

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    def _resolve_columns(self, headers: list[str]) -> dict:
        """Map CSV headers to canonical field names (case-insensitive)."""
        headers_lower = {h.lower().strip(): h for h in headers}
        resolved = {}
        for canon_field, aliases in self.COLUMN_MAP.items():
            for alias in aliases:
                if alias in headers_lower:
                    resolved[canon_field] = headers_lower[alias]
                    break
        return resolved

    def fetch_candidates(self) -> Iterator[Candidate]:
        print(f"Loading candidates from CSV: {self.csv_path}")
        with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            col_map = self._resolve_columns(reader.fieldnames or [])
            print(f"  Detected column mapping: {col_map}")

            for i, row in enumerate(reader):
                def get(field: str) -> str:
                    col = col_map.get(field)
                    return row.get(col, "").strip() if col else ""

                name = get("name")
                email = get("email")
                if not name and not email:
                    continue  # skip completely empty rows

                yield Candidate(
                    candidate_id=f"csv_{i}_{email or name}",
                    name=name,
                    email=email,
                    github=get("github"),
                    linkedin=get("linkedin"),
                    skills=get("skills"),
                    answer_1=get("answer_1"),
                    answer_2=get("answer_2"),
                    applied_at=get("applied_at"),
                    source_platform="csv",
                    job_title=get("job_title"),
                    raw_json=json.dumps(dict(row))[:2000],
                )


# ─────────────────────────────────────────────
# 5. Mock ATS Client (for dev / demo)
# ─────────────────────────────────────────────
class MockATSClient:
    """
    Generates realistic synthetic candidates for testing without any API key.
    Simulates 1,140 candidates across 57 pages — exactly the Internshala scenario.
    Includes realistic noise: missing fields, Hindi-English mixed, empty GitHub profiles.
    """

    NAMES = [
        "Aarav Sharma", "Priya Patel", "Rahul Verma", "Sneha Gupta", "Arjun Singh",
        "Divya Mehta", "Karan Joshi", "Pooja Agarwal", "Vikram Rao", "Ananya Kumar",
        "Rohan Das", "Ishaan Nair", "Meera Pillai", "Siddharth Shah", "Kavya Reddy",
        "Aditya Bose", "Tanvi Mishra", "Nikhil Khanna", "Shreya Tiwari", "Dev Malhotra",
        "Alice Chen", "Bob Martinez", "Charlie Smith", "Diana Johnson", "Eve Wilson",
    ]

    GITHUB_PATTERNS = [
        "https://github.com/{}",
        "",  # No GitHub — 30% of candidates
        "https://github.com/{}123",
        "https://github.com/user_{}",
    ]

    SKILLS_POOL = [
        "Python, FastAPI, Docker, Redis, PostgreSQL",
        "Python, Selenium, Flask, MySQL",
        "JavaScript, React, Node.js, MongoDB",
        "Python, Machine Learning, TensorFlow, pandas",
        "Java, Spring Boot, AWS, Kubernetes",
        "Python, Django, REST API, Git",
        "React, TypeScript, GraphQL, AWS",
        "Python, LangChain, OpenAI, FastAPI, Redis",
        "Rust, Go, distributed systems, Kafka",
        "Python, Scrapy, BeautifulSoup, SQL",
        "Data Analysis, Excel, Power BI",  # Non-technical — should score low
        "I know programming",              # Vague — penalized
    ]

    ANSWERS_Q1_GOOD = [
        "I inspected the Internshala login flow using Chrome DevTools Network tab. "
        "The reCAPTCHA v3 invisible variant sends a POST to https://www.google.com/recaptcha/api2/userverify "
        "with a token scoring the session 0.0-1.0 — headless browsers score 0.1 or lower. "
        "Attempt 1: Playwright stealth mode (playwright-extra + puppeteer-extra-plugin-stealth) — "
        "blocked at form submit, HTTP 403. Attempt 2: extracted cookies from Firefox cookies.sqlite "
        "after manual login — worked locally but failed on server due to IP binding (HTTP 400). "
        "Solution: residential proxy (Bright Data) with sticky sessions — 1 IP per session. "
        "Wrote a Python 3.11 script using httpx + asyncio: 57 pages x 20 candidates = 1140 total. "
        "Rate limit: 2 req/s to avoid detection. Full code at https://github.com/myuser/internshala-scraper",

        "reCAPTCHA v3 scores the browser session using 18+ behavioral signals including mouse "
        "movement, scroll patterns, and keypress timing — not just the form submit. "
        "I used curl-impersonate (Chrome 120 profile) to replicate the TLS ClientHello fingerprint. "
        "Then wrote a mouse-movement simulator in Python: 50-100ms intervals, bezier-curve paths. "
        "Result: reCAPTCHA score of 0.7 (threshold is 0.5). "
        "Session management: FastAPI backend, Redis for session storage, 1 Bright Data residential "
        "IP per session. Handles 57 pages in 4 minutes. Pagination via XHR offset parameter. "
        "Code: ",
    ]

    ANSWERS_Q1_MEDIOCRE = [
        "I would use Selenium with headless Chrome to automate the login. Then use BeautifulSoup "
        "to parse the HTML and extract applicant data. For pagination, I'd loop through all 57 pages.",

        "Use Playwright to automate the browser and get the data. Can handle CAPTCHA with "
        "third-party solving services like 2Captcha.",
    ]

    ANSWERS_Q1_AI = [
        "Certainly! I'd be happy to help with this challenge. Here's a comprehensive overview "
        "of my approach. In today's rapidly evolving landscape of web automation and data extraction, "
        "it's worth noting that leveraging cutting-edge technologies can provide a holistic solution. "
        "I would utilize Selenium WebDriver with headless Chrome to seamlessly automate the login "
        "process. This paradigm shift in approach would foster innovation by enabling us to scale. "
        "To summarize: I would use machine learning to detect and solve the CAPTCHA.",

        "As an AI language model, I can outline a comprehensive approach. Great question! "
        "In today's rapidly evolving technological landscape, I'd be happy to help design "
        "this system. A holistic, cutting-edge solution would leverage multiple approaches.",
    ]

    ANSWERS_Q1_BLANK = ["", "NA", "Will discuss in interview", "no idea"]

    def __init__(self, total_candidates: int = 1140, seed: int = 42):
        self.total_candidates = total_candidates
        random.seed(seed)

    def fetch_candidates(self) -> Iterator[Candidate]:
        print(f"Generating {self.total_candidates} synthetic candidates (mock ATS)...")

        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                   "iitd.ac.in", "bits-pilani.ac.in", "example.com"]

        for i in range(self.total_candidates):
            name = random.choice(self.NAMES) + f"_{i}"
            email_name = name.lower().replace(" ", ".").replace("_", "")
            email = f"{email_name}@{random.choice(domains)}"

            # GitHub: 60% have a link, 40% empty
            if random.random() < 0.6:
                gh_user = name.lower().replace(" ", "").replace("_", "")[:12]
                github = f"https://github.com/{gh_user}{random.randint(1, 999)}"
            else:
                github = ""

            skills = random.choice(self.SKILLS_POOL)

            # Answer distribution mimics real applicants:
            # 10% great, 25% mediocre, 30% AI-generated, 15% blank, 20% short
            r = random.random()
            if r < 0.10:
                a1 = random.choice(self.ANSWERS_Q1_GOOD)
                a2 = "For scoring, I'd embed answers using sentence-transformers and compute cosine similarity."
            elif r < 0.35:
                a1 = random.choice(self.ANSWERS_Q1_MEDIOCRE)
                a2 = "I would create a scoring algorithm based on keywords and answer length."
            elif r < 0.65:
                a1 = random.choice(self.ANSWERS_Q1_AI)
                a2 = random.choice(self.ANSWERS_Q1_AI)
            elif r < 0.80:
                a1 = random.choice(self.ANSWERS_Q1_BLANK)
                a2 = ""
            else:
                # Hindi-English mixed (realistic for Indian job market)
                a1 = f"Main Python use karta hoon, aur {random.choice(['machine learning', 'web dev', 'data science'])} mein experience hai."
                a2 = "Mujhe yeh role bahut pasand aayi kyunki main AI mein interested hoon."

            applied_days_ago = random.randint(1, 30)
            applied_at = (datetime.now(timezone.utc) - timedelta(days=applied_days_ago)).isoformat()

            yield Candidate(
                candidate_id=f"mock_{i}",
                name=name.rsplit("_", 1)[0],
                email=email,
                github=github,
                linkedin=f"https://linkedin.com/in/{email_name}" if random.random() > 0.4 else "",
                skills=skills,
                answer_1=a1,
                answer_2=a2,
                applied_at=applied_at,
                source_platform="mock",
                job_title="AI Agent Developer",
                stage="applied",
                raw_json=json.dumps({"mock": True, "index": i}),
            )

            if i % 100 == 0:
                print(f"  Generated {i}/{self.total_candidates} candidates...")


# ─────────────────────────────────────────────
# 6. ATS Router — auto-detect and normalize
# ─────────────────────────────────────────────
class ATSRouter:
    """
    Entry point. Auto-selects the right client based on environment variables
    or explicit config. Normalizes all candidates to a standard CSV for scorer.py.

    Environment variables:
      ATS_SOURCE       = "greenhouse" | "lever" | "workable" | "csv" | "mock"
      GREENHOUSE_KEY   = your Harvest API key
      GREENHOUSE_JOB   = job ID (optional, pulls all if omitted)
      LEVER_KEY        = your Lever API key
      LEVER_POSTING    = posting ID (optional)
      WORKABLE_KEY     = your Workable access token
      WORKABLE_SUB     = your Workable subdomain
      WORKABLE_JOB     = job shortcode (optional)
      CSV_PATH         = path to CSV file
    """

    def __init__(
        self,
        source: Optional[str] = None,
        greenhouse_key: Optional[str] = None,
        greenhouse_job: Optional[str] = None,
        lever_key: Optional[str] = None,
        lever_posting: Optional[str] = None,
        workable_key: Optional[str] = None,
        workable_subdomain: Optional[str] = None,
        workable_job: Optional[str] = None,
        csv_path: Optional[str] = None,
        mock_count: int = 1140,
    ):
        self.source = (source or os.getenv("ATS_SOURCE", "mock")).lower()

        # Resolve from kwargs or env
        self.greenhouse_key = greenhouse_key or os.getenv("GREENHOUSE_KEY")
        self.greenhouse_job = greenhouse_job or os.getenv("GREENHOUSE_JOB")
        self.lever_key = lever_key or os.getenv("LEVER_KEY")
        self.lever_posting = lever_posting or os.getenv("LEVER_POSTING")
        self.workable_key = workable_key or os.getenv("WORKABLE_KEY")
        self.workable_sub = workable_subdomain or os.getenv("WORKABLE_SUB")
        self.workable_job = workable_job or os.getenv("WORKABLE_JOB")
        self.csv_path = csv_path or os.getenv("CSV_PATH")
        self.mock_count = mock_count

    def _build_client(self):
        if self.source == "greenhouse":
            if not self.greenhouse_key:
                raise ValueError("Set GREENHOUSE_KEY env var (Settings → Dev Center → API Credential Management)")
            return GreenhouseClient(self.greenhouse_key, self.greenhouse_job)

        elif self.source == "lever":
            if not self.lever_key:
                raise ValueError("Set LEVER_KEY env var (Settings → Integrations & API → API Credentials)")
            return LeverClient(self.lever_key, self.lever_posting)

        elif self.source == "workable":
            if not self.workable_key or not self.workable_sub:
                raise ValueError("Set WORKABLE_KEY and WORKABLE_SUB env vars")
            return WorkableClient(self.workable_key, self.workable_sub, self.workable_job)

        elif self.source == "csv":
            if not self.csv_path:
                raise ValueError("Set CSV_PATH env var or pass csv_path")
            return CSVClient(self.csv_path)

        else:  # mock
            return MockATSClient(total_candidates=self.mock_count)

    def fetch_and_save(
        self,
        output_csv: str = "candidates_raw.csv",
        db_path: str = "hiring_agent.db",
        limit: Optional[int] = None,
    ) -> str:
        """
        Fetch all candidates, save to CSV + SQLite, return CSV path.
        The CSV is the handoff point to scorer.py.

        For mock source: dedup is skipped (mock IDs are index-based and
        change each run — we always want fresh data in dev mode).
        """
        client = self._build_client()
        print(f"\nSource: {self.source.upper()}")

        use_dedup = (self.source != "mock")  # No dedup for mock/dev runs

        # SQLite: log raw candidates for audit + dedup
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_candidates (
                candidate_id TEXT PRIMARY KEY,
                name TEXT, email TEXT, github TEXT,
                skills TEXT, answer_1 TEXT, answer_2 TEXT,
                applied_at TEXT, source_platform TEXT,
                job_title TEXT, stage TEXT, raw_json TEXT,
                fetched_at TEXT
            )
        """)

        rows_written = 0
        skipped_dups = 0
        fieldnames = [f.name for f in Candidate.__dataclass_fields__.values()]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for candidate in client.fetch_candidates():
                if limit and rows_written >= limit:
                    break

                # Dedup by email (only for real ATS sources)
                if use_dedup:
                    existing = conn.execute(
                        "SELECT 1 FROM raw_candidates WHERE candidate_id = ? OR email = ?",
                        (candidate.candidate_id, candidate.email)
                    ).fetchone()
                    if existing:
                        skipped_dups += 1
                        continue


                row = asdict(candidate)
                writer.writerow(row)

                conn.execute("""
                    INSERT OR IGNORE INTO raw_candidates VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    candidate.candidate_id, candidate.name, candidate.email,
                    candidate.github, candidate.skills, candidate.answer_1,
                    candidate.answer_2, candidate.applied_at, candidate.source_platform,
                    candidate.job_title, candidate.stage, candidate.raw_json,
                    datetime.now(timezone.utc).isoformat(),
                ))
                conn.commit()
                rows_written += 1

        conn.close()
        print(f"\nFetched {rows_written} candidates → {output_csv}")
        if skipped_dups:
            print(f"Skipped {skipped_dups} duplicates (already in DB)")
        return output_csv


# ─────────────────────────────────────────────
# CLI Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Component 1: ATS Data Access Layer")
    parser.add_argument("--source", default="mock",
                        choices=["mock", "csv", "greenhouse", "lever", "workable"],
                        help="Data source (default: mock)")
    parser.add_argument("--output", default="candidates_raw.csv",
                        help="Output CSV path (feeds into scorer.py)")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of mock candidates (default: 50 for demo)")
    parser.add_argument("--csv-path", help="Path to CSV file (for --source csv)")
    parser.add_argument("--limit", type=int, help="Max candidates to fetch")
    args = parser.parse_args()

    router = ATSRouter(
        source=args.source,
        csv_path=args.csv_path,
        mock_count=args.count,
    )

    output_path = router.fetch_and_save(
        output_csv=args.output,
        limit=args.limit,
    )

    print(f"\nNext step: python scorer.py → score {output_path}")
    print("Or:        python orchestrator.py --score", output_path)

    # Show a preview
    print(f"\nFirst 3 candidates:")
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 3:
                break
            print(f"  [{i+1}] {row['name']} | {row['email']} | github: {row['github'][:40] or 'none'}")
            print(f"       skills: {row['skills'][:60]}")
            print(f"       answer_1: {row['answer_1'][:80]}...")
            print()
