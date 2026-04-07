"""
Component 2: INTELLIGENCE — Scoring & Ranking Applicants
Reads a CSV of applicants and outputs a scored, ranked spreadsheet.
"""

import pandas as pd
import numpy as np
import requests
import re
import json
from dataclasses import dataclass, field
from typing import Optional
import time

# --- AI Fingerprint Phrases ---
AI_PHRASES = [
    "i'd be happy to help", "here's a comprehensive overview",
    "in today's rapidly evolving landscape", "it's worth noting",
    "certainly!", "absolutely!", "great question",
    "as an ai language model", "i hope this helps",
    "let me break this down", "in conclusion",
    "to summarize", "this is a multifaceted",
    "dive deep into", "leverage", "utilize",
    "at its core", "in the realm of",
    "foster innovation", "seamlessly integrate",
    "a testament to", "paradigm shift",
    "holistic approach", "cutting-edge",
]

# --- Technical Skill Keywords ---
TECH_SKILLS = {
    "python": 3, "javascript": 3, "typescript": 3, "rust": 4, "go": 3,
    "react": 2, "fastapi": 3, "django": 2, "flask": 2, "node": 2,
    "docker": 3, "kubernetes": 4, "aws": 3, "gcp": 3, "azure": 3,
    "postgresql": 2, "mongodb": 2, "redis": 3, "kafka": 4,
    "langchain": 3, "openai": 2, "anthropic": 3, "llm": 2,
    "selenium": 1, "playwright": 2, "scrapy": 2, "beautiful soup": 1,
    "machine learning": 2, "deep learning": 2, "nlp": 2,
    "git": 1, "ci/cd": 3, "linux": 2, "bash": 2,
    "api": 1, "rest": 1, "graphql": 3, "websocket": 3,
    "agent": 2, "autonomous": 2, "pipeline": 2,
}


@dataclass
class CandidateScore:
    name: str
    email: str
    raw_score: float = 0.0
    skill_score: float = 0.0
    answer_quality_score: float = 0.0
    github_score: float = 0.0
    ai_penalty: float = 0.0
    completeness_score: float = 0.0
    breakdown: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    tier: str = "Reject"


def detect_ai_phrases(text: str) -> tuple[int, list]:
    """Returns count of AI fingerprint phrases found and which ones."""
    if not text:
        return 0, []
    text_lower = text.lower()
    found = [p for p in AI_PHRASES if p in text_lower]
    return len(found), found


def score_answer_quality(text: str) -> float:
    """
    Score answer quality 0–10 based on:
    - Length (not too short, not padding-long)
    - Specificity (numbers, names, code snippets)
    - Structure (not a wall of text, not single line)
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    score = 0.0
    text = text.strip()
    word_count = len(text.split())

    # Length scoring (sweet spot: 80–400 words)
    if word_count < 10:
        score += 0.5
    elif word_count < 50:
        score += 2.0
    elif word_count <= 400:
        score += 4.0
    else:
        score += 3.0  # Penalize padding

    # Specificity signals
    has_numbers = bool(re.search(r'\b\d+\b', text))
    has_code = bool(re.search(r'```|def |import |class |<code>|\(\)', text))
    has_urls = bool(re.search(r'https?://', text))
    has_named_tools = sum(1 for skill in TECH_SKILLS if skill in text.lower())

    if has_numbers:
        score += 1.0
    if has_code:
        score += 2.0
    if has_urls:
        score += 0.5
    score += min(has_named_tools * 0.3, 2.0)

    # Penalize one-liners disguised as paragraphs
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) >= 3:
        score += 0.5

    return min(score, 10.0)


def score_skills(text: str) -> float:
    """Score based on technical skills mentioned in answers/resume."""
    if not text:
        return 0.0
    text_lower = text.lower()
    total = sum(weight for skill, weight in TECH_SKILLS.items() if skill in text_lower)
    return min(total, 20.0)  # cap at 20


def check_github(github_url: str) -> tuple[float, list]:
    """
    Check GitHub profile quality via GitHub API.
    Returns (score 0–10, list of flags).
    No auth needed for public data (60 req/hour unauthenticated).
    """
    if not github_url or not isinstance(github_url, str):
        return 0.0, ["no_github"]

    # Extract username
    match = re.search(r'github\.com/([a-zA-Z0-9\-]+)', github_url)
    if not match:
        return 0.0, ["invalid_github_url"]

    username = match.group(1)
    flags = []
    score = 2.0  # base for having a valid URL

    try:
        # User profile
        user_resp = requests.get(
            f"https://api.github.com/users/{username}",
            timeout=5,
            headers={"Accept": "application/vnd.github.v3+json"}
        )
        if user_resp.status_code == 404:
            return 0.0, ["github_404"]
        if user_resp.status_code != 200:
            return 1.0, ["github_api_error"]

        user_data = user_resp.json()

        public_repos = user_data.get("public_repos", 0)
        followers = user_data.get("followers", 0)
        following = user_data.get("following", 0)
        created_at = user_data.get("created_at", "")
        bio = user_data.get("bio") or ""

        # Account age (older = more legit)
        if created_at:
            year = int(created_at[:4])
            account_age_years = max(0, 2025 - year)
            score += min(account_age_years * 0.3, 1.5)

        # Repos
        if public_repos == 0:
            flags.append("empty_github_profile")
            return 0.5, flags
        elif public_repos < 3:
            score += 0.5
            flags.append("few_repos")
        elif public_repos < 10:
            score += 1.5
        else:
            score += 3.0

        # Followers signal real activity
        if followers > 10:
            score += 1.0
        elif followers > 50:
            score += 2.0

        # Bio shows care
        if len(bio) > 10:
            score += 0.5

        # Check repos for quality signals
        repos_resp = requests.get(
            f"https://api.github.com/users/{username}/repos?per_page=10&sort=updated",
            timeout=5,
            headers={"Accept": "application/vnd.github.v3+json"}
        )
        if repos_resp.status_code == 200:
            repos = repos_resp.json()
            stars = sum(r.get("stargazers_count", 0) for r in repos)
            has_readme = any(r.get("description") for r in repos)
            non_forked = [r for r in repos if not r.get("fork")]

            if stars > 5:
                score += 1.0
            if has_readme:
                score += 0.5
            if len(non_forked) > 3:
                score += 1.0

            # Flag if all repos are forks (tutorial-follower pattern)
            if repos and len(non_forked) == 0:
                flags.append("all_forked_repos")
                score -= 1.0

    except requests.exceptions.RequestException:
        flags.append("github_request_failed")
        return 1.0, flags

    return min(score, 10.0), flags


def compute_completeness(row: pd.Series, answer_columns: list) -> float:
    """Score how completely the candidate filled out the application."""
    filled = sum(1 for col in answer_columns if pd.notna(row.get(col)) and str(row.get(col, "")).strip())
    if len(answer_columns) == 0:
        return 5.0
    return (filled / len(answer_columns)) * 10.0


def assign_tier(score: float, check_github: bool = False) -> str:
    """
    Convert numeric score to tier label.
    Thresholds are calibrated for real ATS data with GitHub API enabled.
    Without GitHub API (check_github=False), effective max is ~66, so we
    scale down proportionally: Fast-Track>=60, Standard>=44, Review>=28.
    """
    if check_github:
        # Full scoring with live GitHub data — original thresholds
        if score >= 75:
            return "Fast-Track"
        elif score >= 55:
            return "Standard"
        elif score >= 35:
            return "Review"
        else:
            return "Reject"
    else:
        # No GitHub API — lower ceiling, adjusted thresholds
        if score >= 60:
            return "Fast-Track"
        elif score >= 44:
            return "Standard"
        elif score >= 25:
            return "Review"
        else:
            return "Reject"


def score_candidates(
    input_csv: str,
    output_csv: str = "scored_candidates.csv",
    name_col: str = "name",
    email_col: str = "email",
    github_col: str = "github",
    skills_col: str = "skills",
    answer_cols: Optional[list] = None,
    check_github_api: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Main scoring function. Reads a CSV, scores each candidate, writes output.

    Expected CSV columns (flexible — pass column names as args):
      name, email, github, skills, answer_1, answer_2, ...
    """
    df = pd.read_csv(input_csv)

    if verbose:
        print(f"Loaded {len(df)} candidates from {input_csv}")

    # Auto-detect answer columns if not specified
    if answer_cols is None:
        answer_cols = [c for c in df.columns if "answer" in c.lower() or "response" in c.lower() or "question" in c.lower()]
    if verbose:
        print(f"Detected answer columns: {answer_cols}")

    results = []

    for idx, row in df.iterrows():
        name = str(row.get(name_col, f"Candidate_{idx}"))
        email = str(row.get(email_col, ""))

        cs = CandidateScore(name=name, email=email)

        # --- Skill Score ---
        skill_text = " ".join([
            str(row.get(skills_col, "")),
            *[str(row.get(c, "")) for c in answer_cols]
        ])
        cs.skill_score = score_skills(skill_text)

        # --- Answer Quality ---
        answer_scores = []
        for col in answer_cols:
            ans = str(row.get(col, ""))
            q_score = score_answer_quality(ans)

            ai_count, ai_found = detect_ai_phrases(ans)
            if ai_count > 0:
                penalty = min(ai_count * 5, 20)
                cs.ai_penalty += penalty
                cs.flags.append(f"ai_phrases_in_{col}: {ai_found[:2]}")

            answer_scores.append(q_score)

        cs.answer_quality_score = np.mean(answer_scores) if answer_scores else 0.0

        # --- GitHub ---
        github_url = str(row.get(github_col, ""))
        if check_github_api and github_url:
            gh_score, gh_flags = check_github(github_url)
            cs.github_score = gh_score
            cs.flags.extend(gh_flags)
            time.sleep(0.5)  # Respect GitHub rate limit
        else:
            cs.github_score = 2.0 if github_url else 0.0

        # --- Completeness ---
        cs.completeness_score = compute_completeness(row, answer_cols)

        # --- Final Score (weighted) ---
        cs.raw_score = (
            cs.skill_score * 1.5        # max 30
            + cs.answer_quality_score * 2.0  # max 20
            + cs.github_score * 2.0     # max 20
            + cs.completeness_score * 1.0   # max 10
            - cs.ai_penalty             # subtract penalties
        )
        # Normalize to 0–100
        cs.raw_score = max(0.0, min(100.0, cs.raw_score * (100 / 80)))

        cs.tier = assign_tier(cs.raw_score, check_github_api)
        cs.breakdown = {
            "skill_score": round(cs.skill_score, 2),
            "answer_quality": round(cs.answer_quality_score, 2),
            "github_score": round(cs.github_score, 2),
            "completeness": round(cs.completeness_score, 2),
            "ai_penalty": round(cs.ai_penalty, 2),
        }

        results.append(cs)

        if verbose and idx % 50 == 0:
            print(f"  Scored {idx+1}/{len(df)}: {name} → {cs.raw_score:.1f} ({cs.tier})")

    # Sort by score descending
    results.sort(key=lambda x: x.raw_score, reverse=True)

    # Build output DataFrame
    out_rows = []
    for rank, cs in enumerate(results, 1):
        out_rows.append({
            "rank": rank,
            "name": cs.name,
            "email": cs.email,
            "score": round(cs.raw_score, 1),
            "tier": cs.tier,
            "skill_score": cs.breakdown["skill_score"],
            "answer_quality": cs.breakdown["answer_quality"],
            "github_score": cs.breakdown["github_score"],
            "completeness": cs.breakdown["completeness"],
            "ai_penalty": cs.breakdown["ai_penalty"],
            "flags": "; ".join(cs.flags) if cs.flags else "",
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_csv, index=False)

    if out_df.empty:
        if verbose:
            print(f"\nNo candidates to score in {input_csv}.")
            print("If using --full-run, delete hiring_agent.db to clear duplicate cache and retry.")
        return out_df

    if verbose:
        tier_counts = out_df["tier"].value_counts().to_dict()
        print(f"\nScoring complete. Output: {output_csv}")
        print(f"Tier breakdown: {tier_counts}")
        print(f"\nTop 5 candidates:")
        print(out_df[["rank", "name", "score", "tier", "flags"]].head().to_string())

    return out_df


if __name__ == "__main__":
    # Demo: create a small sample CSV and score it
    sample_data = {
        "name": ["Alice Sharma", "Bob Kumar", "Charlie Das", "Diana Mehta", "Eve Patel"],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "diana@example.com", "eve@example.com"],
        "github": [
            "https://github.com/torvalds",   # Well-known profile for demo
            "https://github.com/nonexistent_user_xyz123",
            "",
            "https://github.com/torvalds",
            "https://github.com/gvanrossum",
        ],
        "skills": [
            "Python, FastAPI, Docker, Kubernetes, LangChain, Redis",
            "Python, Selenium, Beautiful Soup",
            "I know Python and ML",
            "Python, React, TypeScript, PostgreSQL, AWS, CI/CD pipelines",
            "Rust, Go, distributed systems, Kafka, WebSocket",
        ],
        "answer_1": [
            "I'd approach this by first setting up a Playwright scraper with stealth mode to handle the reCAPTCHA. I tested this on a local Internshala account and found the cookie binding issue — the session token is tied to IP so rotating proxies alone won't work. You need to maintain the same egress IP per session, which I handled with a dedicated proxy per worker.",
            "I would use Selenium with headless Chrome and BeautifulSoup to parse the HTML and extract all applicant data automatically.",
            "I would use machine learning to analyze the data.",
            "Here's a comprehensive overview of my approach: I'd be happy to help design this system. In today's rapidly evolving landscape of AI recruitment, it's worth noting that leveraging cutting-edge paradigm shifts can foster innovation.",
            "Built a scraper last year for a freelance client — ran into Cloudflare's TLS fingerprinting, ended up using curl-impersonate to mimic a real browser TLS handshake. For reCAPTCHA specifically, the invisible v3 variant scores based on behavioral signals so you need a mouse-movement simulator, not just form submission.",
        ],
        "answer_2": [
            "For scoring, I'd embed answers using sentence-transformers all-MiniLM-L6-v2, compute cosine similarity against ideal answers, then combine with a rules engine for GitHub quality, AI phrase detection, and completeness. The weights I'd use: answers 40%, skills 30%, GitHub 20%, completeness 10%.",
            "I would use NLP and machine learning to analyze candidate profiles based on relevance.",
            "Use GPT to score the answers.",
            "To summarize my comprehensive approach to this multifaceted challenge, I would utilize a holistic strategy that seamlessly integrates multiple cutting-edge technologies.",
            "Scoring is only as good as your features. I'd weight specificity of technical answers (does the answer name actual tools and failure modes?) more than length. GitHub is easy to fake — I'd check commit frequency, not just repo count. A student with 2 repos but 50 commits in the last 3 months beats someone with 20 empty repos.",
        ],
    }

    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv("sample_applicants.csv", index=False)
    print("Created sample_applicants.csv\n")

    result = score_candidates(
        "sample_applicants.csv",
        "scored_output.csv",
        check_github_api=False,  # Set True to enable GitHub API calls
    )
