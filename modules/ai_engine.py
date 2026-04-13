# import json
# import re
# import urllib.request
# import urllib.error

# OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# HARDCODED_API_KEY = "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"

# SYSTEM_PROMPT = (
#     "You are a business analyst and startup consultant for the Indian market. "
#     "Evaluate startup ideas and return ONLY a valid JSON object. "
#     "No markdown, no explanation outside the JSON."
# )

# ANALYSIS_TEMPLATE = """\
# Evaluate this startup idea. Return ONLY valid JSON, no extra text:

# - Market: {locality}
# - Budget: {budget}
# - Team Size: {team_size} people
# - Domain: {domain}
# - Idea: {idea}

# JSON structure to return:
# {{
#   "success_probability": <integer 0-100>,
#   "sustainability_score": <integer 0-100>,
#   "analysis": {{
#     "market_demand": "<2 sentence market demand assessment>",
#     "competition": "<2 sentence competitive landscape>",
#     "feasibility": "<2 sentence feasibility for Indian market>",
#     "scalability": "<2 sentence scalability outlook>"
#   }},
#   "improvements": ["<improvement 1>","<improvement 2>","<improvement 3>"],
#   "new_ideas": [
#     {{"title":"<title>","description":"<1 sentence>","market_potential":"High"}},
#     {{"title":"<title>","description":"<1 sentence>","market_potential":"High"}},
#     {{"title":"<title>","description":"<1 sentence>","market_potential":"Medium"}},
#     {{"title":"<title>","description":"<1 sentence>","market_potential":"Medium"}},
#     {{"title":"<title>","description":"<1 sentence>","market_potential":"Medium"}},
#     {{"title":"<title>","description":"<1 sentence>","market_potential":"Low"}},
#     {{"title":"<title>","description":"<1 sentence>","market_potential":"Low"}}
#   ],
#   "summary": "<2 sentence overall assessment>"
# }}

# Use INR for money. Be concise and specific.
# """


# def _try_parse(raw: str) -> dict:
#     raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
#     raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
#     raw = raw.strip()
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         pass
#     match = re.search(r"\{.*\}", raw, re.DOTALL)
#     if match:
#         try:
#             return json.loads(match.group())
#         except json.JSONDecodeError:
#             pass
#     raise ValueError(f"Could not parse response as JSON: {raw[:200]}")


# def analyze_idea(locality: str, budget: str, team_size: int,
#                  domain: str, idea: str, api_key: str = "") -> dict:
#     key = api_key.strip() or HARDCODED_API_KEY.strip()

#     prompt = ANALYSIS_TEMPLATE.format(
#         locality=locality, budget=budget,
#         team_size=team_size, domain=domain, idea=idea,
#     )

#     models_to_try = [
#         "openrouter/auto",
#         "mistralai/mistral-7b-instruct",
#         "meta-llama/llama-3.1-8b-instruct:free",
#         "google/gemma-2-9b-it:free",
#     ]

#     last_error = None
#     for model in models_to_try:
#         payload = json.dumps({
#             "model": model,
#             "messages": [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": prompt},
#             ],
#             "temperature": 0.3,
#             "max_tokens": 1200,
#         }).encode("utf-8")

#         req = urllib.request.Request(
#             OPENROUTER_URL,
#             data=payload,
#             headers={
#                 "Content-Type": "application/json",
#                 "Authorization": f"Bearer {key}",
#                 "HTTP-Referer": "http://localhost:8501",
#                 "X-Title": "Startup Analyzer",
#             },
#             method="POST",
#         )
#         try:
#             with urllib.request.urlopen(req, timeout=45) as resp:
#                 body = json.loads(resp.read().decode("utf-8"))

#             if "error" in body:
#                 last_error = body["error"]
#                 continue

#             raw = body["choices"][0]["message"]["content"].strip()
#             return _try_parse(raw)

#         except urllib.error.HTTPError as exc:
#             err_body = exc.read().decode("utf-8", errors="replace")
#             last_error = f"HTTP {exc.code}: {err_body[:200]}"
#             continue
#         except Exception as exc:
#             last_error = str(exc)
#             continue

#     raise ValueError(f"All models failed. Last error: {last_error}")

"""
ai_engine.py — AI analysis engine via OpenRouter (FREE)
SUCCESS & SUSTAINABILITY SCORES are now calculated from the real dataset.
The AI receives the dataset scores and provides qualitative analysis only.
"""

import json
import os
import re
import urllib.request
import urllib.error

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HARDCODED_API_KEY = "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"  # ← paste your key once here

SYSTEM_PROMPT = (
    "You are a venture capitalist, market analyst, and sustainability consultant "
    "specialised in the Indian startup ecosystem. "
    "You are given dataset-calculated scores for a startup idea. "
    "Your job is to provide qualitative analysis, explanations, improvements, "
    "and alternative ideas — NOT to assign new scores. "
    "Return ONLY a valid JSON object. No markdown, no text outside the JSON."
)

ANALYSIS_TEMPLATE = """\
A startup idea has been evaluated using real Indian startup dataset (500 records, 2024-2026).
The following scores were CALCULATED FROM THE DATASET — do NOT change them:

Dataset-Calculated Scores:
- Success Probability  : {success_probability}% (based on avg Success_Rate of {funded_startups} real {domain} startups)
- Sustainability Score : {sustainability_score}/100 (based on avg Sustainability_Score of {funded_startups} real {domain} startups)
- Market Demand Score  : {market_demand_score}/100
- Feasibility Score    : {feasibility_score}/100
- Scalability Score    : {scalability_score}/100
- Competition Level    : {competition_level}
- Growth Rate          : {growth_rate}% YoY
- Risk Level           : {risk_level}
- Social Impact Score  : {social_impact_score}/100

User's Startup Idea:
- Market / Locality : {locality}
- Budget            : {budget}
- Team Size         : {team_size} people
- Domain            : {domain}
- Idea              : {idea}

Using the dataset scores above, return this EXACT JSON:
{{
  "success_probability": {success_probability},
  "sustainability_score": {sustainability_score},
  "score_explanations": {{
    "success_reason": "<1 specific sentence explaining why {success_probability}% success based on {domain} market data, competition={competition_level}, growth={growth_rate}%>",
    "sustainability_reason": "<1 specific sentence explaining why {sustainability_score}/100 sustainability based on social_impact={social_impact_score}, risk={risk_level}, env impact in {domain}>"
  }},
  "decision": {{
    "verdict": "<Proceed if success_probability>=55 AND sustainability_score>=60, Improve if either is 40-54, else Avoid>",
    "reason": "<1 sentence combining dataset evidence and user's specific idea>"
  }},
  "analysis": {{
    "market_demand": "<2 sentences about demand for this specific idea in {locality} based on {market_demand_score}/100 dataset score>",
    "competition": "<2 sentences about competition landscape for {idea} in {domain} — competition is {competition_level}>",
    "feasibility": "<2 sentences about feasibility for {budget} budget and {team_size} person team in Indian market>",
    "scalability": "<2 sentences about scaling potential — growth rate is {growth_rate}% YoY in dataset>"
  }},
  "improvements": [
    "<Specific improvement 1 tailored to this idea and budget>",
    "<Specific improvement 2 tailored to {locality} market>",
    "<Specific improvement 3 addressing the {competition_level} competition>"
  ],
  "new_ideas": [
    {{"title":"<title>","description":"<1 sentence>","market_potential":"High"}},
    {{"title":"<title>","description":"<1 sentence>","market_potential":"High"}},
    {{"title":"<title>","description":"<1 sentence>","market_potential":"Medium"}},
    {{"title":"<title>","description":"<1 sentence>","market_potential":"Medium"}},
    {{"title":"<title>","description":"<1 sentence>","market_potential":"Medium"}},
    {{"title":"<title>","description":"<1 sentence>","market_potential":"Low"}},
    {{"title":"<title>","description":"<1 sentence>","market_potential":"Low"}}
  ],
  "summary": "<2 sentences summarising this specific idea using the dataset scores as evidence>"
}}

Rules:
- Keep success_probability = {success_probability} and sustainability_score = {sustainability_score} exactly as given
- verdict: Proceed if success>=55 and sustain>=60, Improve if success>=40 or sustain>=45, else Avoid
- Use INR for all monetary values
- All analysis must reference the dataset scores and the user's specific inputs
"""


def _try_parse(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse AI response as JSON: {raw[:200]}")


def analyze_idea(locality: str, budget: str, team_size: int,
                 domain: str, idea: str, api_key: str = "",
                 dataset_scores: dict = None) -> dict:
    """
    Call OpenRouter API with dataset-calculated scores embedded in the prompt.
    The AI explains the scores and provides analysis — it does NOT generate new scores.

    dataset_scores: dict from get_domain_stats() containing:
        success_rate, sustainability_score, market_demand_score, feasibility_score,
        scalability_score, competition_level, growth_rate, risk_level,
        social_impact_score, funded_startups
    """
    key = api_key.strip() or os.environ.get("OPENROUTER_API_KEY", "").strip() or HARDCODED_API_KEY

    # Use dataset scores if provided, else use fallback defaults
    ds = dataset_scores or {}
    success_prob    = ds.get("success_rate", 50)
    sustain_score   = ds.get("sustainability_score", 60)
    mkt_demand      = ds.get("market_demand_score", 65)
    feasibility     = ds.get("feasibility_score", 60)
    scalability     = ds.get("scalability_score", 55)
    competition     = ds.get("competition_level", "Medium")
    growth          = ds.get("growth_rate", 15.0)
    risk            = ds.get("risk_level", "Medium")
    social_impact   = ds.get("social_impact_score", 65)
    funded_startups = ds.get("funded_startups", 0)

    prompt = ANALYSIS_TEMPLATE.format(
        locality=locality, budget=budget, team_size=team_size,
        domain=domain, idea=idea,
        success_probability=success_prob,
        sustainability_score=sustain_score,
        market_demand_score=mkt_demand,
        feasibility_score=feasibility,
        scalability_score=scalability,
        competition_level=competition,
        growth_rate=growth,
        risk_level=risk,
        social_impact_score=social_impact,
        funded_startups=funded_startups,
    )

    models_to_try = [
        "openrouter/auto",
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-2-9b-it:free",
    ]

    last_error = None
    for model in models_to_try:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1400,
        }).encode("utf-8")

        req = urllib.request.Request(
            OPENROUTER_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "StartupIQ Analyzer",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if "error" in body:
                last_error = body["error"]
                continue
            raw = body["choices"][0]["message"]["content"].strip()
            result = _try_parse(raw)
            # Force dataset scores regardless of what AI returned
            result["success_probability"]  = success_prob
            result["sustainability_score"] = sustain_score
            return result
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:200]}"
            continue
        except Exception as exc:
            last_error = str(exc)
            continue

    raise ValueError(f"All models failed. Last error: {last_error}")

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE MODE — User has no idea yet
# ═══════════════════════════════════════════════════════════════════════════════

GENERATE_SYSTEM = (
    "You are a startup mentor and venture capitalist specialised in the Indian ecosystem. "
    "You are given real dataset scores for a domain. Use them to generate relevant ideas. "
    "Return ONLY valid JSON. No markdown, no text outside JSON. Keep fields SHORT — 1-2 sentences."
)

GENERATE_TEMPLATE = """\
Generate 3 startup ideas for this entrepreneur profile.

Profile:
- City / Market : {locality}
- Budget        : {budget}
- Team Size     : {team_size} people
- Domain        : {domain}
- Context       : {context}

Real dataset benchmarks for {domain} (from 500 Indian startups 2024-2026):
- Success Rate     : {success_rate}%
- Sustainability   : {sustainability_score}/100
- Market Demand    : {market_demand_score}/100
- Competition      : {competition_level}
- Growth Rate      : {growth_rate}% YoY
- Risk Level       : {risk_level}

Return ONLY this JSON (no extra text):
{{
  "mode": "generate",
  "summary": "<1-2 sentences: why these ideas suit this person's profile and dataset>",
  "ideas": [
    {{
      "rank": 1,
      "title": "<startup name>",
      "tagline": "<one line: who + what>",
      "problem": "<1-2 sentences: real problem in {locality}>",
      "solution": "<1-2 sentences: how it solves it>",
      "target_customer": "<1 sentence: specific customer>",
      "revenue_model": "<1 sentence with INR numbers>",
      "market_potential": "High",
      "why_now": "<1 sentence: why this is the right time>",
      "success_probability": {success_rate},
      "sustainability_score": {sustainability_score}
    }},
    {{
      "rank": 2,
      "title": "<startup name>",
      "tagline": "<one line>",
      "problem": "<1-2 sentences>",
      "solution": "<1-2 sentences>",
      "target_customer": "<1 sentence>",
      "revenue_model": "<1 sentence with INR>",
      "market_potential": "Medium",
      "why_now": "<1 sentence>",
      "success_probability": <dataset_rate - 5>,
      "sustainability_score": <dataset_sustain - 5>
    }},
    {{
      "rank": 3,
      "title": "<startup name>",
      "tagline": "<one line>",
      "problem": "<1-2 sentences>",
      "solution": "<1-2 sentences>",
      "target_customer": "<1 sentence>",
      "revenue_model": "<1 sentence with INR>",
      "market_potential": "Medium",
      "why_now": "<1 sentence>",
      "success_probability": <dataset_rate - 8>,
      "sustainability_score": <dataset_sustain - 8>
    }}
  ],
  "best_idea_launch_plan": {{
    "title": "<rank 1 title>",
    "team_needed": "<1 sentence>",
    "biggest_risk": "<1 sentence>",
    "first_milestone": "<1 sentence: 90 day target>",
    "steps": [
      {{"step":1,"title":"Validate (Week 1-2)","action":"<1 sentence>","outcome":"<1 sentence>","cost":"<INR>"}},
      {{"step":2,"title":"Build MVP (Week 3-6)","action":"<1 sentence>","outcome":"<1 sentence>","cost":"<INR>"}},
      {{"step":3,"title":"First Customers (Week 7-8)","action":"<1 sentence>","outcome":"<1 sentence>","cost":"<INR>"}},
      {{"step":4,"title":"Start Earning (Month 3)","action":"<1 sentence>","outcome":"<1 sentence>","cost":"<INR>"}}
    ]
  }}
}}

Use INR for money. Be specific for {locality}."""


def generate_ideas(locality: str, budget: str, team_size: int,
                   domain: str, context: str = "", api_key: str = "",
                   dataset_scores: dict = None) -> dict:
    """
    MODE 1 — User has NO idea yet.
    Uses dataset scores to generate 3 relevant startup ideas + step-by-step launch plan.
    """
    key = api_key.strip() or os.environ.get("OPENROUTER_API_KEY", "").strip() or HARDCODED_API_KEY

    ds = dataset_scores or {}
    prompt = GENERATE_TEMPLATE.format(
        locality=locality, budget=budget, team_size=team_size,
        domain=domain, context=context or "First-time entrepreneur",
        success_rate=ds.get("success_rate", 50),
        sustainability_score=ds.get("sustainability_score", 60),
        market_demand_score=ds.get("market_demand_score", 65),
        competition_level=ds.get("competition_level", "Medium"),
        growth_rate=ds.get("growth_rate", 15.0),
        risk_level=ds.get("risk_level", "Medium"),
    )

    models_to_try = [
        "openrouter/auto",
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-2-9b-it:free",
    ]

    last_error = None
    for model in models_to_try:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": GENERATE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 1400,
        }).encode("utf-8")

        req = urllib.request.Request(
            OPENROUTER_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "StartupIQ Generator",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if "error" in body:
                last_error = body["error"]
                continue
            raw    = body["choices"][0]["message"]["content"].strip()
            result = _try_parse(raw)
            return result
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:200]}"
            continue
        except Exception as exc:
            last_error = str(exc)
            continue

    raise ValueError(f"All models failed. Last error: {last_error}")