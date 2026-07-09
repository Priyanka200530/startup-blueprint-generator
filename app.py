"""
Startup Blueprint Generator
============================
IBM watsonx Orchestrate + IBM Granite + RAG → complete startup blueprints.

Credential priority order for generation:
  1. IBM watsonx Orchestrate  (ORCHESTRATE_AGENT_URL + WATSONX_API_KEY)
  2. IBM Granite via watsonx.ai  (WATSONX_API_KEY + WATSONX_PROJECT_ID)
  3. Demo / mock mode  (no credentials needed – uses built-in sample data)

Quick setup:
  cp .env.example .env   # then fill in your credentials
  python app.py
"""

import os
import json
import logging
import re
import textwrap
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "sbg-dev-secret-change-in-production")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Read credentials (stripped of accidental whitespace)
# ---------------------------------------------------------------------------
WATSONX_API_KEY   = os.getenv("WATSONX_API_KEY", "").strip()
WATSONX_REGION    = os.getenv("WATSONX_REGION",  "eu-gb").strip()
GRANITE_MODEL_ID  = os.getenv("GRANITE_MODEL_ID",
                               "ibm/granite-13b-instruct-v2").strip()

# ── WATSONX_PROJECT_ID: only valid if it is a real UUID v4 ──────────────────
import re as _re
_raw_pid      = os.getenv("WATSONX_PROJECT_ID", "").strip()
_uuid4_re     = _re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    _re.IGNORECASE,
)
WATSONX_PROJECT_ID = _raw_pid if _uuid4_re.match(_raw_pid) else ""
if _raw_pid and not WATSONX_PROJECT_ID:
    logger.warning("WATSONX_PROJECT_ID '%s' is not a UUID v4 – Granite disabled.", _raw_pid)

# ── Orchestrate ─────────────────────────────────────────────────────────────
#
# IBM watsonx Orchestrate has TWO distinct URL concepts:
#
#  A) Instance management URL  (IBM Cloud console / service credentials)
#     https://api.eu-gb.watson-orchestrate.cloud.ibm.com/instances/<uuid>
#     → NOT callable for inference
#
#  B) Inference API base URL  (the callable REST API)
#     https://api.eu-gb.watson-orchestrate.cloud.ibm.com
#     POST /v1/chat/completions          ← agents endpoint
#     POST /v1/text/generation           ← direct LLM
#
# The app derives the inference base URL by stripping /instances/<uuid>
# from whatever is in WATSONX_URL.
# ---------------------------------------------------------------------------
_raw_wx_url = os.getenv("WATSONX_URL", "").strip().rstrip("/")

# Extract the base host (everything before /instances/...)
import re as _re2  # noqa: F811
_instance_match = _re2.match(
    r"(https://[^/]+?)(?:/instances/[0-9a-f-]+)?$",
    _raw_wx_url,
    _re2.IGNORECASE,
)
ORCHESTRATE_BASE_URL = _instance_match.group(1) if _instance_match else _raw_wx_url

# Extract the instance UUID (used only for logging / debug)
_inst_match = _re2.search(r"/instances/([0-9a-f-]+)", _raw_wx_url, _re2.IGNORECASE)
ORCHESTRATE_INSTANCE_ID = _inst_match.group(1) if _inst_match else ""

# Explicit override: if user sets ORCHESTRATE_AGENT_URL use it verbatim
ORCHESTRATE_AGENT_URL  = os.getenv("ORCHESTRATE_AGENT_URL", "").strip()

# Agent ID (optional – only needed if instance has multiple agents)
ORCHESTRATE_AGENT_ID   = os.getenv("ORCHESTRATE_AGENT_ID", "").strip()

# Watson Discovery (RAG) — all optional
DISCOVERY_API_KEY      = os.getenv("DISCOVERY_API_KEY", "").strip()
DISCOVERY_URL          = os.getenv("DISCOVERY_URL", "").strip()
DISCOVERY_PROJECT_ID   = os.getenv("DISCOVERY_PROJECT_ID", "").strip()
DISCOVERY_COLLECTION_ID= os.getenv("DISCOVERY_COLLECTION_ID", "").strip()

# Demo mode: no API key at all
DEMO_MODE = not WATSONX_API_KEY

# watsonx.ai ML endpoint (direct Granite only — not Orchestrate)
_ml_region     = WATSONX_REGION if WATSONX_REGION not in ("", "api.eu-gb") else "eu-gb"
WATSONX_ML_URL = f"https://{_ml_region}.ml.cloud.ibm.com"

logger.info("Demo mode             : %s", DEMO_MODE)
logger.info("API key present       : %s", bool(WATSONX_API_KEY))
logger.info("Raw WATSONX_URL       : %s", _raw_wx_url or "(not set)")
logger.info("Orchestrate base URL  : %s", ORCHESTRATE_BASE_URL or "(not set)")
logger.info("Orchestrate instance  : %s", ORCHESTRATE_INSTANCE_ID or "(not parsed)")
logger.info("Orchestrate agent ID  : %s", ORCHESTRATE_AGENT_ID or "(auto/default)")
logger.info("Explicit agent URL    : %s", ORCHESTRATE_AGENT_URL or "(auto)")
logger.info("Granite project ID    : %s", WATSONX_PROJECT_ID or "(not set)")

# ---------------------------------------------------------------------------
# IAM Token
# ---------------------------------------------------------------------------
_iam_cache: dict = {}


def get_iam_token() -> str:
    """Exchange IBM Cloud API key for a Bearer token (cached)."""
    import time
    now = time.time()
    if _iam_cache.get("token") and now < _iam_cache.get("expires_at", 0) - 60:
        return _iam_cache["token"]

    if not WATSONX_API_KEY:
        raise ValueError("WATSONX_API_KEY is not set in .env")

    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "apikey": WATSONX_API_KEY,
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _iam_cache["token"] = data["access_token"]
    _iam_cache["expires_at"] = now + data.get("expires_in", 3600)
    return _iam_cache["token"]


# ---------------------------------------------------------------------------
# RAG – Watson Discovery (gracefully skipped when not configured)
# ---------------------------------------------------------------------------

def retrieve_rag_context(query: str, top_k: int = 5) -> str:
    """Return relevant passages from Watson Discovery, or '' if not configured."""
    if not all([DISCOVERY_API_KEY, DISCOVERY_URL, DISCOVERY_PROJECT_ID]):
        return ""
    try:
        url = (f"{DISCOVERY_URL}/v2/projects/{DISCOVERY_PROJECT_ID}"
               f"/query?version=2023-03-31")
        payload = {"natural_language_query": query, "count": top_k}
        if DISCOVERY_COLLECTION_ID:
            payload["collection_ids"] = [DISCOVERY_COLLECTION_ID]
        resp = requests.post(url, json=payload,
                             auth=("apikey", DISCOVERY_API_KEY), timeout=30)
        resp.raise_for_status()
        passages = []
        for r in resp.json().get("results", []):
            dp = r.get("document_passages", [])
            if dp and isinstance(dp, list):
                passages.append(dp[0].get("passage_text", ""))
            elif r.get("text"):
                passages.append(str(r["text"])[:600])
        ctx = "\n\n".join(p for p in passages if p)
        logger.info("RAG: %d passages retrieved", len(passages))
        return ctx
    except Exception as exc:
        logger.error("RAG error: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# IBM watsonx Orchestrate  –  correct inference API
# ---------------------------------------------------------------------------

def _orchestrate_post(url: str, payload: dict, token: str) -> requests.Response:
    """
    POST to an Orchestrate endpoint.
    Tries Bearer (IAM) first; auto-retries with ZenApiKey on 401.
    """
    import base64

    def _post(auth: str) -> requests.Response:
        return requests.post(
            url,
            json=payload,
            headers={
                "Authorization": auth,
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            },
            timeout=120,
        )

    resp = _post(f"Bearer {token}")
    if resp.status_code == 401:
        zen = base64.b64encode(f"apikey:{WATSONX_API_KEY}".encode()).decode()
        logger.info("Bearer rejected – retrying with ZenApiKey")
        resp = _post(f"ZenApiKey {zen}")
    return resp


def _parse_orchestrate_response(data: dict) -> str:
    """Extract the text content from any known Orchestrate response shape."""
    # OpenAI chat-completions format
    if "choices" in data:
        content = (data["choices"][0]
                       .get("message", {})
                       .get("content", ""))
        if content:
            return content
    # Orchestrate-native keys
    for key in ("output", "response", "generated_text", "text", "result", "content"):
        if data.get(key):
            return str(data[key])
    logger.warning("Unrecognised Orchestrate response: %s", json.dumps(data)[:300])
    return json.dumps(data)


def call_orchestrate_agent(system_prompt: str, user_prompt: str) -> str:
    """
    Call IBM watsonx Orchestrate.  Tries candidate endpoints in order:

      1. ORCHESTRATE_AGENT_URL  (explicit override from .env)
      2. <base>/v1/chat/completions   (Orchestrate agents API)
      3. <base>/v1/text/generation    (Orchestrate direct-LLM)

    The base URL is derived by stripping /instances/<uuid> from WATSONX_URL,
    giving:  https://api.eu-gb.watson-orchestrate.cloud.ibm.com

    Configure in .env:
      WATSONX_URL          – your Orchestrate service URL (from IBM Cloud)
      WATSONX_API_KEY      – IBM Cloud API key
      ORCHESTRATE_AGENT_ID – (optional) specific agent ID
      ORCHESTRATE_AGENT_URL – (optional) full explicit endpoint override
    """
    if not ORCHESTRATE_BASE_URL and not ORCHESTRATE_AGENT_URL:
        raise ValueError(
            "No Orchestrate URL configured. Set WATSONX_URL in .env."
        )

    token = get_iam_token()

    # Build list of candidate endpoints to try, in priority order
    candidates: list[tuple[str, dict]] = []

    # Build the OpenAI-compatible chat payload (no stream key — causes 500)
    def _chat_payload() -> dict:
        p: dict = {
            "messages": [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            ],
        }
        if ORCHESTRATE_AGENT_ID:
            p["agent_id"] = ORCHESTRATE_AGENT_ID
        return p

    def _gen_payload() -> dict:
        p: dict = {
            "input": f"{system_prompt}\n\n{user_prompt}",
            "parameters": {"max_new_tokens": 3000, "temperature": 0.7},
        }
        if ORCHESTRATE_AGENT_ID:
            p["agent_id"] = ORCHESTRATE_AGENT_ID
        return p

    # ── Explicit override ──────────────────────────────────────────────────
    if ORCHESTRATE_AGENT_URL:
        candidates.append((ORCHESTRATE_AGENT_URL, _chat_payload()))

    # ── Auto-derived endpoints ─────────────────────────────────────────────
    if ORCHESTRATE_BASE_URL:
        candidates += [
            (f"{ORCHESTRATE_BASE_URL}/v1/chat/completions", _chat_payload()),
            (f"{ORCHESTRATE_BASE_URL}/v1/text/generation",  _gen_payload()),
        ]

    last_err: requests.HTTPError | None = None
    for url, body in candidates:
        logger.info("Trying Orchestrate endpoint: %s", url)
        try:
            resp = _orchestrate_post(url, body, token)
            logger.info("Orchestrate %s → HTTP %s", url, resp.status_code)

            if resp.status_code == 404:
                logger.warning("404 at %s – trying next candidate", url)
                continue

            if resp.status_code in (500, 502, 503):
                # Internal Orchestrate error – log full body for diagnosis
                logger.error(
                    "Orchestrate internal error %s at %s\nBody: %s",
                    resp.status_code, url, resp.text[:600],
                )
                err = requests.HTTPError(response=resp)
                last_err = err
                continue   # try next candidate rather than giving up immediately

            if not resp.ok:
                logger.error("Orchestrate HTTP %s at %s: %s",
                             resp.status_code, url, resp.text[:300])
                resp.raise_for_status()

            data = resp.json()
            logger.info("Orchestrate success at %s  keys=%s", url, list(data.keys()))
            return _parse_orchestrate_response(data)

        except requests.HTTPError as exc:
            last_err = exc
            status = exc.response.status_code if exc.response is not None else 0
            if status in (404, 500, 502, 503):
                continue   # non-fatal – try next candidate
            raise          # 401/403/422 etc. surface immediately

    # All candidates failed
    if last_err is not None:
        raise last_err
    raise ValueError("No Orchestrate candidates were tried – check WATSONX_URL in .env.")


# ---------------------------------------------------------------------------
# IBM Granite via watsonx.ai  (text generation)
# ---------------------------------------------------------------------------

def call_granite_watsonx(system_prompt: str, user_prompt: str) -> str:
    """
    Call IBM Granite via the watsonx.ai text generation REST API.

    Requires in .env:
      WATSONX_API_KEY    – IBM Cloud API key
      WATSONX_PROJECT_ID – watsonx.ai project ID

    The WATSONX_REGION (default eu-gb based on your WATSONX_URL) controls
    which regional ML endpoint is used.

    Raises ValueError if WATSONX_PROJECT_ID is missing.
    Raises requests.HTTPError on non-2xx responses.
    """
    if not WATSONX_PROJECT_ID:
        raise ValueError(
            "WATSONX_PROJECT_ID is not set in .env.\n"
            "Create a watsonx.ai project at https://dataplatform.cloud.ibm.com "
            "and paste the project ID into your .env file."
        )

    token = get_iam_token()
    full_prompt = (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_prompt}\n"
        "<|assistant|>\n"
    )
    url = f"{WATSONX_ML_URL}/ml/v1/text/generation?version=2024-03-19"
    payload = {
        "model_id": GRANITE_MODEL_ID,
        "project_id": WATSONX_PROJECT_ID,
        "input": full_prompt,
        "parameters": {
            "max_new_tokens": 3500,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        },
    }
    logger.info("Calling Granite at %s  model=%s", url, GRANITE_MODEL_ID)
    resp = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        timeout=180,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return results[0].get("generated_text", "") if results else ""


# ---------------------------------------------------------------------------
# Demo / mock blueprint  (used when no IBM credentials are present)
# ---------------------------------------------------------------------------

def generate_demo_blueprint(form_data: dict) -> dict:
    """
    Return a realistic hardcoded blueprint populated with the user's form values.
    This lets the full UI be tested without any IBM credentials.
    """
    name    = form_data.get("startup_name", "YourStartup")
    idea    = form_data.get("startup_idea", "an innovative product")
    ind     = form_data.get("industry", "Technology")
    stage   = form_data.get("startup_stage", "Idea")
    budget  = form_data.get("estimated_budget", "$50,000")
    country = form_data.get("country", "United States")
    cust    = form_data.get("target_customers", "SMEs and individuals")

    return {
        "executive_summary": (
            f"{name} is a {stage}-stage {ind} startup targeting {cust} in {country}. "
            f"The core idea — {idea} — addresses a real market gap. "
            f"With an estimated budget of {budget}, the venture is positioned for early traction "
            "within 90 days. This blueprint outlines the strategy, market opportunity, "
            "financial plan, and execution roadmap needed to succeed."
        ),
        "assumptions": [
            f"Total addressable market in {country} is large and growing",
            "Early adopters are willing to pay for a better solution",
            f"Budget of {budget} is sufficient for MVP development",
            "A small founding team (2–3 people) can execute the initial roadmap",
            "Digital-first go-to-market strategy will drive acquisition cost down",
        ],
        "problem_statement": (
            f"Current solutions in the {ind} space fail to address the specific needs "
            f"of {cust}. They are either too expensive, too complex, or lack the "
            "personalisation and accessibility that modern users demand."
        ),
        "proposed_solution": (
            f"{name} solves this by delivering {idea}. "
            "The product is built mobile-first, uses AI to personalise the experience, "
            "and is priced accessibly for the target segment."
        ),
        "business_model_canvas": {
            "key_partners":          f"Cloud providers, {ind} associations, distribution partners",
            "key_activities":        "Product development, customer acquisition, community building",
            "key_resources":         "Engineering team, IP, brand, data",
            "value_propositions":    f"Solving the core problem of {cust} faster and cheaper than alternatives",
            "customer_relationships":"Self-serve onboarding, in-app support, community forum",
            "channels":              "SEO, social media, referral programme, direct sales",
            "customer_segments":     cust,
            "cost_structure":        "Engineering (50%), Marketing (25%), Ops (15%), Legal (10%)",
            "revenue_streams":       "SaaS subscription, freemium upsell, professional services",
        },
        "target_customers": (
            f"Primary: {cust} in {country} aged 25–45, tech-comfortable, "
            "budget-conscious. Secondary: enterprise teams looking for scalable solutions."
        ),
        "market_analysis": (
            f"The {ind} market in {country} is estimated at $2–5B with a CAGR of 18%. "
            "Post-pandemic digital acceleration has created a receptive audience. "
            "Key trends include AI-first products, mobile adoption, and demand for "
            "self-serve tools that reduce dependency on expensive consultants."
        ),
        "competitor_analysis": (
            "Direct competitors include 2–3 established players with legacy UX and high prices. "
            "Indirect competitors are spreadsheets and manual workflows. "
            f"{name}'s differentiation lies in AI-powered personalisation, lower cost, "
            "and a modern mobile-first experience that incumbents cannot easily replicate."
        ),
        "unique_value_proposition": (
            f"{name} is the only {ind} platform that combines AI automation with "
            "an intuitive mobile experience — cutting time-to-value from weeks to minutes."
        ),
        "revenue_model": (
            "Freemium SaaS: free tier for individuals, $29/mo Starter, $99/mo Pro, "
            "$299/mo Business. Annual plans at 20% discount. "
            "Additional revenue from implementation services and API access."
        ),
        "estimated_budget": {
            "total":          budget,
            "technology":     "40%",
            "marketing":      "25%",
            "operations":     "15%",
            "legal":          "8%",
            "human_resources":"7%",
            "miscellaneous":  "5%",
        },
        "funding_options": [
            "Bootstrapping from founders' savings",
            "Friends & Family round",
            "Angel investors (target: $150K–$300K)",
            f"Government grants for {ind} in {country}",
            "Accelerator programmes (Y Combinator, Techstars, local equivalents)",
            "Revenue-based financing once MRR exceeds $10K",
        ],
        "government_schemes": [
            f"Startup India / DPIIT recognition (if applicable in {country})",
            "MSME digital innovation grants",
            "R&D tax credits for AI/ML development",
            "Export promotion grants for global expansion",
        ],
        "legal_requirements": [
            f"Register as a private limited company in {country}",
            "Trademark the brand name and logo",
            "Draft Terms of Service and Privacy Policy (GDPR / local data law compliant)",
            "Employment contracts and IP assignment agreements for all team members",
            "Obtain any sector-specific licences required for {ind}",
        ],
        "go_to_market_strategy": (
            "Phase 1 (Days 1–30): Launch a waitlist landing page, run targeted LinkedIn "
            "and Google ads, partner with 3 micro-influencers in the niche. "
            "Phase 2 (Days 31–60): Onboard first 50 paying beta users, collect feedback, "
            "iterate on core features, publish 2 case studies. "
            "Phase 3 (Days 61–90): Launch Product Hunt, pitch 10 angels, "
            "activate referral programme, target 200 paying users by Day 90."
        ),
        "risk_assessment": [
            {"risk": "Slow user acquisition",      "impact": "High",   "mitigation": "Diversify channels; increase paid spend if organic is slow"},
            {"risk": "Competitor price war",        "impact": "Medium", "mitigation": "Compete on UX and AI features, not price"},
            {"risk": "Regulatory changes",         "impact": "Medium", "mitigation": "Monitor policy, build compliance into roadmap"},
            {"risk": "Technical debt / scaling",   "impact": "Low",    "mitigation": "Cloud-native architecture from day one; load-test before launch"},
            {"risk": "Founder burnout",            "impact": "High",   "mitigation": "Hire a part-time ops person at Month 2; use async-first culture"},
        ],
        "suggested_investors": [
            "Sequoia Capital India",
            "Accel Partners",
            "Y Combinator",
            "500 Startups",
            "Local angel networks",
            "IBM Ventures",
            "Google for Startups",
            "Microsoft for Startups",
        ],
        "roadmap_90_days": [
            {
                "phase": "Days 1–30",
                "milestones": [
                    "Finalise product spec and wireframes",
                    "Set up cloud infrastructure",
                    "Build and launch MVP (core feature only)",
                    "Onboard 10 design-partner beta users",
                    "Create waitlist landing page",
                ],
            },
            {
                "phase": "Days 31–60",
                "milestones": [
                    "Incorporate the company legally",
                    "Reach 50 paying beta customers",
                    "Publish first 2 customer case studies",
                    "Launch referral programme",
                    "Begin angel investor outreach",
                ],
            },
            {
                "phase": "Days 61–90",
                "milestones": [
                    "Launch publicly on Product Hunt",
                    "Hit $5K MRR milestone",
                    "Close first angel cheque",
                    "Hire first customer-support/ops person",
                    "Plan Series A / seed deck",
                ],
            },
        ],
        "final_recommendations": (
            f"Focus relentlessly on the first 50 paying customers — their feedback "
            "will shape the product more than any market research. "
            "Keep burn low, iterate fast, and measure only 3 KPIs: activation rate, "
            "churn rate, and NPS. Once you hit $10K MRR consistently, raise a seed "
            "round to scale marketing and team. IBM watsonx integration will become "
            "a key differentiator as you scale — invest in it early."
        ),
        "swot": {
            "strengths": [
                f"Novel AI-powered approach in the {ind} sector",
                "Lean founding team with low overhead",
                "Mobile-first product for modern users",
                "Clear, measurable value proposition",
            ],
            "weaknesses": [
                "Limited brand recognition at launch",
                "Small team — single point of failure risk",
                f"Constrained initial budget of {budget}",
                "No existing customer base to upsell",
            ],
            "opportunities": [
                f"Fast-growing {ind} market in {country}",
                "AI/ML adoption curve still early",
                "Incumbents have poor NPS — ripe for disruption",
                "Government digital grants available",
            ],
            "threats": [
                "Well-funded competitors can copy features",
                "Regulatory uncertainty in AI-driven products",
                "Economic downturn reducing SME spend",
                "Difficulty attracting senior engineering talent",
            ],
        },
        "readiness_score": 68,
        "risk_level": "Medium",
        "funding_recommendation": (
            "Bootstrap to $5K MRR, then raise a $200K–$300K angel round to accelerate "
            "marketing and hire a second engineer."
        ),
        "government_scheme_recommendation": (
            f"Apply for the Startup India / MSME digital grant programme in {country} "
            "to offset early R&D costs."
        ),
        "_demo_mode": True,
    }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# Full JSON schema sent only to Granite (high token limit).
# Orchestrate gets a compact prompt to stay under its proxy token budget.
_BLUEPRINT_SCHEMA = """{
  "executive_summary":"...","assumptions":["..."],"problem_statement":"...",
  "proposed_solution":"...",
  "business_model_canvas":{"key_partners":"...","key_activities":"...",
    "key_resources":"...","value_propositions":"...","customer_relationships":"...",
    "channels":"...","customer_segments":"...","cost_structure":"...","revenue_streams":"..."},
  "target_customers":"...","market_analysis":"...","competitor_analysis":"...",
  "unique_value_proposition":"...","revenue_model":"...",
  "estimated_budget":{"total":"...","technology":"...","marketing":"...",
    "operations":"...","legal":"...","human_resources":"...","miscellaneous":"..."},
  "funding_options":["..."],"government_schemes":["..."],"legal_requirements":["..."],
  "go_to_market_strategy":"...",
  "risk_assessment":[{"risk":"...","impact":"High|Medium|Low","mitigation":"..."}],
  "suggested_investors":["..."],
  "roadmap_90_days":[
    {"phase":"Days 1-30","milestones":["..."]},
    {"phase":"Days 31-60","milestones":["..."]},
    {"phase":"Days 61-90","milestones":["..."]}],
  "final_recommendations":"...",
  "swot":{"strengths":["..."],"weaknesses":["..."],"opportunities":["..."],"threats":["..."]},
  "readiness_score":72,"risk_level":"Medium",
  "funding_recommendation":"...","government_scheme_recommendation":"..."
}"""


def build_blueprint_prompt(form_data: dict, rag_context: str,
                           compact: bool = False) -> tuple[str, str]:
    """
    Build the system + user prompt for blueprint generation.

    compact=True  → shorter prompt for Orchestrate (avoids proxy 500s)
    compact=False → full prompt with JSON schema for Granite direct
    """
    ctx_section = (f"\nContext: {rag_context[:400]}" if rag_context else "")

    system_prompt = (
        "You are an expert startup consultant. "
        "Generate a complete startup blueprint as a single valid JSON object. "
        "Return ONLY the JSON — no markdown, no extra text."
    )

    # Compact fields summary — fits in ~350 tokens
    fields = (
        f"Name: {form_data.get('startup_name','N/A')} | "
        f"Idea: {form_data.get('startup_idea','N/A')} | "
        f"Industry: {form_data.get('industry','N/A')} | "
        f"Customers: {form_data.get('target_customers','N/A')} | "
        f"Country: {form_data.get('country','N/A')} | "
        f"Budget: {form_data.get('estimated_budget','N/A')} | "
        f"Stage: {form_data.get('startup_stage','N/A')}"
    )

    if compact:
        # Short prompt for Orchestrate proxy — under 500 tokens total
        user_prompt = (
            f"Create a startup blueprint for:\n{fields}{ctx_section}\n\n"
            "Return a JSON object with keys: executive_summary, assumptions, "
            "problem_statement, proposed_solution, business_model_canvas, "
            "target_customers, market_analysis, competitor_analysis, "
            "unique_value_proposition, revenue_model, estimated_budget, "
            "funding_options, government_schemes, legal_requirements, "
            "go_to_market_strategy, risk_assessment, suggested_investors, "
            "roadmap_90_days, final_recommendations, swot, readiness_score, "
            "risk_level, funding_recommendation, government_scheme_recommendation."
        )
    else:
        # Full prompt with exact schema for Granite direct
        user_prompt = (
            f"Generate a startup blueprint for:\n{fields}{ctx_section}\n\n"
            f"Return ONLY this JSON structure:\n{_BLUEPRINT_SCHEMA}"
        )

    return system_prompt, user_prompt


def build_chat_prompt(message: str, context: dict) -> tuple[str, str]:
    system_prompt = (
        "You are a knowledgeable startup advisor. "
        "Answer questions about the startup concisely and helpfully."
    )
    # Keep context compact for Orchestrate
    ctx_keys = ["executive_summary", "unique_value_proposition",
                "market_analysis", "revenue_model", "readiness_score"]
    slim_ctx = {k: context[k] for k in ctx_keys if k in context} if context else {}
    ctx_str  = json.dumps(slim_ctx) if slim_ctx else "(no blueprint generated yet)"
    user_prompt = f"Startup blueprint summary: {ctx_str}\n\nQuestion: {message}"
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Unified call_ai() – tries all backends in order
# ---------------------------------------------------------------------------

def call_ai(system_prompt: str, user_prompt: str) -> str:
    """
    Attempts AI backends in this order:
      1. IBM watsonx Orchestrate  (requires a deployed agent – WXO-PROXY-11112E means none deployed)
      2. IBM Granite via watsonx.ai  (requires WATSONX_PROJECT_ID)

    Raises RuntimeError only when a backend is configured but returns a
    non-retryable error (401, 403, 422 etc.).
    Raises NoAgentError (subclass of RuntimeError) when Orchestrate returns 500
    with WXO-PROXY-11112E so the caller can fall back to demo mode gracefully.
    """
    errors = []

    # ── 1. Orchestrate ───────────────────────────────────────────────────────
    if ORCHESTRATE_BASE_URL or ORCHESTRATE_AGENT_URL:
        try:
            result = call_orchestrate_agent(system_prompt, user_prompt)
            if result:
                logger.info("Response via Orchestrate (%d chars)", len(result))
                return result
            errors.append("Orchestrate returned empty response")
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            body   = e.response.text[:300]  if e.response is not None else ""
            errors.append(f"Orchestrate HTTP {status}: {body}")
            logger.warning("Orchestrate failed: %s", errors[-1])
            # 500 WXO-PROXY-11112E = no agent deployed → fall through
            if status == 500 and "11112E" in body:
                raise NoAgentDeployedError(
                    "Orchestrate instance has no deployed agent (WXO-PROXY-11112E). "
                    "Deploy an agent in IBM Cloud → watsonx Orchestrate → Agents, "
                    "then set ORCHESTRATE_AGENT_ID in .env."
                )
        except NoAgentDeployedError:
            raise
        except Exception as e:
            errors.append(f"Orchestrate error: {e}")
            logger.warning("Orchestrate failed: %s", errors[-1])

    # ── 2. Granite via watsonx.ai ────────────────────────────────────────────
    if WATSONX_PROJECT_ID:
        try:
            result = call_granite_watsonx(system_prompt, user_prompt)
            if result:
                logger.info("Response via Granite/watsonx.ai (%d chars)", len(result))
                return result
            errors.append("Granite returned empty response")
        except requests.HTTPError as e:
            body = e.response.text[:300] if e.response is not None else ""
            errors.append(f"Granite HTTP {e.response.status_code}: {body}")
            logger.warning("Granite failed: %s", errors[-1])
        except Exception as e:
            errors.append(f"Granite error: {e}")
            logger.warning("Granite failed: %s", errors[-1])

    raise RuntimeError(
        "All AI backends failed. " + " | ".join(errors)
        if errors else
        "No AI backend is configured."
    )


class NoAgentDeployedError(RuntimeError):
    """Raised when Orchestrate has no deployed agent (WXO-PROXY-11112E)."""


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", now=datetime.now())


@app.route("/generator")
def generator():
    return render_template("generator.html", now=datetime.now())


@app.route("/api/generate-blueprint", methods=["POST"])
def generate_blueprint():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    required = ["startup_name", "startup_idea", "industry"]
    missing = [f for f in required if not data.get(f, "").strip()]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 422

    raw = ""
    try:
        # ── Demo mode (no API key at all) ──────────────────────────────────────
        if DEMO_MODE:
            logger.info("DEMO MODE – returning mock blueprint")
            bp = generate_demo_blueprint(data)
            return jsonify({"success": True, "blueprint": bp, "demo": True,
                            "demo_reason": "no_api_key"})

        # ── Live mode ──────────────────────────────────────────────────────────
        rag_context = retrieve_rag_context(
            f"{data.get('startup_idea','')} {data.get('industry','')} startup"
        )
        using_orchestrate = bool(ORCHESTRATE_BASE_URL or ORCHESTRATE_AGENT_URL)
        sys_p, usr_p = build_blueprint_prompt(
            data, rag_context, compact=using_orchestrate
        )
        raw = call_ai(sys_p, usr_p)

        # Clean model output
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)

        blueprint = json.loads(cleaned)
        return jsonify({"success": True, "blueprint": blueprint})

    except NoAgentDeployedError as exc:
        # Orchestrate instance exists but has no deployed agent.
        # Fall back to demo blueprint with a clear setup message.
        logger.warning("No Orchestrate agent deployed – serving demo blueprint. %s", exc)
        bp = generate_demo_blueprint(data)
        bp["_agent_setup_required"] = True
        return jsonify({
            "success": True,
            "blueprint": bp,
            "demo": True,
            "demo_reason": "no_agent_deployed",
            "setup_message": (
                "Your Orchestrate instance is reachable but has no deployed agent. "
                "Follow the setup steps at /api/probe to deploy one."
            ),
        })

    except json.JSONDecodeError as exc:
        logger.error("JSON parse error: %s\nRaw (first 500): %s", exc, raw[:500])
        return jsonify({
            "error": "AI returned an unexpected format. Please try again.",
            "detail": str(exc),
        }), 502
    except RuntimeError as exc:
        logger.error("AI backend error: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        body   = exc.response.text[:400]  if exc.response is not None else ""
        logger.error("IBM HTTP %s: %s", status, body)
        return jsonify({
            "error": f"IBM API returned HTTP {status}. Check credentials and endpoint.",
            "detail": body,
        }), 502
    except Exception as exc:
        logger.exception("Unexpected error in generate_blueprint")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "No message provided"}), 400

    try:
        if DEMO_MODE:
            msg = data["message"].lower()
            if any(w in msg for w in ("fund", "invest", "money", "capital")):
                reply = ("Great question! For early-stage funding consider bootstrapping, "
                         "angel investors, and accelerator programmes like Y Combinator or "
                         "Techstars. Government grants are also worth exploring for your industry.")
            elif any(w in msg for w in ("market", "competitor", "competition")):
                reply = ("Focus on your niche first. Identify 3 direct competitors, "
                         "study their weaknesses, and position your product to fill exactly "
                         "that gap. Differentiate on UX, price, or a specific feature they lack.")
            else:
                reply = ("That's a great question for your startup journey! "
                         "Focus on customer discovery first — talk to 20 potential users "
                         "before building anything. Validate your assumptions early and iterate fast. "
                         "\n\n(Demo mode active – connect IBM watsonx for real AI responses.)")
            return jsonify({"reply": reply, "demo": True})

        rag_ctx = retrieve_rag_context(data["message"])
        sys_p, usr_p = build_chat_prompt(
            data["message"], data.get("blueprint_context", {}))
        if rag_ctx:
            usr_p += f"\n\nKnowledge base context:\n{rag_ctx}"
        reply = call_ai(sys_p, usr_p)
        return jsonify({"reply": reply.strip()})

    except Exception as exc:
        logger.exception("Error in /api/chat")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def health():
    _has_orch = bool(ORCHESTRATE_BASE_URL or ORCHESTRATE_AGENT_URL)
    return jsonify({
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "api_key_set": bool(WATSONX_API_KEY),
        "project_id_set": bool(WATSONX_PROJECT_ID),
        "orchestrate_configured": _has_orch,
        "granite_model": GRANITE_MODEL_ID,
        "active_backend": (
            "Orchestrate" if _has_orch else
            "Granite/watsonx.ai" if WATSONX_PROJECT_ID else
            "DEMO"
        ),
    })


@app.route("/api/debug")
def debug_config():
    """Shows resolved configuration (no secrets) – helps diagnose 404/401 errors."""
    _has_orch = bool(ORCHESTRATE_BASE_URL or ORCHESTRATE_AGENT_URL)
    return jsonify({
        "demo_mode": DEMO_MODE,
        "api_key_present": bool(WATSONX_API_KEY),
        "api_key_prefix": WATSONX_API_KEY[:8] + "…" if WATSONX_API_KEY else "(not set)",
        "raw_watsonx_url": _raw_wx_url or "(not set)",
        "orchestrate_base_url_resolved": ORCHESTRATE_BASE_URL or "(not set)",
        "orchestrate_instance_id": ORCHESTRATE_INSTANCE_ID or "(not parsed)",
        "orchestrate_agent_id": ORCHESTRATE_AGENT_ID or "(not set – default agent)",
        "orchestrate_agent_url_override": ORCHESTRATE_AGENT_URL or "(auto)",
        "endpoints_will_try": [
            *([ ORCHESTRATE_AGENT_URL ] if ORCHESTRATE_AGENT_URL else []),
            *([ f"{ORCHESTRATE_BASE_URL}/v1/chat/completions",
                f"{ORCHESTRATE_BASE_URL}/v1/text/generation" ]
              if ORCHESTRATE_BASE_URL else []),
        ],
        "project_id": WATSONX_PROJECT_ID or "(not set – Granite disabled)",
        "granite_model": GRANITE_MODEL_ID,
        "ml_base_url": WATSONX_ML_URL,
        "rag_configured": bool(DISCOVERY_API_KEY and DISCOVERY_URL),
        "active_backend": (
            "Orchestrate" if _has_orch else
            "Granite/watsonx.ai" if WATSONX_PROJECT_ID else
            "DEMO (no credentials)"
        ),
    })


@app.route("/api/probe")
def probe():
    """
    Live diagnostic: sends a tiny test message to every candidate Orchestrate
    endpoint and returns the raw HTTP status + body for each.
    Visit http://localhost:5000/api/probe to diagnose 500 / 404 errors.
    """
    import base64
    if not WATSONX_API_KEY:
        return jsonify({"error": "WATSONX_API_KEY not set"}), 400

    results = []

    try:
        token = get_iam_token()
        iam_ok = True
    except Exception as e:
        return jsonify({"iam_token": "FAILED", "error": str(e)}), 500

    zen = base64.b64encode(f"apikey:{WATSONX_API_KEY}".encode()).decode()

    # Tiny test payloads
    chat_body = {"messages": [{"role": "user", "content": "Say hello in one word."}]}
    gen_body  = {"input": "Say hello in one word.", "parameters": {"max_new_tokens": 10}}

    candidates = []
    if ORCHESTRATE_AGENT_URL:
        candidates.append(("explicit_override", ORCHESTRATE_AGENT_URL, chat_body))
    if ORCHESTRATE_BASE_URL:
        candidates += [
            ("base/v1/chat/completions", f"{ORCHESTRATE_BASE_URL}/v1/chat/completions", chat_body),
            ("base/v1/text/generation",  f"{ORCHESTRATE_BASE_URL}/v1/text/generation",  gen_body),
        ]

    for label, url, body in candidates:
        for auth_name, auth_val in [
            ("Bearer", f"Bearer {token}"),
            ("ZenApiKey", f"ZenApiKey {zen}"),
        ]:
            try:
                r = requests.post(
                    url, json=body,
                    headers={"Authorization": auth_val,
                             "Content-Type": "application/json",
                             "Accept": "application/json"},
                    timeout=15,
                )
                results.append({
                    "label": label,
                    "auth": auth_name,
                    "url": url,
                    "status": r.status_code,
                    "body_preview": r.text[:400],
                })
                if r.ok:
                    break   # no need to try second auth if first succeeded
            except Exception as exc:
                results.append({
                    "label": label,
                    "auth": auth_name,
                    "url": url,
                    "status": "ERROR",
                    "body_preview": str(exc),
                })

    return jsonify({
        "iam_token_ok": iam_ok,
        "api_key_prefix": WATSONX_API_KEY[:8] + "…",
        "orchestrate_base": ORCHESTRATE_BASE_URL,
        "results": results,
        "advice": (
            "Look for a result with status 200. "
            "Use that label's url + auth combination. "
            "Set ORCHESTRATE_AGENT_URL in .env to the working URL."
        ),
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
