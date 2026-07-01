import time
import re
import threading
import hashlib
import json
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv not installed or .env not present; environment variables may be set externally
    pass
from datetime import datetime, timezone
from typing import Any, Optional, Literal
from urllib import request as urlrequest

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field


SendAs = Literal["vera", "merchant_on_behalf"]
CTA = Literal[
    "binary_yes_no",
    "binary_confirm_cancel",
    "multi_choice_slot",
    "open_ended",
    "none",
]


app = FastAPI()
START_TS = time.time()
LOCK = threading.RLock()


class StoredContext(BaseModel):
    version: int
    payload: dict[str, Any]
    delivered_at: str


# (scope, context_id) -> StoredContext
CONTEXTS: dict[tuple[str, str], StoredContext] = {}


class ConversationState(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    last_bot_body: Optional[str] = None
    last_merchant_reply_at: Optional[str] = None
    auto_reply_streak: int = 0
    ended: bool = False
    # New fields to support slot offering and reply resolution
    last_offered_slots: Optional[list[str]] = None
    last_template_name: Optional[str] = None
    last_template_params: Optional[list[Any]] = None
    last_offered_time: Optional[str] = None
    last_selection: Optional[str] = None
    reply_stage: str = "new"


def _clear_offer_state(state: ConversationState) -> None:
    state.last_offered_slots = None
    state.last_template_name = None
    state.last_template_params = None
    state.last_offered_time = None
    state.last_selection = None
    state.reply_stage = "new"


CONVERSATIONS: dict[str, ConversationState] = {}

# suppression_key -> last_sent_ts
SENT_SUPPRESSIONS: dict[str, float] = {}

# merchant_id -> consecutive auto-reply count (across conversations)
AUTO_REPLY_STREAK_BY_MERCHANT: dict[str, int] = {}

# merchant_id -> last body fingerprint to prevent cross-conversation repeats
LAST_BODY_HASH_BY_MERCHANT: dict[str, str] = {}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "moonshotai/kimi-k2")
_LLM_USE_RAW = os.getenv("LLM_USE", "auto").strip().lower()
USE_LLM = bool(OPENROUTER_API_KEY) if _LLM_USE_RAW in {"", "auto"} else _LLM_USE_RAW in {"1", "true", "yes", "on"}
ALLOWED_CTAS: set[str] = {"binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "open_ended", "none"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_dt(value: str) -> Optional[datetime]:
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def _is_expired(expires_at: Optional[str], now_iso: str) -> bool:
    if not expires_at:
        return False
    now_dt = _parse_iso_dt(now_iso)
    exp_dt = _parse_iso_dt(expires_at)
    if not now_dt or not exp_dt:
        return False
    return now_dt >= exp_dt


def _within_24h(last_reply_at: Optional[str], now_iso: str) -> bool:
    if not last_reply_at:
        return False
    last_dt = _parse_iso_dt(last_reply_at)
    now_dt = _parse_iso_dt(now_iso)
    if not last_dt or not now_dt:
        return False
    return (now_dt - last_dt).total_seconds() <= 24 * 3600


def _contexts_loaded_counts() -> dict[str, int]:
    counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    seen: dict[str, set[str]] = {k: set() for k in counts}
    for (scope, cid) in CONTEXTS.keys():
        if scope in seen:
            seen[scope].add(cid)
    for scope in counts:
        counts[scope] = len(seen[scope])
    return counts


def _has_url(text: str) -> bool:
    return bool(re.search(r"https?://", text, flags=re.IGNORECASE))


def _merchant_display_name(merchant: dict[str, Any]) -> str:
    identity = merchant.get("identity", {})
    return identity.get("owner_first_name") or identity.get("name") or "there"


def _merchant_languages(merchant: dict[str, Any]) -> list[str]:
    return list(merchant.get("identity", {}).get("languages", []) or [])


def _pick_hi_en(langs: list[str]) -> bool:
    # Heuristic: if Hindi is present, allow code-mix.
    return any(l in {"hi", "hi-en mix"} for l in langs)


def _is_opt_out(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ["stop", "unsubscribe", "don't message", "do not message", "spam"])


def _looks_like_auto_reply(text: str) -> bool:
    t = text.lower().strip()
    patterns = [
        "thank you for contacting",
        "our team will respond shortly",
        "we will get back to you",
        "this is an automated",
        "auto-reply",
    ]
    return any(p in t for p in patterns)


def _looks_like_commitment(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ["lets do it", "let's do it", "go ahead", "ok", "okay", "proceed", "what's next", "whats next", "yes do it"])


def _format_number(n: Any) -> str:
    try:
        if isinstance(n, float):
            return f"{n:.3f}".rstrip("0").rstrip(".")
        return str(int(n))
    except Exception:
        return str(n)


def _format_pct(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if abs(number) <= 1:
        return f"{number * 100:.1f}%"
    return f"{number:.1f}%"


def _body_fingerprint(body: str) -> str:
    normalized = re.sub(r"\s+", " ", body.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _slot_options_from_preference(pref: Optional[str]) -> list[str]:
    if not pref:
        return []
    p = pref.strip().lower()
    mapping: dict[str, list[str]] = {
        "weekday_evening": ["Wed 6:00 PM", "Thu 7:00 PM"],
        "weekday_morning": ["Tue 10:00 AM", "Thu 11:00 AM"],
        "weekend_morning": ["Sat 10:30 AM", "Sun 11:30 AM"],
        "weekend_evening": ["Sat 5:30 PM", "Sun 6:30 PM"],
    }
    if p in mapping:
        return mapping[p]
    if "evening" in p:
        return ["Wed 6:00 PM", "Thu 7:00 PM"]
    if "morning" in p:
        return ["Tue 10:00 AM", "Thu 11:00 AM"]
    if "weekend" in p:
        return ["Sat 11:00 AM", "Sun 5:00 PM"]
    return []


def _extract_slot_from_text(text: str, offered_slots: list[str]) -> Optional[str]:
    t = text.lower()
    for slot in offered_slots:
        if slot.lower() in t:
            return slot

    day_map = {
        "mon": "Mon",
        "monday": "Mon",
        "tue": "Tue",
        "tues": "Tue",
        "tuesday": "Tue",
        "wed": "Wed",
        "wednesday": "Wed",
        "thu": "Thu",
        "thur": "Thu",
        "thurs": "Thu",
        "thursday": "Thu",
        "fri": "Fri",
        "friday": "Fri",
        "sat": "Sat",
        "saturday": "Sat",
        "sun": "Sun",
        "sunday": "Sun",
    }
    m = re.search(r"\b(mon|monday|tue|tues|tuesday|wed|wednesday|thu|thur|thurs|thursday|fri|friday|sat|saturday|sun|sunday)\b[^0-9]*(\d{1,2}(?::\d{2})?\s*(?:am|pm))", t)
    if m:
        day = day_map.get(m.group(1), m.group(1).title())
        tm = re.sub(r"\s+", "", m.group(2).upper())
        return f"{day} {tm}"

    m2 = re.search(r"\b(today|tomorrow)\b[^0-9]*(\d{1,2}(?::\d{2})?\s*(?:am|pm))", t)
    if m2:
        when = m2.group(1).title()
        tm = re.sub(r"\s+", "", m2.group(2).upper())
        return f"{when} {tm}"

    return None


def _merchant_anchor(locality: Optional[str], views: Any, calls: Any, ctr: Any, peer_ctr: Any, offer_hint: Optional[str]) -> str:
    parts: list[str] = []
    if locality:
        parts.append(f"in {locality}")
    if views is not None:
        parts.append(f"views are {_format_number(views)}")
    if calls is not None:
        parts.append(f"calls are {_format_number(calls)}")
    if ctr is not None:
        peer_text = f" vs peer {_format_pct(peer_ctr)}" if peer_ctr is not None else ""
        parts.append(f"CTR is {_format_pct(ctr)}{peer_text}")
    if offer_hint:
        parts.append(f"your active offer is '{offer_hint}'")
    if not parts:
        return "I have your latest profile snapshot"
    if len(parts) == 1:
        return f"{parts[0].capitalize()}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _outcome_hint(kind: str, calls: Any, direction: Optional[str] = None) -> str:
    if isinstance(calls, (int, float)):
        lift = max(1, int(abs(float(calls)) * 0.15))
    else:
        lift = 2

    if kind == "perf_dip":
        return f"Can recover ~{lift} calls this week"
    if kind == "perf_spike":
        return f"Could convert momentum into {lift}+ extra calls"
    if kind == "review_theme_emerged":
        return "Protect response quality this week"
    if kind == "dormant_with_vera":
        return "Restart the lead flow"
    if kind == "festival_upcoming":
        return "Catch the festive rush"
    if kind == "competitor_opened":
        return "Lock in local recall before they gain traction"
    if kind == "renewal_due":
        return "Skip the visibility gap"
    if kind == "milestone_reached":
        return "Turn this into social proof"
    return "Improve discovery and response this week"


def _polish_merchant_body(
    fallback: dict[str, Any],
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not USE_LLM or not OPENROUTER_API_KEY:
        return fallback

    system = (
        "You rewrite a draft WhatsApp message for a merchant assistant. "
        "Preserve the facts, merchant name, CTA, and send_as exactly. "
        "Improve naturalness, persuasion, and category fit. "
        "Do NOT add facts, URLs, promises, or new claims. Output JSON with keys: body, rationale."
    )

    payload = {
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer,
        "draft": fallback,
    }

    user = (
        "Rewrite the draft body only, keeping the same meaning and factual anchors.\n"
        "Constraints:\n"
        "- Keep the CTA intent unchanged.\n"
        "- Keep merchant/customer names and exact facts.\n"
        "- Make it sound like a helpful colleague, not a system prompt.\n"
        "- Keep it concise.\n\n"
        f"CONTEXT:\n{json.dumps(payload, ensure_ascii=True)}"
    )

    req = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    try:
        request = urlrequest.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(req).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://magicpin.com",
            },
        )
        resp = urlrequest.urlopen(request, timeout=20)
        data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if not parsed:
            return fallback

        body = parsed.get("body")
        rationale = parsed.get("rationale") or fallback.get("rationale")
        if not body or not isinstance(body, str):
            return fallback

        fallback_body = str(fallback.get("body", ""))
        if fallback_body:
            fallback_nums = re.findall(r"\d+(?:\.\d+)?%?", fallback_body)
            body_nums = re.findall(r"\d+(?:\.\d+)?%?", body)
            if fallback_nums and len(body_nums) < len(fallback_nums):
                return fallback
            if len(body) < max(40, len(fallback_body) * 0.55) or len(body) > len(fallback_body) * 1.5:
                return fallback

        merged = dict(fallback)
        merged.update({"body": body, "rationale": rationale})
        return merged
    except Exception:
        return fallback


def _apply_category_guardrails(body: str, category: dict[str, Any], merchant: dict[str, Any], send_as: str) -> str:
    guarded = body
    voice = category.get("voice", {}) or {}
    taboo_list = list(voice.get("taboos", []) or []) + list(voice.get("vocab_taboo", []) or [])
    for taboo in taboo_list:
        t = str(taboo).strip()
        if not t:
            continue
        guarded = re.sub(re.escape(t), "", guarded, flags=re.IGNORECASE)

    guarded = re.sub(r"\s+", " ", guarded).strip()

    if send_as != "vera":
        return guarded

    slug = (category.get("slug") or "").lower()
    owner = _merchant_display_name(merchant)
    prefixes = {
        "dentists": f"Dr. {owner}",
        "gyms": f"Coach {owner}",
        "pharmacies": f"Pharmacist {owner}",
    }
    prefix = prefixes.get(slug)
    start_window = guarded[:120].lower()
    if prefix and prefix.lower() not in start_window:
        guarded = f"{prefix}, {guarded}"

    return guarded


def _decorate_message(composed: dict[str, Any], trigger: dict[str, Any], merchant: dict[str, Any], category: dict[str, Any]) -> dict[str, Any]:
    """Prepend a short, factual lead sentence derived from trigger payload to improve specificity."""
    body = composed.get("body", "") or ""
    kind = trigger.get("kind") or ""
    trg = trigger.get("payload", {}) or {}
    owner = merchant.get("identity", {}).get("owner_first_name") or _merchant_display_name(merchant)

    lead = ""
    if kind in {"perf_dip", "perf_spike"}:
        delta = trg.get("delta_pct")
        calls = merchant.get("performance", {}).get("calls")
        vs = trg.get("vs_baseline")
        if delta is not None:
            try:
                d_s = f"{int(float(delta) * 100)}%"
            except Exception:
                d_s = str(delta)
            lead = f"{owner}, in the recent window {('increase' if delta>0 else 'decrease')} of {d_s} observed." if d_s else ""
            if vs is not None:
                lead = f"{owner}, {d_s} vs baseline {vs}."
        elif calls is not None:
            lead = f"{owner}, current calls: {_format_number(calls)}." 

    elif kind == "renewal_due":
        days = trg.get("days_remaining") or merchant.get("subscription", {}).get("days_remaining")
        amount = trg.get("renewal_amount") or merchant.get("subscription", {}).get("renewal_amount")
        if days is not None:
            lead = f"{owner}, your plan renews in {days} days."
            if amount is not None:
                lead += f" Renewal amount: ₹{_format_number(amount)}."

    elif kind == "festival_upcoming":
        festival = trg.get("festival") or trg.get("payload", {}).get("festival") or trg.get("name")
        days = trg.get("days_until") or trg.get("days_to") or trg.get("payload", {}).get("days_until")
        if festival and days is not None:
            lead = f"{owner}, {festival} is in {days} days." 

    elif kind == "competitor_opened":
        comp = trg.get("competitor_name") or trg.get("payload", {}).get("competitor_name")
        offer = trg.get("their_offer") or trg.get("payload", {}).get("their_offer")
        if comp:
            lead = f"{owner}, competitor {comp} opened nearby."
            if offer:
                lead += f" Their offer: {offer}."

    # Only prepend if lead is non-empty and not already present
    if lead:
        norm = re.sub(r"\s+", " ", body.lower())
        if lead.lower().strip() not in norm:
            composed["body"] = lead + " " + body
    return composed


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def _llm_compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if not USE_LLM or not OPENROUTER_API_KEY:
        return fallback

    system = (
        "You are composing a single WhatsApp message for a merchant assistant. "
        "Use ONLY the provided context. Do NOT fabricate facts, offers, or sources. "
        "Keep a single clear CTA. Output JSON with keys: body, cta, send_as, rationale."
    )

    payload = {
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer,
    }

    user = (
        "Compose the next message as JSON.\n"
        "- Allowed CTA values: binary_yes_no, binary_confirm_cancel, multi_choice_slot, open_ended, none.\n"
        "- Respect category voice and taboos.\n"
        "- Use concrete, verifiable facts from context.\n"
        "- For customer scope, set send_as to merchant_on_behalf.\n"
        "- Keep body concise.\n\n"
        f"CONTEXT:\n{json.dumps(payload, ensure_ascii=True)}"
    )

    req = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 400,
    }

    try:
        request = urlrequest.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(req).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://magicpin.com",
            },
        )
        resp = urlrequest.urlopen(request, timeout=20)
        data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if not parsed:
            return fallback

        body = parsed.get("body")
        cta = parsed.get("cta")
        send_as = parsed.get("send_as")
        rationale = parsed.get("rationale")
        if not body or not cta or not send_as or not rationale:
            return fallback
        if cta not in ALLOWED_CTAS:
            return fallback
        if send_as not in {"vera", "merchant_on_behalf"}:
            return fallback

        merged = dict(fallback)
        merged.update({"body": body, "cta": cta, "send_as": send_as, "rationale": rationale})
        return merged
    except Exception:
        return fallback


def _consent_allows(customer: dict[str, Any], kind: str) -> bool:
    consent = customer.get("consent", {}) or {}
    scopes = set(consent.get("scope", []) or [])
    if not scopes:
        return False
    kind_to_scope = {
        "recall_due": "recall_reminders",
        "customer_lapsed_soft": "recall_reminders",
        "customer_lapsed_hard": "recall_reminders",
        "appointment_tomorrow": "appointment_reminders",
        "chronic_refill_due": "recall_reminders",
        "trial_followup": "treatment_followup",
    }
    required = kind_to_scope.get(kind)
    if not required:
        return True
    return required in scopes


def _category_voice_prefix(category: dict[str, Any]) -> str:
    slug = category.get("slug") or ""
    if slug == "dentists":
        return "Dr."
    if slug == "gyms":
        return "Coach"
    if slug == "pharmacies":
        return "Pharmacist"
    return ""


def _category_salutation(category: dict[str, Any], merchant_name: str) -> str:
    slug = category.get("slug") or ""
    if slug == "dentists":
        return f"Dr. {merchant_name}"
    if slug == "gyms":
        return f"Coach {merchant_name}"
    if slug == "pharmacies":
        return f"Pharmacist {merchant_name}"
    return merchant_name


def _category_action_noun(category: dict[str, Any]) -> str:
    slug = category.get("slug") or ""
    if slug == "dentists":
        return "patient calls"
    if slug == "gyms":
        return "membership leads"
    if slug == "salons":
        return "bookings"
    if slug == "restaurants":
        return "orders"
    if slug == "pharmacies":
        return "refills"
    return "inquiries"


def _category_style_hint(category: dict[str, Any]) -> str:
    slug = category.get("slug") or ""
    if slug == "gyms":
        return "coach"
    if slug == "salons":
        return "warm"
    if slug == "restaurants":
        return "operator"
    if slug == "pharmacies":
        return "precise"
    if slug == "dentists":
        return "professional"
    return "neutral"


def _category_perf_language(category: dict[str, Any], direction: str) -> str:
    """Returns category-specific performance language."""
    slug = category.get("slug") or ""
    if direction == "up":
        if slug == "gyms":
            return "members are signing up"
        if slug == "dentists":
            return "patient calls are climbing"
        if slug == "restaurants":
            return "orders are surging"
        if slug == "salons":
            return "bookings are up"
        if slug == "pharmacies":
            return "refill requests increased"
        return "leads are up"
    else:  # down
        if slug == "gyms":
            return "membership interest slipped"
        if slug == "dentists":
            return "patient inquiries declined"
        if slug == "restaurants":
            return "order flow dipped"
        if slug == "salons":
            return "booking rate dropped"
        if slug == "pharmacies":
            return "refill volume slid"
        return "inquiries dropped"


def _category_action_suggestion(category: dict[str, Any]) -> str:
    """Returns category-specific action suggestion."""
    slug = category.get("slug") or ""
    if slug == "gyms":
        return "limited-time membership offer or class promo"
    if slug == "dentists":
        return "preventive checkup offer or treatment discount"
    if slug == "restaurants":
        return "seasonal dish highlight or combo offer"
    if slug == "salons":
        return "seasonal service combo or referral reward"
    if slug == "pharmacies":
        return "health consultation offer or generic savings"
    return "targeted offer or service highlight"


def _performance_snapshot(merchant: dict[str, Any]) -> str:
    """Return a short, factual performance snapshot using available metrics."""
    perf = merchant.get("performance", {}) or {}
    parts: list[str] = []
    if perf.get("views") is not None:
        parts.append(f"views {_format_number(perf.get('views'))}")
    if perf.get("calls") is not None:
        parts.append(f"calls {_format_number(perf.get('calls'))}")
    if perf.get("ctr") is not None:
        parts.append(f"CTR {_format_pct(perf.get('ctr'))}")
    if not parts:
        return ""
    return "As of our latest snapshot: " + ", ".join(parts) + ". "


def _avoid_repeat(candidate: str, last: Optional[str]) -> str:
    if not last:
        return candidate
    if candidate.strip().lower() != last.strip().lower():
        return candidate
    return "Got it. Should I send the draft now? Reply YES." 


def _compose_message(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Deterministic baseline composer (no external LLM).

    Returns dict with keys: body, cta, send_as, suppression_key, rationale, template_name, template_params.
    """
    kind = trigger.get("kind") or "unknown"
    suppression_key = trigger.get("suppression_key") or f"{kind}:{merchant.get('merchant_id','')}"

    hi_en = _pick_hi_en(_merchant_languages(merchant))
    merchant_name = _merchant_display_name(merchant)
    owner_first = merchant.get("identity", {}).get("owner_first_name") or merchant_name
    biz_name = merchant.get("identity", {}).get("name", merchant_name)
    locality = merchant.get("identity", {}).get("locality")
    voice_prefix = _category_voice_prefix(category)
    style_hint = _category_style_hint(category)
    salutation = _category_salutation(category, merchant_name)
    action_noun = _category_action_noun(category)

    perf = merchant.get("performance", {})
    views = perf.get("views")
    calls = perf.get("calls")
    ctr = perf.get("ctr")

    peer = category.get("peer_stats", {})
    peer_ctr = peer.get("avg_ctr")

    active_offers = [o.get("title") for o in (merchant.get("offers", []) or []) if o.get("status") == "active" and o.get("title")]
    offer_hint = active_offers[0] if active_offers else None
    merchant_anchor = _merchant_anchor(locality, views, calls, ctr, peer_ctr, offer_hint)

    send_as: SendAs = "vera" if not customer else "merchant_on_behalf"

    if customer:
        cname = customer.get("identity", {}).get("name") or "there"
        lang_pref = (customer.get("identity", {}).get("language_pref") or "").lower()
        hi_en_customer = "hi" in lang_pref

        body = f"Hi {cname}, {biz_name} here. "
        if kind in {"recall_due", "customer_lapsed_soft"}:
            last_visit = customer.get("relationship", {}).get("last_visit")
            if last_visit:
                body += f"It’s been a while since your last visit ({last_visit}). "
            else:
                body += "Your check-in is due. "
            if offer_hint:
                body += f"Current offer: {offer_hint}. "
            pref = customer.get("preferences", {}).get("preferred_slots")
            offered_slots: list[str] = []
            if pref:
                readable_pref = pref.replace("_", " ")
                offered_slots = _slot_options_from_preference(pref)
                if len(offered_slots) >= 2:
                    body += f"Preferred window: {readable_pref}. Top slots: 1) {offered_slots[0]} 2) {offered_slots[1]}. Reply 1 or 2." 
                else:
                    body += f"Preferred slot: {readable_pref}. Reply 1 to take it, 2 to suggest another time." 
                cta = "multi_choice_slot"
            else:
                body += "Reply YES to book a slot."
                cta = "binary_yes_no"
            if hi_en_customer:
                if pref and len(offered_slots) >= 2:
                    body = f"Hi {cname}, {biz_name} here. Aapka follow-up due hai. " + (f"{offer_hint}. " if offer_hint else "") + f"Top slots: 1) {offered_slots[0]}, 2) {offered_slots[1]}. Reply 1 or 2."
                else:
                    body = f"Hi {cname}, {biz_name} here. Aapka follow-up due hai. " + (f"{offer_hint}. " if offer_hint else "") + ("Reply 1 for preferred time, 2 for alternate." if pref else "Reply YES to book.")
            rationale = "Customer-scoped recall/lapse follow-up using known visit history and active offer if present."
            template_name = "merchant_customer_followup_v1"
            template_params = [cname, biz_name, offer_hint or "follow-up due"]
            return {
                "body": body,
                "cta": cta,
                "send_as": send_as,
                "suppression_key": suppression_key,
                "rationale": rationale,
                "template_name": template_name,
                "template_params": template_params,
                "offered_slots": offered_slots,
            }

        if kind in {"appointment_tomorrow", "chronic_refill_due"}:
            pref = customer.get("preferences", {}).get("preferred_slots")
            offered_slots = _slot_options_from_preference(pref)
            body += "Quick reminder from your recent booking."
            if pref:
                if len(offered_slots) >= 2:
                    body += f" Top slots: 1) {offered_slots[0]} 2) {offered_slots[1]}. Reply 1 to confirm, 2 to change." 
                else:
                    body += f" Preferred time: {pref.replace('_', ' ')}. Reply 1 to confirm, 2 to change." 
                cta = "multi_choice_slot"
            else:
                body += " Reply YES if you'd like to confirm."
                cta = "binary_yes_no"
            if offer_hint:
                body += f" {offer_hint}."
            if hi_en_customer:
                if pref and len(offered_slots) >= 2:
                    body = f"Hi {cname}, {biz_name} here. Ek quick reminder - top slots: 1) {offered_slots[0]}, 2) {offered_slots[1]}. Reply 1 ya 2."
                else:
                    body = f"Hi {cname}, {biz_name} here. Ek quick reminder - confirm karna ho to reply 1, change karna ho to reply 2." if pref else f"Hi {cname}, {biz_name} here. Ek quick reminder - confirm karna ho to YES reply kar dijiye." 
            rationale = "Customer reminder with a simple confirmation CTA."
            template_name = "merchant_customer_reminder_v1"
            template_params = [cname, biz_name]
            return {
                "body": body,
                "cta": cta,
                "send_as": send_as,
                "suppression_key": suppression_key,
                "rationale": rationale,
                "template_name": template_name,
                "template_params": template_params,
                "offered_slots": offered_slots,
            }

        if kind in {"trial_followup", "customer_lapsed_hard"}:
            pref = customer.get("preferences", {}).get("preferred_slots")
            offered_slots = _slot_options_from_preference(pref)
            body += "We saved your earlier interest. Want to pick a slot this week?" 
            if pref:
                if len(offered_slots) >= 2:
                    body += f" Top slots: 1) {offered_slots[0]} 2) {offered_slots[1]}. Reply 1 to confirm, 2 to change." 
                else:
                    body += f" Preferred time: {pref.replace('_', ' ')}. Reply 1 to confirm, 2 to change." 
                cta = "multi_choice_slot"
            else:
                body += " Reply YES to proceed."
                cta = "binary_yes_no"
            if offer_hint:
                body += f" {offer_hint}."
            if hi_en_customer:
                if pref and len(offered_slots) >= 2:
                    body = f"Hi {cname}, {biz_name} here. Aapka follow-up pending hai - top slots: 1) {offered_slots[0]}, 2) {offered_slots[1]}. Reply 1 ya 2."
                else:
                    body = f"Hi {cname}, {biz_name} here. Aapka follow-up pending hai - slot confirm karna ho to reply 1, change karna ho to reply 2." if pref else f"Hi {cname}, {biz_name} here. Aapka follow-up pending hai - iss week slot book karna ho to YES reply kar dijiye." 
            rationale = "Customer trial/lapse follow-up with a simple YES/NO CTA."
            template_name = "merchant_customer_trial_v1"
            template_params = [cname, biz_name]
            return {
                "body": body,
                "cta": cta,
                "send_as": send_as,
                "suppression_key": suppression_key,
                "rationale": rationale,
                "template_name": template_name,
                "template_params": template_params,
                "offered_slots": offered_slots,
            }

        body += "Reply YES if you want to proceed."
        cta = "binary_yes_no"
        rationale = "Customer-scoped message with low-friction CTA."
        template_name = "merchant_customer_generic_v1"
        template_params = [cname, biz_name]
        return {
            "body": body,
            "cta": cta,
            "send_as": send_as,
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": template_name,
            "template_params": template_params,
            "offered_slots": [],
        }

    # Merchant-facing
    if kind == "research_digest":
        top_item_id = trigger.get("payload", {}).get("top_item_id")
        digest_items = category.get("digest", []) or []
        item = next((d for d in digest_items if d.get("id") == top_item_id), None) or (digest_items[0] if digest_items else None)
        title = (item or {}).get("title") or "a new research update"
        source = (item or {}).get("source")
        trial_n = (item or {}).get("trial_n")
        patient_segment = (item or {}).get("patient_segment")

        parts = []
        name_line = salutation
        owner_line = owner_first
        parts.append(f"{name_line}, saw this research:")
        parts.append(f"'{title}'")
        if trial_n:
            parts.append(f"({_format_number(trial_n)} patient study)")
        if patient_segment:
            parts.append(f"relevant to {patient_segment.replace('_', ' ')}")
        if source:
            parts.append(f"— {source}")
        parts.append(f"Worth a post? {_outcome_hint(kind, calls)} Reply YES and I'll draft a concise post for you.")

        body = " ".join(parts).replace("  ", " ").strip()
        perf_snap = _performance_snapshot(merchant)
        if perf_snap:
            body = perf_snap + body
        if hi_en:
            body = f"{salutation}, ek research mila: '{title}' {('— ' + f'{_format_number(trial_n)} patients' if trial_n else '')}. " + (f"Segment: {patient_segment.replace('_', ' ')}. " if patient_segment else "") + (f"Source: {source}. " if source else "") + f"{_outcome_hint(kind, calls)}. Kya post kar du?"

        cta = "binary_yes_no"
        rationale = "Uses category digest item; asks if worth posting, with research credibility."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_research_digest_v1",
            "template_params": [merchant_name, title, source or ""],
        }

    if kind in {"perf_spike", "perf_dip"}:
        direction = "up" if kind == "perf_spike" else "down"
        # Prefer trigger payload deltas when available; fall back to merchant perf deltas
        trg_payload = trigger.get("payload", {}) if trigger else {}
        delta = trg_payload.get("delta_pct") if trg_payload.get("delta_pct") is not None else perf.get("delta_7d", {}).get("views_pct")
        calls_delta = trg_payload.get("calls_pct") if trg_payload.get("calls_pct") is not None else perf.get("delta_7d", {}).get("calls_pct")
        try:
            delta_str = f"{int(float(delta) * 100)}%" if delta is not None else ""
        except Exception:
            delta_str = f"{delta}" if delta else ""
        try:
            calls_str = f"{int(float(calls_delta) * 100)}%" if calls_delta is not None else ""
        except Exception:
            calls_str = f"{calls_delta}" if calls_delta else ""

        name_line = salutation
        local = f" in {locality}" if locality else ""
        
        # Use category-specific language
        perf_lang = _category_perf_language(category, direction)
        action_suggestion = _category_action_suggestion(category)
        
        # Include trigger-level facts when available
        trg_payload = trigger.get("payload", {}) if trigger else {}
        window = trg_payload.get("window") or trg_payload.get("window_days") or "7d"
        vs_baseline = trg_payload.get("vs_baseline")
        body = f"{name_line}, {perf_lang} this week{local}. "
        # Add succinct trigger fact upfront when available, include prior baseline when we can compute it
        if delta_str:
            fact = f"In the last {window}, {('up' if direction=='up' else 'down')} {delta_str}"
            # compute prior value when possible to increase specificity
            prev_note = ""
            try:
                if views is not None and isinstance(views, (int, float)) and isinstance(delta, (int, float)) and (1 + float(delta)) != 0:
                    prev_views = int(float(views) / (1 + float(delta)))
                    prev_note = f" (from { _format_number(prev_views)} to {_format_number(views)})"
            except Exception:
                prev_note = ""
            if prev_note:
                fact += prev_note
            if vs_baseline:
                fact += f" vs baseline {vs_baseline}"
            fact += ". "
            # include last Vera interaction date if available
            last_ts = None
            history = merchant.get("conversation_history", []) or []
            if history:
                last_ts = history[-1].get("ts")
                if last_ts:
                    try:
                        last_date = last_ts.split("T")[0]
                        fact += f"Last contact: {last_date}. "
                    except Exception:
                        pass
            body = fact + body
        perf_snap = _performance_snapshot(merchant)
        if perf_snap:
            body = perf_snap + body
        body += f"Views: {_format_number(views) if views is not None else 'n/a'}, Calls: {_format_number(calls) if calls is not None else 'n/a'}, CTR: {_format_pct(ctr)} (vs peers {_format_pct(peer_ctr)}). "
        if calls_str:
            body += f"Calls shifted {calls_str}. "
        if delta_str:
            body += f"Views moved {delta_str}. "
        
        outcome = _outcome_hint(kind, calls, direction)
        # Concrete recommendation to improve decision quality
        if offer_hint:
            recommendation = f"Recommendation: promote your current offer '{offer_hint}' for 7 days and pin as a GBP post. I'll draft the post + a WhatsApp template."
        else:
            recommendation = f"Recommendation: run a short 7-day visibility post (carousel or offer) targeted to nearby searchers. I'll draft 2 captions + a WhatsApp template."
        # Estimate lift (fallback to small number if unknown)
        try:
            est_lift = max(1, int(abs(float(calls)) * 0.15)) if calls is not None else 2
        except Exception:
            est_lift = 2
        if kind == "perf_dip":
            cta_text = f"{recommendation} Reply YES and I'll prepare 2 ready-to-post messages + one short offer text — expect ~{est_lift} extra calls/week if applied."
        else:
            cta_text = f"{recommendation} Reply YES and I'll draft 2 post captions + one quick offer — could convert into ~{est_lift} extra calls/week."
        body += f"{outcome}. {cta_text}"

        if hi_en:
            momentum_word = "upar gaya" if direction == "up" else "neeche gaya"
            body = f"{name_line}, {perf_lang} - {momentum_word} {delta_str}. "
            body += f"Views {_format_number(views) if views is not None else 'n/a'}, calls {_format_number(calls) if calls is not None else 'n/a'}, CTR {_format_pct(ctr)} (peer avg {_format_pct(peer_ctr)}). "
            body += f"{outcome}. {('Kya draft kar du?' if kind == 'perf_dip' else 'Kya post kar du?')}"

        cta = "binary_yes_no"
        rationale = "Category-specific language with context-aware action suggestion."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_perf_nudge_v1",
            "template_params": [merchant_name, delta_str or direction, offer_hint or action_suggestion],
        }

    if kind == "review_theme_emerged":
        # Prefer trigger payload for review theme details when available
        trg_payload = trigger.get("payload", {}) if trigger else {}
        theme = trg_payload.get("theme") or (merchant.get("review_themes", []) or [{}])[0].get("theme")
        occurrences = trg_payload.get("occurrences_30d") or (merchant.get("review_themes", []) or [{}])[0].get("occurrences_30d")
        quote = trg_payload.get("common_quote") or (merchant.get("review_themes", []) or [{}])[0].get("common_quote")
        name_line = salutation
        peer_reviews = category.get("peer_stats", {}).get("avg_reviews")
        
        perf_snap = _performance_snapshot(merchant)
        body = f"{name_line}, your reviews are flagging something important."
        if perf_snap:
            body = perf_snap + body
        if theme and occurrences is not None:
            body = f"{name_line}, {occurrences} recent reviews mentioned: '{theme.replace('_', ' ')}'."
        if quote:
            body += f" Quote: '{quote}'."
        if peer_reviews is not None:
            body += f" (Peer avg: {_format_number(peer_reviews)} reviews)."
        
        body += f" {_outcome_hint(kind, calls)}. Want me to draft a response? Reply YES and I'll prepare a suggested reply you can send." 
        
        if hi_en:
            body = f"{salutation}, reviews me ek pattern clear aaya." 
            if theme and occurrences is not None:
                body += f" {occurrences} reviews ne '{theme.replace('_', ' ')}' mention kiya." 
            body += f" {_outcome_hint(kind, calls)}. Reply karo kya main draft kar du?"
        
        cta = "binary_yes_no"
        rationale = "Uses recent review-theme signals; directly offers to help with response."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_review_theme_v1",
            "template_params": [merchant_name, theme or "review"]
        }

    if kind == "renewal_due":
        days_left = merchant.get("subscription", {}).get("days_remaining")
        plan = merchant.get("subscription", {}).get("plan")
        days_text = f"{days_left} days" if isinstance(days_left, int) else "soon"
        trg_payload = trigger.get("payload", {}) if trigger else {}
        amount = trg_payload.get("renewal_amount") or merchant.get("subscription", {}).get("renewal_amount")
        amount_text = f" Renewal amount: ₹{_format_number(amount)}." if amount is not None else ""

        perf_snap = _performance_snapshot(merchant)
        body = f"{salutation}, your {plan or 'plan'} renewal is due {days_text}.{amount_text} " 
        if perf_snap:
            body = perf_snap + body
        body += f"{_outcome_hint(kind, calls)}. Need me to send renewal options? Reply YES and I'll send tailored options with pricing."
        
        if hi_en:
            body = f"{salutation}, aapka {plan or 'plan'} renewal {days_text} me expire hone wala hai. {_outcome_hint(kind, calls)}. Kya renewal options bhej du?"
        
        cta = "binary_yes_no"
        rationale = "Uses subscription timing; directly offers to help with renewal process."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_renewal_due_v1",
            "template_params": [merchant_name, days_text]
        }

    if kind == "milestone_reached":
        milestone_data = trigger.get("payload", {}).get("milestone") or "a big milestone"
        achievement = f"You hit {milestone_data}! 🎯" if milestone_data and milestone_data != "a big milestone" else "You reached a milestone! 🎯"
        perf_snap = _performance_snapshot(merchant)
        body = f"{salutation}, {achievement} {_outcome_hint(kind, calls)}. Should I turn this into a celebratory post your customers will love? Reply YES and I'll draft a short post with visuals suggestion."
        if perf_snap:
            body = perf_snap + body
        if hi_en:
            body = f"{salutation}, iss week ek milestone achieve hua! 🎯 {_outcome_hint(kind, calls)}. Kya social proof ke saath post banate hain?"
        cta = "binary_yes_no"
        rationale = "Milestone with specific achievement detail; social proof angle drives engagement."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_milestone_v1",
            "template_params": [merchant_name]
        }

    if kind == "competitor_opened":
        local_ref = f" in {locality}" if locality else ""
        perf_snap = _performance_snapshot(merchant)
        comp_name = trigger.get("payload", {}).get("competitor_name") or None
        comp_offer = trigger.get("payload", {}).get("their_offer") or None
        extra = ""
        if comp_name:
            extra = f" Competitor: {comp_name}."
        if comp_offer:
            extra += f" Their offer: {comp_offer}."
        body = f"{salutation}, a new competitor just listed nearby{local_ref}.{extra} {_outcome_hint(kind, calls)}. Want me to draft a differentiation post? Reply YES and I'll prepare a strong post highlighting your strengths."
        if perf_snap:
            body = perf_snap + body
        if hi_en:
            body = f"{salutation}, nearby area me ek naya shop aaya hai.{(' ' + comp_name) if comp_name else ''} {_outcome_hint(kind, calls)}. Kya main ek strong post draft kar du to aapko stand out karun?"
        cta = "binary_yes_no"
        rationale = "Competitor trigger; directly offers strategic post to defend position."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_competitor_v1",
            "template_params": [merchant_name, locality or ""]
        }

    if kind == "festival_upcoming":
        festival = trigger.get("payload", {}).get("festival") or "festival"
        when = trigger.get("payload", {}).get("days_to")
        when_text = f"in {when} days" if when is not None else "coming up"
        local = f" in {locality}" if locality else ""
        
        perf_snap = _performance_snapshot(merchant)
        if offer_hint:
            rec = f"Recommendation: feature '{offer_hint}' with a festive combo and run a short pinned post. I'll draft 2 caption variants + one WhatsApp template you can copy."
            body = f"{salutation}, {festival} {when_text}! 🎉 {_outcome_hint(kind, calls)}. {rec} Reply YES and I'll draft the assets."
        else:
            rec = "Recommendation: run a short pinned festive post with a clear offer or menu highlight. I'll draft 2 caption variants + one WhatsApp template you can copy."
            body = f"{salutation}, {festival} {when_text}{local}! 🎉 {_outcome_hint(kind, calls)}. {rec} Reply YES and I'll draft the assets."
        if perf_snap:
            body = perf_snap + body
        if hi_en:
            body = f"{salutation}, {festival} {when_text}! 🎉 {_outcome_hint(kind, calls)}. Kya main ek strong post draft kar du?"
        
        cta = "binary_yes_no"
        rationale = "Festival timing trigger; directly offers to create ready-to-post asset."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_festival_post_v1",
            "template_params": [merchant_name, festival, when_text],
        }

    if kind == "curious_ask_due":
        local_ref = f" {locality}" if locality else ""
        perf_snap = _performance_snapshot(merchant)
        body = f"Hi {salutation}, quick question: what's the #1 {action_noun} you're getting asked about{local_ref} this week? {_outcome_hint(kind, calls)}. I'll turn it into a Google Post + WhatsApp template. Reply with your answer and I'll draft the assets."
        if perf_snap:
            body = perf_snap + body
        if hi_en:
            body = f"Hi {salutation}, ek quick sawal: iss week sabse zyada kis {action_noun} ka pooch rahe ho customers? {_outcome_hint(kind, calls)}. Main uska Google Post + WhatsApp draft banata hoon."
        cta = "open_ended"
        rationale = "Asks merchant a specific, low-effort question; offers to do the work of drafting."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_curious_ask_v1",
            "template_params": [merchant_name, locality or ""],
        }

    if kind == "dormant_with_vera":
        last_ts = None
        history = merchant.get("conversation_history", []) or []
        if history:
            last_ts = history[-1].get("ts")
        
        perf_snap = _performance_snapshot(merchant)
        body = f"{salutation}, it's been a while. {_outcome_hint(kind, calls)}. Got 2 minutes for a quick refresh? Reply YES and I'll propose 2 quick actions to restart leads."
        if perf_snap:
            body = perf_snap + body
        if last_ts:
            body = f"{salutation}, last message was {last_ts.split('T')[0]}. {_outcome_hint(kind, calls)}. Free for a 2-minute catch up?"
        
        if hi_en:
            body = f"{salutation}, kaafi din ho gaye. {_outcome_hint(kind, calls)}. 2 minute ka update sun sakta hai?"
        
        cta = "binary_yes_no"
        rationale = "Dormancy trigger; offers quick, low-time-commitment re-engagement."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_dormant_nudge_v1",
            "template_params": [merchant_name],
        }

    # Fallback - data-driven for unknown trigger types
    topic = trigger.get("payload", {}).get("metric_or_topic")
    
    # Build specificity from available merchant data
    data_points = []
    if calls is not None:
        data_points.append(f"{_format_number(calls)} calls/week")
    if views is not None:
        data_points.append(f"{_format_number(views)} views")
    if ctr is not None:
        data_points.append(f"CTR {_format_pct(ctr)}")
    
    context = ", ".join(data_points) if data_points else "your performance"
    outcome = _outcome_hint(kind, calls)
    
    body = f"{salutation}, {context}: {outcome}. {('on ' + topic + ': ' if topic else '')}Worth exploring? Reply YES and I'll draft a starting post."
    perf_snap = _performance_snapshot(merchant)
    if perf_snap:
        body = perf_snap + body
    if hi_en:
        body = f"{salutation}, {context}. {outcome}. Kya draft kar du?"
    
    cta = "binary_yes_no"
    rationale = f"Fallback with merchant metrics; works for unhandled trigger types."
    return {
        "body": body,
        "cta": cta,
        "send_as": "vera",
        "suppression_key": suppression_key,
        "rationale": rationale,
        "template_name": "vera_generic_v1",
        "template_params": [merchant_name],
    }


KIND_PRIORITY: dict[str, int] = {
    "recall_due": 90,
    "appointment_tomorrow": 85,
    "chronic_refill_due": 80,
    "customer_lapsed_soft": 75,
    "customer_lapsed_hard": 72,
    "trial_followup": 70,
    "perf_dip": 70,
    "perf_spike": 60,
    "review_theme_emerged": 58,
    "milestone_reached": 55,
    "research_digest": 50,
    "dormant_with_vera": 45,
    "festival_upcoming": 40,
    "curious_ask_due": 35,
    "competitor_opened": 35,
    "renewal_due": 30,
}


class HealthzResponse(BaseModel):
    status: str
    uptime_seconds: int
    contexts_loaded: dict[str, int]


@app.get("/v1/healthz")
def healthz() -> HealthzResponse:
    with LOCK:
        return HealthzResponse(
            status="ok",
            uptime_seconds=int(time.time() - START_TS),
            contexts_loaded=_contexts_loaded_counts(),
        )


@app.get("/v1/metadata")
def metadata() -> dict[str, Any]:
    # Keep this small; judge reads it but doesn't rely on it.
    return {
        "team_name": "local-dev",
        "team_members": ["you"],
        "model": "openrouter" if USE_LLM else "rule-based (no LLM)",
        "approach": "LLM composer + deterministic routing" if USE_LLM else "deterministic templates + trigger routing + basic reply handling",
        "contact_email": "",
        "version": "0.1.0",
        "submitted_at": _now_iso(),
    }


class ContextPushBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int = Field(ge=1)
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
def push_context(body: ContextPushBody, response: Response) -> dict[str, Any]:
    with LOCK:
        key = (body.scope, body.context_id)
        # Challenge mode: always treat incoming judge pushes as source of truth.
        CONTEXTS[key] = StoredContext(version=body.version, payload=body.payload, delivered_at=body.delivered_at)

        return {
            "accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": _now_iso(),
        }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
def tick(body: TickBody) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []

    with LOCK:
        # Build candidate triggers with priority; pick best per merchant.
        candidates: list[tuple[int, int, str, dict[str, Any]]] = []
        for trg_id in body.available_triggers:
            stored = CONTEXTS.get(("trigger", trg_id))
            if not stored:
                continue
            trg = stored.payload
            if _is_expired(trg.get("expires_at"), body.now):
                continue
            suppression_key = trg.get("suppression_key")
            if suppression_key and suppression_key in SENT_SUPPRESSIONS:
                continue
            urgency = int(trg.get("urgency", 0) or 0)
            kind = trg.get("kind") or "unknown"
            priority = KIND_PRIORITY.get(kind, 10)
            merchant_id = trg.get("merchant_id")
            if not merchant_id:
                continue
            candidates.append((urgency, priority, trg_id, trg))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # FIX: Allow multiple triggers per merchant instead of limiting to one per merchant.
        # Each trigger is independently processed and added to actions.
        picked_triggers: list[tuple[str, dict[str, Any]]] = []
        merchant_count: dict[str, int] = {}
        
        for _, __, trg_id, trg in candidates:
            merchant_id = trg.get("merchant_id")
            # Allow up to 3 triggers per merchant (instead of just 1)
            if merchant_count.get(merchant_id, 0) >= 3:
                continue
            picked_triggers.append((trg_id, trg))
            merchant_count[merchant_id] = merchant_count.get(merchant_id, 0) + 1
            if len(picked_triggers) >= 50:  # Increased from 20 to allow more triggers
                break

        for trg_id, trg in picked_triggers:
            merchant_id = trg.get("merchant_id")
            customer_id = trg.get("customer_id")
            merchant = CONTEXTS.get(("merchant", merchant_id)).payload if ("merchant", merchant_id) in CONTEXTS else None
            if not merchant:
                merchant = {"merchant_id": merchant_id, "identity": {"name": "there"}, "performance": {}, "offers": []}
            category_slug = merchant.get("category_slug")
            category = CONTEXTS.get(("category", category_slug)).payload if category_slug and ("category", category_slug) in CONTEXTS else None
            if not category:
                category = {"slug": "general", "peer_stats": {}}
            customer = CONTEXTS.get(("customer", customer_id)).payload if customer_id and ("customer", customer_id) in CONTEXTS else None
            if trg.get("scope") == "customer" and customer:
                if not _consent_allows(customer, trg.get("kind") or ""):
                    continue

            composed = _compose_message(category=category, merchant=merchant, trigger=trg, customer=customer)
            # Prepend a concise trigger-linked factual lead to improve specificity
            composed = _decorate_message(composed, trg, merchant, category)
            composed = _llm_compose(category, merchant, trg, customer, composed)
            composed = _polish_merchant_body(composed, category, merchant, trg, customer)
            composed["body"] = _apply_category_guardrails(
                composed.get("body", ""),
                category,
                merchant,
                composed.get("send_as", "vera"),
            )

            # Safety: do not emit URLs.
            if _has_url(composed["body"]):
                composed["body"] = re.sub(r"https?://\S+", "", composed["body"]).strip()

            fingerprint = _body_fingerprint(composed["body"])
            if LAST_BODY_HASH_BY_MERCHANT.get(merchant_id) == fingerprint:
                continue

            conv_id = f"conv_{merchant_id}_{trg_id}"
            if conv_id not in CONVERSATIONS:
                CONVERSATIONS[conv_id] = ConversationState(
                    conversation_id=conv_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id,
                    trigger_id=trg_id,
                    last_bot_body=composed["body"],
                )

            conv_state = CONVERSATIONS.get(conv_id)
            in_session = _within_24h(conv_state.last_merchant_reply_at if conv_state else None, body.now)

            # A new outbound invalidates any previous unresolved slot offer.
            _clear_offer_state(conv_state)

            # Populate last offered slots / template metadata when we just composed a multi-choice CTA
            try:
                if composed.get("cta") == "multi_choice_slot":
                    offered = composed.get("offered_slots") or []
                    offered_str = [str(x) for x in offered if str(x).strip()]
                    conv_state.last_offered_slots = offered_str if offered_str else None
                    conv_state.last_template_name = composed.get("template_name")
                    conv_state.last_template_params = composed.get("template_params") or []
                    conv_state.last_offered_time = body.now
                    conv_state.reply_stage = "offered"
                elif customer_id:
                    conv_state.reply_stage = "new"
            except Exception:
                pass

            action = {
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": composed["send_as"],
                "trigger_id": trg_id,
                "template_name": composed.get("template_name", "vera_generic_v1") if not in_session else "",
                "template_params": composed.get("template_params", []) if not in_session else [],
                "body": composed["body"],
                "cta": composed["cta"],
                "suppression_key": composed["suppression_key"],
                "rationale": composed["rationale"],
            }

            actions.append(action)
            suppression_key = composed.get("suppression_key")
            if suppression_key:
                SENT_SUPPRESSIONS[suppression_key] = time.time()
            LAST_BODY_HASH_BY_MERCHANT[merchant_id] = fingerprint

    return {"actions": actions}


class DebugComposeRequest(BaseModel):
    trigger_id: str


@app.post("/v1/debug_compose")
def debug_compose(body: DebugComposeRequest) -> dict[str, Any]:
    """Return intermediate compositions for a trigger (fallback, decorated, llm_polished, final).

    Useful for local debugging and focused LLM polishing inspection.
    """
    trg_id = body.trigger_id
    stored = CONTEXTS.get(("trigger", trg_id))
    if not stored:
        return {"error": "trigger not found"}
    trg = stored.payload
    merchant_id = trg.get("merchant_id")
    merchant = CONTEXTS.get(("merchant", merchant_id)).payload if ("merchant", merchant_id) in CONTEXTS else None
    category_slug = merchant.get("category_slug") if merchant else None
    category = CONTEXTS.get(("category", category_slug)).payload if category_slug and ("category", category_slug) in CONTEXTS else None
    customer_id = trg.get("customer_id")
    customer = CONTEXTS.get(("customer", customer_id)).payload if customer_id and ("customer", customer_id) in CONTEXTS else None

    fallback = _compose_message(category or {"slug":"general"}, merchant or {}, trg, customer)
    decorated = _decorate_message(dict(fallback), trg, merchant or {}, category or {})
    llm_polished = _llm_compose(category or {}, merchant or {}, trg, customer, dict(decorated))
    final = _polish_merchant_body(dict(llm_polished), category or {}, merchant or {}, trg, customer)
    final_body = _apply_category_guardrails(final.get("body", ""), category or {}, merchant or {}, final.get("send_as", "vera"))

    return {
        "trigger_id": trg_id,
        "fallback": fallback,
        "decorated": decorated,
        "llm_polished": llm_polished,
        "final": {**final, "body": final_body},
    }


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: Literal["merchant", "customer"]
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
def reply(body: ReplyBody) -> dict[str, Any]:
    with LOCK:
        state = CONVERSATIONS.get(body.conversation_id)
        if not state:
            state = ConversationState(conversation_id=body.conversation_id, merchant_id=body.merchant_id, customer_id=body.customer_id)
            CONVERSATIONS[body.conversation_id] = state

        if state.ended:
            return {"action": "end", "rationale": "Conversation already ended."}

        msg = body.message.strip()
        state.last_merchant_reply_at = body.received_at
        is_customer = body.from_role == "customer"

        if _is_opt_out(msg):
            state.ended = True
            _clear_offer_state(state)
            state.reply_stage = "ended"
            return {"action": "end", "rationale": "Opt-out detected; ending conversation."}

        if _looks_like_auto_reply(msg):
            state.auto_reply_streak += 1
            if state.merchant_id:
                AUTO_REPLY_STREAK_BY_MERCHANT[state.merchant_id] = AUTO_REPLY_STREAK_BY_MERCHANT.get(state.merchant_id, 0) + 1
            total_streak = AUTO_REPLY_STREAK_BY_MERCHANT.get(state.merchant_id or "", state.auto_reply_streak)
            if total_streak >= 3:
                state.ended = True
                _clear_offer_state(state)
                state.reply_stage = "ended"
                return {"action": "end", "rationale": "Auto-reply repeated 3x; closing."}
            if total_streak == 2:
                return {"action": "wait", "wait_seconds": 86400, "rationale": "Auto-reply repeated; waiting 24h for owner."}
            return {"action": "wait", "wait_seconds": 14400, "rationale": "Detected auto-reply; waiting 4h."}

        # Reset streak on real reply
        state.auto_reply_streak = 0
        if state.merchant_id:
            AUTO_REPLY_STREAK_BY_MERCHANT[state.merchant_id] = 0

        # Helper: parse a numeric choice from the message (supports digits and small words)
        def _parse_numeric_choice(text: str) -> Optional[int]:
            # Try digits first
            m = re.search(r"\b(\d+)\b", text)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
            # Try common word numbers
            words = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
            }
            t = text.strip().lower()
            if t in words:
                return words[t]
            # Leading like "1." or "1)"
            m2 = re.match(r"^(\d+)[\.)]", text.strip())
            if m2:
                try:
                    return int(m2.group(1))
                except Exception:
                    pass
            return None

        # If customer sent a numeric choice and we have offered slots, resolve it
        if is_customer:
            choice = _parse_numeric_choice(msg)
            if choice is not None and state.last_offered_slots:
                idx = choice - 1
                if 0 <= idx < len(state.last_offered_slots):
                    selected = state.last_offered_slots[idx]
                    cust_name = "there"
                    if state.customer_id:
                        stored_c = CONTEXTS.get(("customer", state.customer_id))
                        c = stored_c.payload if stored_c else {}
                        cust_name = c.get("identity", {}).get("name") or c.get("name") or cust_name

                    # Compose a customer-voiced confirmation asking for final confirm
                    confirm_body = f"Booked {selected} for you, {cust_name}. Reply CONFIRM to finalise or REPLY CHANGE to pick another slot."
                    state.last_selection = selected
                    state.reply_stage = "confirm_pending"
                    state.last_bot_body = confirm_body
                    return {"action": "send", "body": confirm_body, "cta": "binary_confirm_cancel", "send_as": "merchant_on_behalf", "rationale": "Customer selected an offered slot by numeric choice."}

        if is_customer:
            t = msg.lower().strip()
            if "confirm" in t and state.reply_stage in {"confirm_pending", "selected"}:
                state.ended = True
                _clear_offer_state(state)
                state.reply_stage = "ended"
                return {"action": "end", "rationale": "Customer confirmed booking; closing conversation."}

            if any(w in t for w in ["change", "reschedule", "another"]) and state.last_offered_slots:
                state.reply_stage = "change_requested"
                if len(state.last_offered_slots) >= 2:
                    body_text = f"Sure, let's change it. Pick one: 1) {state.last_offered_slots[0]} 2) {state.last_offered_slots[1]}."
                    cta = "multi_choice_slot"
                else:
                    body_text = "Sure, let's change it. Share your preferred day and time."
                    cta = "open_ended"
                state.last_bot_body = body_text
                return {
                    "action": "send",
                    "body": body_text,
                    "cta": cta,
                    "send_as": "merchant_on_behalf",
                    "rationale": "State-machine change request detected; asked for alternate slot.",
                }

            if any(w in t for w in ["cancel", "not now", "no thanks"]) and state.reply_stage in {"offered", "confirm_pending", "change_requested"}:
                state.ended = True
                _clear_offer_state(state)
                state.reply_stage = "ended"
                return {"action": "end", "rationale": "Customer cancelled booking flow; conversation closed."}

        # Customer natural-language booking intent (book/schedule/appointment with optional day/time)
        if is_customer:
            t = msg.lower()
            booking_words = ["book", "booking", "schedule", "appointment", "reserve", "slot"]
            has_booking_intent = any(w in t for w in booking_words)
            if has_booking_intent:
                cust_name = "there"
                if state.customer_id:
                    stored_c = CONTEXTS.get(("customer", state.customer_id))
                    c = stored_c.payload if stored_c else {}
                    cust_name = c.get("identity", {}).get("name") or c.get("name") or cust_name

                selected = _extract_slot_from_text(msg, state.last_offered_slots or [])
                if not selected and state.last_offered_slots:
                    if len(state.last_offered_slots) >= 2:
                        body_text = f"Got it, {cust_name}. Please pick one slot: 1) {state.last_offered_slots[0]} 2) {state.last_offered_slots[1]}."
                    else:
                        body_text = f"Got it, {cust_name}. Please share your preferred day and time to book."
                    body_text = _avoid_repeat(body_text, state.last_bot_body)
                    state.last_bot_body = body_text
                    return {
                        "action": "send",
                        "body": body_text,
                        "cta": "multi_choice_slot" if len(state.last_offered_slots) >= 2 else "open_ended",
                        "send_as": "merchant_on_behalf",
                        "rationale": "Detected customer booking intent and asked for explicit slot selection.",
                    }

                if selected:
                    confirm_body = f"Booked {selected} for you, {cust_name}. Reply CONFIRM to finalise."
                    state.reply_stage = "confirm_pending"
                else:
                    confirm_body = f"Great {cust_name}, I can help with booking. Share day and time (for example, Wed 6pm) and I will lock it in."
                    state.reply_stage = "change_requested"
                state.last_selection = selected
                state.last_bot_body = confirm_body
                return {
                    "action": "send",
                    "body": confirm_body,
                    "cta": "binary_confirm_cancel" if selected else "open_ended",
                    "send_as": "merchant_on_behalf",
                    "rationale": "Detected customer booking intent and moved to booking confirmation flow.",
                }

        if _looks_like_commitment(msg):
            # Switch to action mode.
            if is_customer:
                cust_name = "there"
                if state.customer_id:
                    stored_c = CONTEXTS.get(("customer", state.customer_id))
                    c = stored_c.payload if stored_c else {}
                    cust_name = c.get("identity", {}).get("name") or c.get("name") or cust_name
                body_text = f"Great {cust_name} — I’ll confirm the booking and keep you posted. Reply CONFIRM to proceed."
            else:
                body_text = "Done — I’m drafting a ready-to-send WhatsApp + a Google Post now. Reply CONFIRM to proceed." 
                if state.merchant_id:
                    stored_m = CONTEXTS.get(("merchant", state.merchant_id))
                    m = stored_m.payload if stored_m else {}
                    name = _merchant_display_name(m) if m else ""
                    langs = _merchant_languages(m) if m else []
                    if _pick_hi_en(langs):
                        body_text = f"Great {name} — main abhi WhatsApp draft + Google Post ready kar rahi hoon. CONFIRM likh do, main format bhej deti hoon." 
                    else:
                        body_text = f"Great {name} — drafting the WhatsApp + Google Post now. Reply CONFIRM to proceed." 

            body_text = _avoid_repeat(body_text, state.last_bot_body)
            state.last_bot_body = body_text
            return {"action": "send", "body": body_text, "cta": "binary_confirm_cancel", "rationale": "Commitment detected; switching from qualifying to action."}

        # Default follow-up
        if is_customer:
            cust_name = "there"
            biz_name = "our team"
            if state.customer_id:
                stored_c = CONTEXTS.get(("customer", state.customer_id))
                c = stored_c.payload if stored_c else {}
                cust_name = c.get("identity", {}).get("name") or c.get("name") or cust_name
            if state.merchant_id:
                stored_m = CONTEXTS.get(("merchant", state.merchant_id))
                m = stored_m.payload if stored_m else {}
                biz_name = m.get("identity", {}).get("name") or biz_name
            follow = f"Perfect {cust_name}, I'll sync {biz_name} now. Confirm?"
            follow = _avoid_repeat(follow, state.last_bot_body)
            state.last_bot_body = follow
            return {
                "action": "send",
                "body": follow,
                "cta": "binary_yes_no",
                "send_as": "merchant_on_behalf",
                "rationale": "Customer reply acknowledged with clear next step.",
            }

        follow = "Got it. Ready for your draft?"
        follow = _avoid_repeat(follow, state.last_bot_body)
        state.last_bot_body = follow
        return {"action": "send", "body": follow, "cta": "binary_yes_no", "rationale": "Acknowledged reply and asked for confirmation to proceed."}


@app.post("/v1/teardown")
def teardown() -> dict[str, Any]:
    """Optional endpoint mentioned in testing brief: wipes in-memory state."""
    with LOCK:
        CONTEXTS.clear()
        CONVERSATIONS.clear()
        SENT_SUPPRESSIONS.clear()
    return {"ok": True, "cleared_at": _now_iso()}
