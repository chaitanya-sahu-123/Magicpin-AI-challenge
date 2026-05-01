import time
import re
import threading
import hashlib
import json
import os
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


CONVERSATIONS: dict[str, ConversationState] = {}

# suppression_key -> last_sent_ts
SENT_SUPPRESSIONS: dict[str, float] = {}

# merchant_id -> consecutive auto-reply count (across conversations)
AUTO_REPLY_STREAK_BY_MERCHANT: dict[str, int] = {}

# merchant_id -> last body fingerprint to prevent cross-conversation repeats
LAST_BODY_HASH_BY_MERCHANT: dict[str, str] = {}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "moonshotai/kimi-k2")
USE_LLM = os.getenv("LLM_USE", "false").lower() in {"1", "true", "yes"}
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


def _body_fingerprint(body: str) -> str:
    normalized = re.sub(r"\s+", " ", body.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
    return ""


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
    return "neutral"


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
    biz_name = merchant.get("identity", {}).get("name", merchant_name)
    locality = merchant.get("identity", {}).get("locality")
    voice_prefix = _category_voice_prefix(category)
    style_hint = _category_style_hint(category)

    perf = merchant.get("performance", {})
    views = perf.get("views")
    calls = perf.get("calls")
    ctr = perf.get("ctr")

    peer = category.get("peer_stats", {})
    peer_ctr = peer.get("avg_ctr")

    active_offers = [o.get("title") for o in (merchant.get("offers", []) or []) if o.get("status") == "active" and o.get("title")]
    offer_hint = active_offers[0] if active_offers else None

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
            if pref:
                readable_pref = pref.replace("_", " ")
                body += f"Preferred slot: {readable_pref}. Reply 1 to take it, 2 to suggest another time." 
                cta = "multi_choice_slot"
            else:
                body += "Reply YES to book a slot."
                cta = "binary_yes_no"
            if hi_en_customer:
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
            }

        if kind in {"appointment_tomorrow", "chronic_refill_due"}:
            pref = customer.get("preferences", {}).get("preferred_slots")
            body += "Quick reminder from your recent booking."
            if pref:
                body += f" Preferred time: {pref.replace('_', ' ')}. Reply 1 to confirm, 2 to change." 
                cta = "multi_choice_slot"
            else:
                body += " Reply YES if you'd like to confirm."
                cta = "binary_yes_no"
            if offer_hint:
                body += f" {offer_hint}."
            if hi_en_customer:
                body = f"Hi {cname}, {biz_name} here. Ek quick reminder — confirm karna ho to reply 1, change karna ho to reply 2." if pref else f"Hi {cname}, {biz_name} here. Ek quick reminder — confirm karna ho to YES reply kar dijiye." 
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
            }

        if kind in {"trial_followup", "customer_lapsed_hard"}:
            pref = customer.get("preferences", {}).get("preferred_slots")
            body += "We saved your earlier interest. Want to pick a slot this week?" 
            if pref:
                body += f" Preferred time: {pref.replace('_', ' ')}. Reply 1 to confirm, 2 to change." 
                cta = "multi_choice_slot"
            else:
                body += " Reply YES to proceed."
                cta = "binary_yes_no"
            if offer_hint:
                body += f" {offer_hint}."
            if hi_en_customer:
                body = f"Hi {cname}, {biz_name} here. Aapka follow-up pending hai — slot confirm karna ho to reply 1, change karna ho to reply 2." if pref else f"Hi {cname}, {biz_name} here. Aapka follow-up pending hai — iss week slot book karna ho to YES reply kar dijiye." 
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
        name_line = f"{voice_prefix} {merchant_name}".strip()
        parts.append(f"{name_line}, {title}.")
        if trial_n:
            parts.append(f"Study size: {_format_number(trial_n)}.")
        if patient_segment:
            parts.append(f"Segment: {patient_segment.replace('_', ' ')}.")
        if source:
            parts.append(f"Source: {source}.")
        parts.append("Want me to draft a short WhatsApp post you can share with patients?")

        body = " ".join(parts)
        if hi_en:
            body = f"{merchant_name}, ek quick update: {title}. " + (f"({_format_number(trial_n)} patients) " if trial_n else "") + (f"Source: {source}. " if source else "") + "Chahen to main 4-line patient WhatsApp draft bana du?"

        cta = "binary_yes_no"
        rationale = "Uses category digest item referenced by the trigger; ends with a single low-effort CTA."
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
        delta = perf.get("delta_7d", {}).get("views_pct")
        calls_delta = perf.get("delta_7d", {}).get("calls_pct")
        delta_str = f"{int(delta * 100)}%" if isinstance(delta, (float, int)) else ""
        calls_str = f"{int(calls_delta * 100)}%" if isinstance(calls_delta, (float, int)) else ""

        name_line = f"{voice_prefix} {merchant_name}".strip()
        local = f" in {locality}" if locality else ""
        body = f"{name_line}, quick heads-up: your views are {direction} {delta_str} this week{local}. "
        if calls_str:
            body += f"Calls are {calls_str} vs last week. "
        if ctr is not None and peer_ctr is not None:
            body += f"CTR is {_format_number(ctr)} vs peer {_format_number(peer_ctr)}. "
        if offer_hint:
            body += f"Want me to push a specific offer like ‘{offer_hint}’ as a Google Post today? Reply YES."
        else:
            body += "Want me to draft a Google Post for today? Reply YES."

        if hi_en:
            body = f"{merchant_name}, quick update: is week views {('upar' if direction=='up' else 'neeche')} {delta_str}. "
            if calls_str:
                body += f"Calls {calls_str} vs last week. "
            if ctr is not None and peer_ctr is not None:
                body += f"CTR {_format_number(ctr)} vs peer {_format_number(peer_ctr)}. "
            body += "Main aaj ka ek Google Post draft kar du? YES/NO"

        cta = "binary_yes_no"
        rationale = "Anchors on merchant performance numbers and peer benchmark if present; asks a single YES/NO." 
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_perf_nudge_v1",
            "template_params": [merchant_name, delta_str or direction, offer_hint or "Google Post"],
        }

    if kind == "review_theme_emerged":
        themes = merchant.get("review_themes", []) or []
        top = themes[0] if themes else {}
        theme = top.get("theme")
        occurrences = top.get("occurrences_30d")
        quote = top.get("common_quote")
        name_line = f"{voice_prefix} {merchant_name}".strip()
        peer_reviews = category.get("peer_stats", {}).get("avg_reviews")
        body = f"{name_line}, a review pattern emerged recently."
        if theme and occurrences is not None:
            body = f"{name_line}, {occurrences} reviews this month mentioned {theme.replace('_', ' ')}."
        if quote:
            body += f" Example: “{quote}”."
        if peer_reviews is not None:
            body += f" Peer median reviews: {_format_number(peer_reviews)}."
        body += " Want me to draft a 2-line response + a quick ops fix note? Reply YES." 
        if hi_en:
            body = f"{merchant_name}, recent reviews me ek pattern aaya hai." 
            if theme and occurrences is not None:
                body += f" {occurrences} reviews me {theme.replace('_', ' ')} mention hua." 
            body += " Main 2-line reply + quick fix note draft kar du? YES/NO"
        cta = "binary_yes_no"
        rationale = "Uses recent review-theme signals when available and offers a low-effort response draft."
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
        body = f"{merchant_name}, your {plan or ''} plan renewal is due {days_text}. Want me to share a quick renewal summary + next steps? Reply YES." 
        if hi_en:
            body = f"{merchant_name}, aapka plan renewal {days_text} me due hai. Main short summary + next steps bhej du? YES/NO"
        cta = "binary_yes_no"
        rationale = "Uses subscription timing (if available) with a clear renewal CTA."
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
        body = f"{merchant_name}, congrats on a new milestone this week. Want a quick celebratory Google Post draft? Reply YES."
        if hi_en:
            body = f"{merchant_name}, iss week ek milestone hit hua — congratulations! Main ek short celebration post draft kar du? YES/NO"
        cta = "binary_yes_no"
        rationale = "Milestone trigger acknowledged with a low-effort celebration asset."
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
        body = f"{merchant_name}, a new listing just opened nearby {('in ' + locality) if locality else ''}. Want me to draft a differentiation post to keep you top-of-mind? Reply YES." 
        if hi_en:
            body = f"{merchant_name}, nearby area me ek naya listing aaya hai. Main differentiation post draft kar du taaki aap top-of-mind rahein? YES/NO"
        cta = "binary_yes_no"
        rationale = "Competitor trigger prompts a differentiation message without fabricating names or distances."
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
        when_text = f"in {when} days" if when is not None else "soon"
        local = f" in {locality}" if locality else ""
        body = f"{merchant_name}, {festival} is coming {when_text}{local}. Want a 1-line offer + poster copy for your Google profile? Reply YES."
        if offer_hint:
            body = f"{merchant_name}, {festival} {when_text}. Aapka ‘{offer_hint}’ highlight karke ek short post bana du? YES/NO"
        cta = "binary_yes_no"
        rationale = "Festival timing trigger; offers to draft a ready-to-post asset with a binary CTA."
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
        body = f"Hi {merchant_name} — quick Q: what’s the most asked-for service this week {('in ' + locality) if locality else ''}? I’ll turn it into a Google Post + WhatsApp reply draft."
        if hi_en:
            body = f"Hi {merchant_name} — quick question: iss week sabse zyada kis service ka pucha ja raha hai? Main uska Google Post + WhatsApp reply draft bana dungi."
        cta = "open_ended"
        rationale = "Curiosity-based cadence trigger; asks the merchant a single low-effort question and offers to do the work."
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
        body = f"{merchant_name}, it’s been a bit since we last spoke. Want me to share a quick 2‑min profile update to boost visibility? Reply YES." 
        if last_ts:
            body = f"{merchant_name}, last chat was on {last_ts.split('T')[0]}. Want a quick 2‑min profile update to boost visibility? Reply YES." 
        if hi_en:
            body = f"{merchant_name}, kaafi time ho gaya hai. Main 2‑min ka profile update bhej du? YES/NO"
        cta = "binary_yes_no"
        rationale = "Dormancy trigger; offers a low-effort re-engagement update without fabricating data."
        return {
            "body": body,
            "cta": cta,
            "send_as": "vera",
            "suppression_key": suppression_key,
            "rationale": rationale,
            "template_name": "vera_dormant_nudge_v1",
            "template_params": [merchant_name],
        }

    # Fallback
    topic = trigger.get("payload", {}).get("metric_or_topic")
    body = f"Hi {merchant_name} — quick check: want me to draft a small update for your Google profile today? Reply YES."
    if topic:
        body = f"Hi {merchant_name} — quick check on {topic}: want me to draft a small update for your Google profile today? Reply YES."
    if hi_en:
        body = f"Hi {merchant_name} — aaj aapke Google profile ke liye ek short update draft kar du? YES/NO"
    cta = "binary_yes_no"
    rationale = f"Fallback message for trigger kind '{kind}' when no specialized template is implemented."
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
        cur = CONTEXTS.get(key)
        if cur and body.version <= cur.version:
            response.status_code = 409
            return {"accepted": False, "reason": "stale_version", "current_version": cur.version}

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

        picked_by_merchant: dict[str, tuple[str, dict[str, Any]]] = {}
        for _, __, trg_id, trg in candidates:
            merchant_id = trg.get("merchant_id")
            if merchant_id in picked_by_merchant:
                continue
            picked_by_merchant[merchant_id] = (trg_id, trg)
            if len(picked_by_merchant) >= 20:
                break

        for merchant_id, (trg_id, trg) in picked_by_merchant.items():
            customer_id = trg.get("customer_id")
            merchant = CONTEXTS.get(("merchant", merchant_id)).payload if ("merchant", merchant_id) in CONTEXTS else None
            if not merchant:
                continue
            category_slug = merchant.get("category_slug")
            category = CONTEXTS.get(("category", category_slug)).payload if category_slug and ("category", category_slug) in CONTEXTS else None
            if not category:
                continue
            customer = CONTEXTS.get(("customer", customer_id)).payload if customer_id and ("customer", customer_id) in CONTEXTS else None
            if trg.get("scope") == "customer" and not customer:
                continue
            if trg.get("scope") == "customer" and customer:
                if not _consent_allows(customer, trg.get("kind") or ""):
                    continue

            composed = _compose_message(category=category, merchant=merchant, trigger=trg, customer=customer)
            composed = _llm_compose(category, merchant, trg, customer, composed)

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

        if _is_opt_out(msg):
            state.ended = True
            return {"action": "end", "rationale": "Opt-out detected; ending conversation."}

        if _looks_like_auto_reply(msg):
            state.auto_reply_streak += 1
            if state.merchant_id:
                AUTO_REPLY_STREAK_BY_MERCHANT[state.merchant_id] = AUTO_REPLY_STREAK_BY_MERCHANT.get(state.merchant_id, 0) + 1
            total_streak = AUTO_REPLY_STREAK_BY_MERCHANT.get(state.merchant_id or "", state.auto_reply_streak)
            if total_streak >= 3:
                state.ended = True
                return {"action": "end", "rationale": "Auto-reply repeated 3x; closing."}
            if total_streak == 2:
                return {"action": "wait", "wait_seconds": 86400, "rationale": "Auto-reply repeated; waiting 24h for owner."}
            return {"action": "wait", "wait_seconds": 14400, "rationale": "Detected auto-reply; waiting 4h."}

        # Reset streak on real reply
        state.auto_reply_streak = 0
        if state.merchant_id:
            AUTO_REPLY_STREAK_BY_MERCHANT[state.merchant_id] = 0

        if _looks_like_commitment(msg):
            # Switch to action mode.
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
        follow = "Got it. Want me to proceed with a draft (YES/NO)?"
        follow = _avoid_repeat(follow, state.last_bot_body)
        state.last_bot_body = follow
        return {"action": "send", "body": follow, "cta": "binary_yes_no", "rationale": "Acknowledged reply and offered the next step with a low-friction CTA."}


@app.post("/v1/teardown")
def teardown() -> dict[str, Any]:
    """Optional endpoint mentioned in testing brief: wipes in-memory state."""
    with LOCK:
        CONTEXTS.clear()
        CONVERSATIONS.clear()
        SENT_SUPPRESSIONS.clear()
    return {"ok": True, "cleared_at": _now_iso()}
