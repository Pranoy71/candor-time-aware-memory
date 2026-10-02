"""Safety layer: secrets never leave the system; planted instructions are content, never commands."""
import re

# Secret patterns. Applied at load time (so a secret can never be indexed, retrieved or quoted)
# and again on every outgoing answer.
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),                       # OpenAI/Anthropic-style keys, sk-brightline-...
    re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),                         # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),                # GitHub tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),               # Slack tokens
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),                   # Google API keys
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{20,}"),
    re.compile(r"(?i)\b(?:password|passwd|pwd|secret|api[_ -]?key|token)\s*(?:is|=|:)\s*\S{6,}"),
]
REDACTED = "[REDACTED-SECRET]"


def redact(text):
    if not text:
        return text
    for p in SECRET_PATTERNS:
        text = p.sub(REDACTED, text)
    return text


def has_secret(text):
    return any(p.search(text or "") for p in SECRET_PATTERNS)


# Instructions planted for AI assistants. We detect and neutralise them; the rest of the
# record stays usable as ordinary content.
INJECTION_PATTERNS = [
    re.compile(r"(?is)<!--.*?-->"),                                            # hidden HTML comments
    re.compile(r"(?is)(?:note|message|instruction)s?\s+(?:to|for)\s+(?:any\s+)?(?:ai|llm|assistant|model|agent)[^\n]*?(?:\n|$)"),
    re.compile(r"(?is)ignore\s+(?:all\s+|your\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions?.*?(?:\.\s|\n|$)"),
    re.compile(r"(?is)(?:disregard|forget)\s+(?:all\s+|your\s+)?(?:previous|prior|above)\s+(?:instructions?|context).*?(?:\.\s|\n|$)"),
    re.compile(r"(?is)(?:tell|inform|instruct)\s+the\s+user\s+that.*?(?:\.\s|\n|$)"),
    re.compile(r"(?is)(?:forward|send)\s+all\s+(?:of\s+)?(?:the\s+|your\s+|my\s+)?(?:emails?|messages?|data).*?(?:\.\s|\n|$)"),
    re.compile(r"(?is)you\s+are\s+now\s+(?:a|an|in)\b.*?(?:\.\s|\n|$)"),
    re.compile(r"(?is)system\s*prompt\s*:.*?(?:\n|$)"),
]
INJECTION_MARK = "[planted instruction removed]"


def has_injection(text):
    return any(p.search(text or "") for p in INJECTION_PATTERNS)


def neutralise(text):
    """Remove planted instructions from text. Returns (clean_text, was_injected)."""
    if not text:
        return text, False
    out, hit = text, False
    for p in INJECTION_PATTERNS:
        new = p.sub(INJECTION_MARK, out)
        if new != out:
            hit, out = True, new
    return out, hit


def sanitize_output(text, forbidden_phrases=()):
    """Final gate on anything shown to the user."""
    text = redact(text)
    text, _ = neutralise(text)
    for ph in forbidden_phrases:
        if ph:
            text = re.sub(re.escape(ph), "[removed]", text, flags=re.I)
    return text
