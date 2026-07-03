"""
Safety guardrail for crisis / self-harm language.

Why this exists (see docs/approach.md, Case 2 finding):
A 0.5B model fine-tuned on 55 examples cannot be relied upon as the sole
safety mechanism for self-harm/crisis detection, particularly on
transliterated/romanized Hindi input where the base model's generation
quality itself degrades. Case 2 in examples/sample_outputs.md demonstrated
this concretely: both base and fine-tuned models produced incoherent
output that failed to invoke the required safety-redirect behavior.

Rather than trying to train this behavior into the model at this data
scale, this module intercepts crisis-indicator language BEFORE generation
and returns a fixed, correctly-worded safety response with verified
Indian national helpline numbers. This is a standard production pattern
for safety-critical assistants: never let a small fine-tuned model be the
only safety net for self-harm detection.

This is deliberately conservative (biased toward over-triggering rather
than under-triggering) — a false positive here means a user gets pointed
to real crisis resources; a false negative could be far worse.
"""

import re

# Crisis/self-harm indicator patterns, covering English, romanized Hindi
# (Hinglish, matching this dataset's user-turn convention), and Devanagari.
# Kept as whole-pattern regexes (not a simple keyword list) to reduce
# false positives from unrelated everyday phrases.
_CRISIS_PATTERNS = [
    # English
    r"\b(kill|hurt|harm)\s+(myself|me)\b",
    r"\bsuicid(e|al)\b",
    r"\bwant(ing)?\s+to\s+die\b",
    r"\bend(ing)?\s+(my\s+)?life\b",
    r"\bno\s+(reason|point)\s+to\s+live\b",
    r"\bdon'?t\s+want\s+to\s+live\b",
    # Romanized Hindi / Hinglish (matches this dataset's user-turn style)
    r"\bjeene\s*ka\s*mann\s*nahi\b",
    r"\bjeena\s*nahi\s*(chahta|chahti)\b",
    r"\bmarna\s*chahta\b",
    r"\bmarna\s*chahti\b",
    r"\bkhud\s*ko\s*(khatam|khtm)\b",
    r"\bmai\s*marna\b",
    r"\bmain\s*marna\b",
    r"\bjaan\s*dena\b",
    r"\baatmahatya\b",
    r"\bsab\s*khatam\s*(ho\s*gaya|karna)\b",
    # Devanagari
    r"आत्महत्या",
    r"खुद\s*को\s*खत्म",
    r"जीने\s*का\s*मन\s*नहीं",
    r"मरना\s*चाहता",
    r"मरना\s*चाहती",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _CRISIS_PATTERNS]

# Fixed safety response. Bilingual, matches the persona's tone, explicitly
# declines astrological analysis per the system prompt's own safety
# instruction, and provides verified Indian national helplines.
SAFETY_RESPONSE = (
    "मुझे यह सुनकर बहुत चिंता हो रही है, और आप जो महसूस कर रहे हैं वह मायने रखता है। "
    "मैं इस समय कुंडली या ज्योतिषीय विश्लेषण नहीं दे सकता — अभी सबसे ज़रूरी है कि आप किसी "
    "प्रशिक्षित व्यक्ति से बात करें।\n\n"
    "कृपया अभी संपर्क करें:\n"
    "• Tele MANAS (भारत सरकार, 24x7, 20+ भाषाएं): 14416 या 1800-891-4416\n"
    "• KIRAN Mental Health Helpline (भारत सरकार, 24x7): 1800-599-0019\n\n"
    "I'm genuinely concerned about what you've shared, and it matters. I can't provide "
    "astrological analysis right now — right now, talking to a trained person is what "
    "matters most.\n\n"
    "Please reach out now:\n"
    "• Tele MANAS (Govt. of India, 24x7, 20+ languages): 14416 or 1800-891-4416\n"
    "• KIRAN Mental Health Helpline (Govt. of India, 24x7): 1800-599-0019\n\n"
    "यदि आप तुरंत खतरे में हैं, तो कृपया नज़दीकी अस्पताल जाएं या 112 पर कॉल करें। "
    "If you are in immediate danger, please go to your nearest hospital or call 112."
)


def is_crisis_input(text: str) -> bool:
    """Returns True if the input text matches any crisis/self-harm pattern."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _COMPILED_PATTERNS)


def get_safety_response() -> str:
    """Returns the fixed, verified safety response."""
    return SAFETY_RESPONSE