"""Safety rules of the assistant. They run before and after the language model.

1. Red flags (signs of an emergency) give a fixed escalation message. No model call.
2. Requests for a diagnosis, a prognosis or a medicine dose give a fixed
   non-diagnostic answer plus general, cited information.
3. A model answer that states a diagnosis is replaced by a safe fallback.
4. Every answer ends with the disclaimer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DISCLAIMER = ("This is general information, not medical advice or a diagnosis. "
              "Discuss your results and your health with your care team.")

ESCALATION = ("Some of what you describe can be a sign of an emergency. Contact emergency services or your "
              "care team now. Do not wait for an online answer.")

NON_DIAGNOSTIC = ("I cannot tell you whether you have a disease, what a lesion is, how a condition will develop "
                  "or which medicine dose to take. Only your doctor can answer this with your full history and "
                  "the radiology report.")

RED_FLAGS = [
    r"vomit\w*\s+(?:up\s+)?blood", r"coffee[- ]grounds?", r"black\s+(?:tarry\s+)?stools?", r"blood\w*\s+stools?",
    r"blood in (?:my )?stool", r"severe\s+(?:abdominal|belly|stomach|tummy)\s+pain", r"\bconfus\w*", r"\bfaint\w*",
    r"chest pain", r"(?:can'?t|cannot|hard to|difficult\w*)\s+breath\w*", r"short(?:ness)? of breath",
    r"suicid\w*", r"kill myself", r"end my life", r"passed out", r"unconscious",
]

DIAGNOSIS_REQUESTS = [
    r"\bdo i have\b", r"\bam i (?:dying|sick|ill)\b", r"\bis (?:it|this|that|my \w+) (?:cancer|malignant|benign)\b",
    r"\bwhat stage\b", r"\bprognosis\b", r"\bhow long (?:do|will|have) i\b", r"\bsurviv\w*\b", r"\bdiagnos\w*\b",
    r"\bhow (?:much|many) .{0,40}(?:take|dose)\b", r"\bdos(?:e|age)\b", r"\bshould i (?:stop|start|take)\b",
]

DIAGNOSTIC_CLAIMS = [
    r"\byou (?:have|do not have|don'?t have) (?:a |an )?(?:cancer|tumou?r|carcinoma|metasta\w+|cirrhosis)\b",
    r"\bit is (?:definitely |probably |likely )?(?:benign|malignant|cancer)\b",
    r"\byour (?:lesion|tumou?r) is (?:benign|malignant|cancer\w*)\b",
    r"\btake \d+\s?(?:mg|ml|tablets?)\b",
]

_RED = re.compile("|".join(RED_FLAGS), re.I)
_DIAG = re.compile("|".join(DIAGNOSIS_REQUESTS), re.I)
_CLAIM = re.compile("|".join(DIAGNOSTIC_CLAIMS), re.I)


@dataclass(frozen=True)
class Triage:
    red_flag: bool
    diagnosis_request: bool


def triage(question: str) -> Triage:
    return Triage(bool(_RED.search(question)), bool(_DIAG.search(question)))


def has_diagnostic_claim(answer: str) -> bool:
    return bool(_CLAIM.search(answer))


def with_disclaimer(answer: str) -> str:
    answer = answer.rstrip()
    return answer if answer.endswith(DISCLAIMER) else f"{answer}\n\n{DISCLAIMER}"
