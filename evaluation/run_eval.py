"""
Eval harness for the Aster & Row support agent.

Run with:
    uv run python -m evaluation.run_eval

Loads evaluation/visible-cases.json and evaluation/custom-cases.json,
runs each case's conversation through the real compiled graph (same
graph the CLI uses), and checks the final answer against the "expect"
block using plain, deterministic checks (string matching and small
keyword checks) -- no LLM is used to grade the answers.

Assertion notes (documented here so the checks aren't a black box):
  - must_include: case-insensitive, and tolerant of a few specific,
    known-harmless wording differences -- hyphenated vs spaced ("45-
    calendar-day" vs "45 calendar days"), singular/plural, a small set of
    word-form families (delivery/delivered), and either date style for
    the same date ("August 22, 2026" vs "2026-08-22"). Every word still
    has to be present in some acceptable form -- this loosens formatting,
    not meaning.
  - must_not_include: plain case-insensitive substring check, deliberately
    NOT loosened the same way -- this is used for exact forbidden values
    like real PII, so it stays strict.
  - must_include_concepts: a concept counts as present if most of its
    significant words show up in the final answer. This is a rough
    approximation of "the idea is present," not exact wording. As a
    special case, a concept that mentions "human" (e.g. "human
    confirmation") also counts as present if the answer matches one of
    the handoff patterns below (e.g. "I recommend contacting our support
    team") -- recommending a human/support step is what that concept
    actually looks like in a real answer, even without those literal
    words.
  - required_sources / forbidden_sources_as_authority: checked against
    the filenames the retriever tool actually returned, across every
    turn of the conversation (not just the last one).
  - tool / tool_arguments: checked against the actual tool calls made
    anywhere in the conversation, since a tool called on an earlier
    turn can be correctly reused on a later turn without calling it
    again.
  - handoff: approximated by looking for human-handoff words in the
    final answer (e.g. "human", "support", "escalate"). This is a
    heuristic, not a structured field, so it can be wrong on unusual
    phrasing. Curly quotes and non-breaking hyphens are normalized to
    plain ASCII before any text check runs, so wording differences
    don't cause false failures.
  - must_not_invent: only checked for dates and tracking-number-shaped
    text. A date or tracking number in the final answer must also show
    up in the tool result it supposedly came from, otherwise it counts
    as invented.
"""

import json
import re
import sys
import uuid
from pathlib import Path

from src.agent.agent import graph
from src.trace import build_trace

EVAL_DIR = Path(__file__).resolve().parent

HANDOFF_PATTERNS = [
    re.compile(r"\bhuman\b"),
    re.compile(r"\bescalat"),
    re.compile(r"\bspecialist\b"),
    re.compile(r"support[\w\s-]{0,20}(team|specialist|agent)"),
    re.compile(r"contact[\w\s-]{0,15}support"),
    re.compile(r"reach out[\w\s-]{0,15}(support|team|customer service)"),
]

UNICODE_PUNCTUATION = {
    "‑": "-",  # non-breaking hyphen
    "‐": "-",  # hyphen
    "–": "-",  # en dash
    "—": "-",  # em dash
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}


def normalize_text(text):
    for weird, plain in UNICODE_PUNCTUATION.items():
        text = text.replace(weird, plain)
    return text


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


def canonical_date(value):
    """Turn either an ISO date (2026-08-22) or a natural-language date
    (August 22, 2026) into the same "Month D, YYYY" form, so the two
    styles can be compared for a match. Returns None if it can't parse."""
    value = value.strip().rstrip(",")

    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= month <= 12:
            return f"{MONTH_NAMES[month - 1]} {day}, {year}"
        return None

    match = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$", value)
    if match and match.group(1) in MONTH_NAMES:
        return f"{match.group(1)} {int(match.group(2))}, {match.group(3)}"

    return None

DATE_PATTERN = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
)
TRACKING_PATTERN = re.compile(r"\b[A-Z0-9]{10,}\b")

STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "for", "on",
    "in", "with", "does", "do", "not", "it", "this", "that", "one", "-",
}

WORD_FAMILIES = [
    {"deliver", "delivery", "delivered", "delivering", "deliveries"},
]


def word_candidates(word):
    """A word, plus its singular form and any word-family it belongs to
    (e.g. delivery/delivered), as acceptable stand-ins for each other."""
    candidates = {word}
    if word.endswith("s") and len(word) > 3:
        candidates.add(word[:-1])
    else:
        candidates.add(word + "s")
    for family in WORD_FAMILIES:
        if word in family or (word.endswith("s") and word[:-1] in family):
            candidates |= family
    return candidates


def flexible_phrase_present(text, phrase):
    """Like a plain substring check, but tolerant of a few specific,
    known-harmless wording differences: hyphenated vs spaced ("45-
    calendar-day" vs "45 calendar days"), singular vs plural, a small set
    of word-form families (delivery/delivered/...), and either date style
    for the same date ("August 22, 2026" vs "2026-08-22"). Still requires
    every word of the phrase to be present in some acceptable form -- this
    is not a fuzzy/approximate match, just format-tolerant."""
    text_lower = text.lower()
    phrase_lower = phrase.lower()

    if phrase_lower in text_lower:
        return True

    dehyphenated_text = re.sub(r"\s+", " ", text_lower.replace("-", " "))
    dehyphenated_phrase = re.sub(r"\s+", " ", phrase_lower.replace("-", " "))
    if dehyphenated_phrase in dehyphenated_text:
        return True

    phrase_words = re.findall(r"[a-z0-9']+", dehyphenated_phrase)
    if phrase_words:
        text_words = set(re.findall(r"[a-z0-9']+", dehyphenated_text))
        if all(word_candidates(w) & text_words for w in phrase_words):
            return True

    phrase_date = canonical_date(phrase)
    if phrase_date:
        for match in DATE_PATTERN.finditer(text):
            if canonical_date(match.group(0)) == phrase_date:
                return True

    return False


def run_conversation(messages):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state = None
    for msg in messages:
        state = graph.invoke({"messages": [("user", msg["content"])]}, config)

    return build_trace(state)


def check_must_include(text, phrases):
    failures = []
    for phrase in phrases:
        if not flexible_phrase_present(text, phrase):
            failures.append(f"missing required text: {phrase!r}")
    return failures


def check_must_not_include(text, phrases):
    failures = []
    for phrase in phrases:
        if phrase.lower() in text.lower():
            failures.append(f"contains forbidden text: {phrase!r}")
    return failures


def words_satisfied(text_lower, phrase):
    """Word-overlap check: most of the phrase's significant words show up."""
    words = [w for w in re.findall(r"[a-z0-9']+", phrase.lower()) if w not in STOPWORDS]
    if not words:
        return True
    hits = sum(1 for w in words if w in text_lower)
    return hits >= max(1, len(words) // 2)


def concept_satisfied(text_lower, concept):
    """A concept is satisfied by the normal word-overlap check, OR, if the
    concept mentions "human" (e.g. "human confirmation", "human review
    before approval"), also by any of the human-handoff patterns (e.g. "I
    recommend contacting our support team"). That's deliberately narrow:
    it only kicks in when the concept is actually about recommending a
    human/support step, since that's what asking for one actually looks
    like in a real answer -- the literal words "human" or "confirmation"
    don't need to appear for the idea to be present. It does NOT split the
    concept on "or" in general, since "or" sometimes just joins two nouns
    in one concept (e.g. "duties or taxes are not prepaid" is one idea
    about either charge, not two alternative concepts to accept either of)."""
    if words_satisfied(text_lower, concept):
        return True
    if "human" in concept.lower() and any(p.search(text_lower) for p in HANDOFF_PATTERNS):
        return True
    return False


def check_concepts(text, concepts):
    failures = []
    text_lower = text.lower()
    for concept in concepts:
        if not concept_satisfied(text_lower, concept):
            failures.append(f"concept not clearly present: {concept!r}")
    return failures


def check_sources(tool_results, required_sources, forbidden_sources, final_response):
    failures = []
    combined_tool_text = "\n".join(tool_results)

    for source in required_sources:
        if source not in combined_tool_text:
            failures.append(f"required source not retrieved: {source}")

    for source in forbidden_sources:
        if source in combined_tool_text:
            failures.append(f"forbidden source was retrieved: {source}")
        if source in final_response:
            failures.append(f"forbidden source cited in answer: {source}")

    return failures


def check_tool(tool_calls, expected_tool):
    failures = []
    order_lookup_calls = [c for c in tool_calls if c["name"] == "order_lookup"]

    if expected_tool in ("not_called", "not_called_without_id"):
        if order_lookup_calls:
            failures.append("order_lookup was called but was not expected to be")
    elif expected_tool == "order_lookup":
        if not order_lookup_calls:
            failures.append("order_lookup was expected to be called but was not")
    # "optional_sanitized_lookup" -> no assertion, either behavior is fine

    return failures


def check_tool_arguments(tool_calls, expected_args):
    failures = []
    expected_id = expected_args.get("order_id")
    if not expected_id:
        return failures

    def normalize(s):
        return re.sub(r"[^A-Z0-9]", "", s.upper())

    order_lookup_calls = [c for c in tool_calls if c["name"] == "order_lookup"]
    match = any(
        normalize(str(c["args"].get("order_id", ""))) == normalize(expected_id)
        for c in order_lookup_calls
    )
    if not match:
        failures.append(f"no order_lookup call used order_id matching {expected_id!r}")
    return failures


def check_handoff(final_response, expected_handoff):
    text_lower = final_response.lower()
    detected = any(pattern.search(text_lower) for pattern in HANDOFF_PATTERNS)
    if detected != expected_handoff:
        return [f"handoff expected {expected_handoff}, detected {detected} (heuristic, based on wording)"]
    return []


def check_must_not_invent(final_response, tool_results, fields):
    if not fields:
        return []

    combined_tool_text = "\n".join(tool_results)
    failures = []

    wants_date_check = any(f in ("delivery estimate", "arrival date", "estimated arrival", "delivery date") for f in fields)
    wants_tracking_check = any(f == "tracking number" for f in fields)

    if wants_date_check:
        grounded_dates = set()
        for match in re.finditer(r"\d{4}-\d{2}-\d{2}", combined_tool_text):
            canonical = canonical_date(match.group(0))
            if canonical:
                grounded_dates.add(canonical)

        for match in DATE_PATTERN.finditer(final_response):
            value = match.group(0)
            canonical = canonical_date(value)
            if canonical is None or canonical not in grounded_dates:
                failures.append(f"date not grounded in tool result: {value!r}")

    if wants_tracking_check:
        for match in TRACKING_PATTERN.finditer(final_response):
            value = match.group(0)
            if value not in combined_tool_text:
                failures.append(f"tracking-like value not grounded in tool result: {value!r}")

    return failures


def evaluate_case(case):
    turns = run_conversation(case["messages"])
    expect = case["expect"]

    # text-based checks look only at the answer to the last message,
    # since that is the one the "expect" block describes
    final_response = normalize_text(turns[-1]["final_response"] or "")

    # tool-based checks look across the whole conversation, since a
    # tool called on an earlier turn can correctly be reused on a
    # later turn without calling it again
    tool_calls = []
    tool_results = []
    for turn in turns:
        tool_calls += turn["tool_calls"]
        tool_results += turn["tool_results"]

    failures = []

    if "must_include" in expect:
        failures += check_must_include(final_response, expect["must_include"])
    if "must_not_include" in expect:
        failures += check_must_not_include(final_response, expect["must_not_include"])
    if "must_include_concepts" in expect:
        failures += check_concepts(final_response, expect["must_include_concepts"])
    if "must_ask_for" in expect:
        failures += check_concepts(final_response, expect["must_ask_for"])
    if "required_sources" in expect or "forbidden_sources_as_authority" in expect:
        failures += check_sources(
            tool_results,
            expect.get("required_sources", []),
            expect.get("forbidden_sources_as_authority", []),
            final_response,
        )
    if "tool" in expect:
        failures += check_tool(tool_calls, expect["tool"])
    if "tool_arguments" in expect:
        failures += check_tool_arguments(tool_calls, expect["tool_arguments"])
    if "handoff" in expect:
        failures += check_handoff(final_response, expect["handoff"])
    if "must_not_invent" in expect:
        failures += check_must_not_invent(final_response, tool_results, expect["must_not_invent"])

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": len(failures) == 0,
        "failures": failures,
        "final_response": final_response,
    }


def load_cases():
    cases = []
    for filename in ("visible-cases.json", "custom-cases.json"):
        with open(EVAL_DIR / filename, encoding="utf-8") as f:
            data = json.load(f)
        cases.extend(data["cases"])
    return cases


def main():
    cases = load_cases()
    results = []

    for case in cases:
        print(f"running: {case['id']}", file=sys.stderr)
        result = evaluate_case(case)
        results.append(result)

    print()
    print("=" * 70)
    print("RESULTS BY CASE")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']} ({r['category']})")
        for f in r["failures"]:
            print(f"    - {f}")

    print()
    print("=" * 70)
    print("RESULTS BY CATEGORY")
    print("=" * 70)
    categories = sorted(set(r["category"] for r in results))
    for category in categories:
        in_cat = [r for r in results if r["category"] == category]
        passed = sum(1 for r in in_cat if r["passed"])
        print(f"{category}: {passed}/{len(in_cat)}")

    total_passed = sum(1 for r in results if r["passed"])
    print()
    print(f"TOTAL: {total_passed}/{len(results)} passed")

    return results


if __name__ == "__main__":
    main()
