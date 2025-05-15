import pytest
from app import filter_text, match_intent, get_prompt

def test_filter_permits_telugu_digits_and_keywords():
    assert filter_text("12", "te") == "12"
    assert filter_text("బీసీ", "te") == "బీసీ"
    assert filter_text("హాహా", "te") is None      # short laugh filtered

def test_match_intent_telugu_skip_and_start():
    # From prompts.yaml → INTENT_KEYWORDS.START.te.start includes babu, ప్రారంభం
    assert match_intent("START", "te", "బాబు") == "start"
    # skip
    assert match_intent("START", "te", "వద్దు") == "skip"

def test_get_prompt_telugu_awating_target():
    q = get_prompt("AWAITING_TARGET", "te", key="q")
    assert "మీకోసమా లేక ఇతరుల కోసమా" in q
