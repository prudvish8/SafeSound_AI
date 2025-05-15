import pytest
from utils import (
    filter_text, telugu_to_number, extract_gender, extract_yes_no,
    extract_occupation, extract_student_choice
)

@pytest.mark.parametrize("raw,lang,out", [
    ("blah blah", "en", None),
    ("హ్మ్",      "te", None),
    ("yes",       "en", "yes"),
])
def test_filter_text(raw, lang, out):
    assert filter_text(raw, lang) == out

@pytest.mark.parametrize("txt,val", [
    ("ఒకటి", 1),
    ("ముప్పై ఆరు", 36),
    ("వంద", 100),
])
def test_telugu_to_number(txt, val):
    assert telugu_to_number(txt) == val

def test_extracts():
    assert extract_gender("Female teacher", "en") == "female"
    assert extract_yes_no("అవును", "te") is True
    assert extract_occupation("రైతు", "te") == "farmer"
    assert extract_student_choice("job", "en") == 2

def test_telugu_to_number_edge_cases():
    assert telugu_to_number("సున్నా") == 0
    assert telugu_to_number("పది")   == 10
    assert telugu_to_number("ఇరవై") == 20
    assert telugu_to_number("ముప్పై")== 30
