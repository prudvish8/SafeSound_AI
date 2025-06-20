# tests/test_app.py
import os
import sqlite3
import pytest
from app import app, setup_database
from unittest.mock import patch

USER_STATE_DB = "user_states.db"

@pytest.fixture(autouse=True)
def reset_databases(tmp_path, monkeypatch):
    """
    Before each test:
      1. Point DB_FILE at a fresh file under tmp_path
      2. Initialize it via setup_database()
      3. Wipe out the user_states table so every conversation starts from scratch
    """
    # 1) create a fresh test DB path
    test_db = tmp_path / "welfare_schemes.db"
    # 2) monkey-patch the environment so setup_database writes here
    monkeypatch.setenv("DB_FILE", str(test_db))
    # 3) blow away any old data & re-populate it
    setup_database()

    # 4) clear user_states so no leftover state
    if os.path.exists(USER_STATE_DB):
        conn = sqlite3.connect(USER_STATE_DB)
        conn.execute("DELETE FROM user_states;")
        conn.commit()
        conn.close()

@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()

def test_start_state(client):
    resp = client.post("/whatsapp", data={"From": "whatsapp:+1000", "Body": "hi"})
    text = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "Welcome to the AP Welfare Schemes Assistant" in text
    assert "for yourself or someone else" in text.lower()

def test_target_self_then_age(client):
    client.post("/whatsapp", data={"From": "whatsapp:+2000", "Body": "hi"})
    resp = client.post("/whatsapp", data={"From": "whatsapp:+2000", "Body": "myself"})
    assert "Please enter the age" in resp.get_data(as_text=True)

def test_age_retry_and_success(client):
    client.post("/whatsapp", data={"From": "whatsapp:+3000", "Body": "hi"})
    client.post("/whatsapp", data={"From": "whatsapp:+3000", "Body": "me"})
    # invalid age → retry prompt
    resp = client.post("/whatsapp", data={"From": "whatsapp:+3000", "Body": "abc"})
    text = resp.get_data(as_text=True).lower()
    assert "please enter the age of the person" in text
    # then a valid age → moves on to gender
    resp = client.post("/whatsapp", data={"From": "whatsapp:+3000", "Body": "25"})
    assert "gender" in resp.get_data(as_text=True).lower()

def test_full_happy_path_to_scheme_list(client):
    user = "whatsapp:+4000"
    steps = [
        ("hi",     "for yourself or someone else"),    # START
        ("myself", "enter the age"),                  # TARGET
        ("30",    "gender"),                         # AGE
        ("male",  "occupation"),                     # GENDER
        ("farmer","monthly income"),                 # OCCUPATION
        ("5000",  "caste"),                          # INCOME
        ("bc",    "own a house"),                    # CASTE
        ("yes",   "how many school-going children"), # OWNERSHIP → triggers kids count
        # Check for confirmation prompt wording
        ("2",     "is this correct"),                # KIDS_COUNT → summary
        ("yes",   "type a number or scheme name"),   # CONFIRM → scheme list
    ]
    for body, expected in steps:
        resp = client.post("/whatsapp", data={"From": user, "Body": body})
        assert expected.lower() in resp.get_data(as_text=True).lower()

def test_scheme_selection_documents(client):
    user = "whatsapp:+5550001111"
    # walk the user through to AWAITING_SCHEME_INTEREST
    for body in ["hi","self","30","male","farmer","5000","bc","yes","2","yes"]:
        client.post("/whatsapp", data={"From": user, "Body": body})

    # At this point the last response was your scheme list.
    # Now patch get_required_documents and choose option “2”
    fake_docs = ["• Aadhaar Card (Mandatory)", "• Income Proof (Mandatory)"]
    with patch("app.get_required_documents", return_value=fake_docs):
        resp = client.post("/whatsapp", data={"From": user, "Body": "2"})
        text = resp.get_data(as_text=True)
        assert "The documents required for *Annadata Sukhibhava*" in text
        assert "Aadhaar Card (Mandatory)" in text
        assert "Income Proof (Mandatory)" in text
