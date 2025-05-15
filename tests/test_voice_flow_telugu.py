import re
import pytest
from unittest.mock import patch

from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()

def make_voice_request(client, user, media_url, transcript):
    with patch("app.process_voice_message", return_value=transcript):
        return client.post(
            "/whatsapp",
            data={
                "From": user,
                "NumMedia": "1",
                "MediaUrl0": media_url,
                "MediaContentType0": "audio/ogg"
            }
        )

def extract_message_body(twiml_xml: str) -> str:
    """Grab the first <Message>…</Message> payload."""
    m = re.search(r"<Message>(.*?)</Message>", twiml_xml, re.DOTALL)
    return m.group(1) if m else twiml_xml

def test_telugu_voice_activation(client):
    user = "whatsapp:+919876543210"
    resp = make_voice_request(client, user, "https://example.com/fake.ogg", "హాయ్")
    body = extract_message_body(resp.get_data(as_text=True))

    assert "నమస్కారం! AP సంక్షేమ పథకాల సహాయకుడికి స్వాగతం." in body
    assert "ఈ సమాచారం మీకోసమా లేక ఇతరుల కోసమా?" in body

def test_telugu_voice_full_path_to_scheme_list(client):
    user = "whatsapp:+919876543211"
    # drive through the flow via voice
    steps = [
        ("https://ex/1.ogg", "బాబు"),         # activate
        ("https://ex/2.ogg", "నాకు"),         # self
        ("https://ex/3.ogg", "25"),          # age
        ("https://ex/4.ogg", "పురుషుడు"),    # gender
        ("https://ex/5.ogg", "రైతు"),        # occupation
        ("https://ex/6.ogg", "10000"),       # income
        ("https://ex/7.ogg", "బీసీ"),        # caste
        ("https://ex/8.ogg", "అవును"),        # ownership
        ("https://ex/9.ogg", "2"),           # kids count
        # Now confirmation:
        ("https://ex/10.ogg", "అవును"),      # yes to confirmation
    ]

    # run each except last just for status
    for url, txt in steps[:-1]:
        resp = make_voice_request(client, user, url, txt)
        assert resp.status_code == 200

    # final step → should return numbered menu
    resp = make_voice_request(client, user, steps[-1][0], steps[-1][1])
    body = extract_message_body(resp.get_data(as_text=True))

    assert "1)" in body
    assert "2)" in body
