import unittest
from utils import filter_text

class TestFilterText(unittest.TestCase):
    def test_empty_input(self):
        self.assertIsNone(filter_text("", "te"))
        self.assertIsNone(filter_text("   ", "en"))

    def test_essential_short_answers(self):
        self.assertEqual(filter_text("మగ", "te"), "మగ")
        self.assertEqual(filter_text("ఆడ", "te"), "ఆడ")
        self.assertEqual(filter_text("yes", "en"), "yes")
        self.assertEqual(filter_text("no", "en"), "no")
        self.assertEqual(filter_text("oc", "en"), "oc")

    def test_garbage(self):
        self.assertIsNone(filter_text("asdfasdf", "en"))
        self.assertIsNone(filter_text("బ్లా బ్లా", "te"))

    def test_short_sound(self):
        self.assertIsNone(filter_text("hm", "en"))
        self.assertIsNone(filter_text("హ్మ్", "te"))

    def test_too_short(self):
        self.assertIsNone(filter_text("a", "en"))
        self.assertIsNone(filter_text("ఎ", "te"))

    def test_valid_long_input(self):
        self.assertEqual(filter_text("This is valid.", "en"), "This is valid.")
        self.assertEqual(filter_text("ఇది సరైన వాక్యం.", "te"), "ఇది సరైన వాక్యం.")

if __name__ == "__main__":
    unittest.main()
