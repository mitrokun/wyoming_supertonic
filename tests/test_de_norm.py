"""Unit tests for the German text normalizer.

Run from the repository root:

    python3 -m unittest discover -s tests

Only num2words is required; no onnxruntime or model download.
"""

import unittest
from unittest import mock

from wyoming_supertonic.de_norm import GermanTextNormalizer


class GermanTextNormalizerTest(unittest.TestCase):

    def setUp(self):
        self.norm = GermanTextNormalizer()

    def assertNormalized(self, text, expected):
        self.assertEqual(self.norm.normalize(text), expected)

    # --- IPv4, CIDR, dotted version numbers ---------------------------------

    def test_ipv4_is_read_group_by_group_and_keeps_sentence_period(self):
        self.assertNormalized(
            "Der Container hat die IP 10.85.0.65.",
            "Der Container hat die IP zehn Punkt fünfundachtzig Punkt null Punkt fünfundsechzig.",
        )

    def test_ipv4_with_port_reads_colon_and_spells_the_digits(self):
        self.assertNormalized(
            "halo ist 10.85.0.15:13305",
            "halo ist zehn Punkt fünfundachtzig Punkt null Punkt fünfzehn Doppelpunkt "
            "eins drei drei null fünf",
        )

    def test_ipv4_with_port_keeps_sentence_period(self):
        self.assertNormalized(
            "Erreichbar unter 10.85.0.15:22.",
            "Erreichbar unter zehn Punkt fünfundachtzig Punkt null Punkt fünfzehn "
            "Doppelpunkt zwei zwei.",
        )

    def test_colon_after_version_number_is_not_spelled(self):
        self.assertNormalized("1.3.1:22", "eins Punkt drei Punkt eins:zweiundzwanzig")

    def test_ipv4_colon_followed_by_space_stays_punctuation(self):
        self.assertNormalized(
            "Server 10.85.0.65: erreichbar",
            "Server zehn Punkt fünfundachtzig Punkt null Punkt fünfundsechzig: erreichbar",
        )

    def test_cidr_prefixes_for_ipv4_and_ipv6(self):
        self.assertNormalized(
            "Netz 10.85.0.0/23 und 2a01:41e0::/29",
            "Netz zehn Punkt fünfundachtzig Punkt null Punkt null Schrägstrich dreiundzwanzig "
            "und zwei a null eins Doppelpunkt vier eins e null Doppelpunkt Doppelpunkt "
            "Schrägstrich neunundzwanzig",
        )

    def test_invalid_ipv4_prefix_length_keeps_the_slash(self):
        self.assertNormalized(
            "10.0.0.0/99", "zehn Punkt null Punkt null Punkt null/neunundneunzig"
        )

    def test_version_number_and_clock_time_in_one_sentence(self):
        self.assertNormalized(
            "Version 1.3.1 und Uhrzeit 12:30",
            "Version eins Punkt drei Punkt eins und Uhrzeit zwölf Uhr dreißig",
        )

    def test_valid_ipv4_in_thousands_grouping_shape_is_an_address(self):
        self.assertNormalized(
            "192.168.178.100",
            "einhundertzweiundneunzig Punkt einhundertachtundsechzig Punkt "
            "einhundertachtundsiebzig Punkt einhundert",
        )

    # Deliberate exception to the "three or more groups" rule: German thousands
    # grouping (first group 1-3 digits, every further group exactly 3) is still
    # one number unless the sequence is a valid IPv4 address.

    def test_three_groups_in_thousands_grouping_stay_one_number(self):
        self.assertNormalized(
            "1.500.000 Einwohner", "eine Million fünfhunderttausend Einwohner"
        )

    def test_four_groups_in_thousands_grouping_stay_one_number(self):
        self.assertNormalized(
            "1.500.000.000 Euro", "eine Milliarde fünfhundert Millionen Euro"
        )

    def test_thousands_grouping_that_is_no_valid_ipv4_stays_one_number(self):
        self.assertNormalized(
            "100.200.300.400",
            "einhundert Milliarden zweihundert Millionen dreihunderttausendvierhundert",
        )

    # --- IPv6 ----------------------------------------------------------------

    def test_ipv6_is_spelled_per_character_with_double_colon(self):
        self.assertNormalized(
            "Adresse 2a01:2761:0:1::5",
            "Adresse zwei a null eins Doppelpunkt zwei sieben sechs eins Doppelpunkt null "
            "Doppelpunkt eins Doppelpunkt Doppelpunkt fünf",
        )

    def test_ipv6_keeps_sentence_period(self):
        self.assertNormalized("Loopback ::1.", "Loopback Doppelpunkt Doppelpunkt eins.")

    def test_ipv6_hex_letters_are_spoken_as_lowercase_letters(self):
        self.assertNormalized("2A01::F", "zwei a null eins Doppelpunkt Doppelpunkt f")

    def test_ipv6_lookalikes_are_left_alone(self):
        for text in ("00:11:22:33:44:55", "12:30:45", "a:b:c", "x2a01::1"):
            with self.subTest(text=text):
                self.assertNotIn("Doppelpunkt", self.norm.normalize(text))

    # --- regressions: existing behaviour must not change ----------------------

    def test_clock_times(self):
        self.assertNormalized("12:30", "zwölf Uhr dreißig")
        self.assertNormalized("Um 8:05 Uhr", "Um acht Uhr fünf")

    def test_temperature(self):
        self.assertNormalized("19,4 °C", "neunzehn Komma vier Grad")

    def test_percent(self):
        self.assertNormalized("54,3 %", "vierundfünfzig Komma drei Prozent")

    def test_thousands_separator_with_two_groups(self):
        self.assertNormalized("1.000", "eintausend")

    def test_price_with_thousands_separator_and_decimal_comma(self):
        self.assertNormalized("Preis 1.000,50 Euro", "Preis eintausend Komma fünf null Euro")

    def test_negative_numbers(self):
        self.assertNormalized("-5 Grad", "minus fünf Grad")
        self.assertNormalized("Es sind -3,5 °C", "Es sind minus drei Komma fünf Grad")

    def test_expression_tags_are_preserved(self):
        self.assertNormalized("<laugh> Hallo 3 Mal", "<laugh> Hallo drei Mal")
        self.assertNormalized(
            "IP 10.85.0.1 <laugh>", "IP zehn Punkt fünfundachtzig Punkt null Punkt eins <laugh>"
        )

    def test_empty_and_blank_text(self):
        self.assertNormalized("", "")
        self.assertNormalized("   ", "   ")

    def test_fail_open_returns_original_text(self):
        with mock.patch.object(
            GermanTextNormalizer, "_pipeline", side_effect=RuntimeError("boom")
        ), self.assertLogs("wyoming_supertonic.de_norm", level="WARNING"):
            self.assertNormalized("IP 10.85.0.65", "IP 10.85.0.65")


if __name__ == "__main__":
    unittest.main()
