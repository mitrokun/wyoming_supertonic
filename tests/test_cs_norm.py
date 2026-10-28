"""Unit tests for the Czech text normalizer.

Run from the repository root:

    python3 -m unittest discover -s tests

Only num2words is required.
"""

import unittest
from unittest import mock

from wyoming_supertonic.cs_norm import CzechTextNormalizer


class CzechTextNormalizerTest(unittest.TestCase):

    def setUp(self):
        self.norm = CzechTextNormalizer()

    def assertNormalized(self, text, expected):
        self.assertEqual(self.norm.normalize(text), expected)

    def test_ipv4_is_read_group_by_group_and_keeps_sentence_period(self):
        self.assertNormalized(
            "Kontejner má IP 10.85.0.65.",
            "Kontejner má IP deset tečka osmdesát pět tečka nula tečka šedesát pět.",
        )

    def test_ipv4_with_port_reads_colon_and_spells_the_digits(self):
        self.assertNormalized(
            "halo je 10.85.0.15:13305",
            "halo je deset tečka osmdesát pět tečka nula tečka patnáct dvojtečka "
            "jedna tři tři nula pět",
        )

    def test_ipv4_with_port_keeps_sentence_period(self):
        self.assertNormalized(
            "Dostupný na 10.85.0.15:22.",
            "Dostupný na deset tečka osmdesát pět tečka nula tečka patnáct "
            "dvojtečka dva dva.",
        )

    def test_colon_after_version_number_is_not_spelled(self):
        self.assertNormalized("1.3.1:22", "jedna tečka tři tečka jedna:dvacet dva")

    def test_ipv4_colon_followed_by_space_stays_punctuation(self):
        self.assertNormalized(
            "Server 10.85.0.65: dostupný",
            "Server deset tečka osmdesát pět tečka nula tečka šedesát pět: dostupný",
        )

    def test_cidr_prefixes_for_ipv4_and_ipv6(self):
        self.assertNormalized(
            "Síť 10.85.0.0/23 a 2a01:41e0::/29",
            "Síť deset tečka osmdesát pět tečka nula tečka nula lomítko dvacet tři "
            "a dva a nula jedna dvojtečka čtyři jedna e nula dvojtečka dvojtečka "
            "lomítko dvacet devět",
        )

    def test_invalid_ipv4_prefix_length_keeps_the_slash(self):
        self.assertNormalized(
            "10.0.0.0/99", "deset tečka nula tečka nula tečka nula/devadesát devět"
        )

    def test_version_number_and_clock_time_in_one_sentence(self):
        self.assertNormalized(
            "Verze 1.3.1 a čas 12:30",
            "Verze jedna tečka tři tečka jedna a čas dvanáct hodin třicet",
        )

    def test_valid_ipv4_in_thousands_grouping_shape_is_an_address(self):
        self.assertNormalized(
            "192.168.178.100",
            "sto devadesát dva tečka sto šedesát osm tečka "
            "sto sedmdesát osm tečka sto",
        )

    def test_three_groups_in_thousands_grouping_stay_one_number(self):
        self.assertNormalized(
            "1.500.000 obyvatel", "milion pětset tisíc obyvatel"
        )

    def test_ipv6_is_spelled_per_character_with_double_colon(self):
        self.assertNormalized(
            "Adresa 2a01:2761:0:1::5",
            "Adresa dva a nula jedna dvojtečka dva sedm šest jedna dvojtečka nula "
            "dvojtečka jedna dvojtečka dvojtečka pět",
        )

    def test_ipv6_keeps_sentence_period(self):
        self.assertNormalized("Loopback ::1.", "Loopback dvojtečka dvojtečka jedna.")

    def test_ipv6_hex_letters_are_spoken_as_lowercase_letters(self):
        self.assertNormalized("2A01::F", "dva a nula jedna dvojtečka dvojtečka f")

    def test_ipv6_lookalikes_are_left_alone(self):
        for text in ("00:11:22:33:44:55", "12:30:45", "a:b:c", "x2a01::1"):
            with self.subTest(text=text):
                self.assertNotIn("dvojtečka", self.norm.normalize(text))

    def test_clock_times(self):
        self.assertNormalized("12:30", "dvanáct hodin třicet")
        self.assertNormalized("V 8:05 hodin", "V osm hodin pět")

    def test_temperature(self):
        self.assertNormalized("19,4 °C", "devatenáct celá čtyři stupňů")

    def test_percent(self):
        self.assertNormalized("54,3 %", "padesát čtyři celá tři procent")

    def test_thousands_separator_with_two_groups(self):
        self.assertNormalized("1.000", "tisíc")

    def test_price_with_thousands_separator_and_decimal_comma(self):
        self.assertNormalized("Cena 1.000,50 Kč", "Cena tisíc celá pět nula Kč")

    def test_negative_numbers(self):
        self.assertNormalized("-5 stupňů", "mínus pět stupňů")
        self.assertNormalized("Je -3,5 °C", "Je mínus tři celá pět stupně")

    def test_declension(self):
        self.assertNormalized("1 °", "jeden stupeň")
        self.assertNormalized("2 °", "dva stupně")
        self.assertNormalized("5 °", "pět stupňů")
        self.assertNormalized("1:00", "jedna hodina")
        self.assertNormalized("2:00", "dvě hodiny")
        self.assertNormalized("5:00", "pět hodin")
        self.assertNormalized("1 %", "jedno procento")
        self.assertNormalized("2 %", "dvě procenta")
        self.assertNormalized("5 %", "pět procent")

    def test_expression_tags_are_preserved(self):
        self.assertNormalized("<laugh> Ahoj 3 krát", "<laugh> Ahoj tři krát")
        self.assertNormalized(
            "IP 10.85.0.1 <laugh>", "IP deset tečka osmdesát pět tečka nula tečka jedna <laugh>"
        )

    def test_empty_and_blank_text(self):
        self.assertNormalized("", "")
        self.assertNormalized("   ", "   ")

    def test_fail_open_returns_original_text(self):
        with mock.patch.object(
            CzechTextNormalizer, "_pipeline", side_effect=RuntimeError("boom")
        ), self.assertLogs("wyoming_supertonic.cs_norm", level="WARNING"):
            self.assertNormalized("IP 10.85.0.65", "IP 10.85.0.65")


if __name__ == "__main__":
    unittest.main()