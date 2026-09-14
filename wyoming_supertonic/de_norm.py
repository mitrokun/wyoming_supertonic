"""German text normalizer for Wyoming-Supertonic.

Converts digits, decimals, clock times, percentages, degrees, IPv4/IPv6
addresses and dotted version numbers into spoken German words so a fast local TTS that
cannot handle "," "." ":" inside numbers produces correct speech.

Design notes:
- fail-open: any internal error returns the ORIGINAL text unmodified, so a
  normalizer bug can never make the assistant go mute (it just speaks the
  un-normalized text instead).
- expression tags like <laugh> are preserved (not touched / not stripped),
  per ADD_NORM.md.
- scope is deliberately limited to what the voice assistant actually emits
  (time of day, temperatures, percentages, counts). mm:ss durations are NOT
  disambiguated from HH:MM clock times - see _times().
"""

import ipaddress
import logging
import re

from num2words import num2words

log = logging.getLogger(__name__)


class GermanTextNormalizer:

    _TAGS_PATTERN = re.compile(r"(<[a-zA-Z/]+>)")

    # Characters to drop. Keeps . , : - ° % < > for processing.
    _chars_to_delete = "=#$“”„«»*\"‘’‚`"

    _emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF☀-⛿✀-➿\U0001F900-\U0001F9FF"
        "‍️"
        "]+",
        flags=re.UNICODE,
    )

    # HH:MM 24h clock (hour 0-23, minute 00-59), optional trailing "Uhr"
    _TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(\s*Uhr)?\b")
    _PERCENT_PATTERN = re.compile(r"(-?\d+(?:[.,]\d+)*)\s*%")
    _DEG_C_PATTERN = re.compile(r"(-?\d+(?:[.,]\d+)*)\s*°\s*C\b")
    _DEG_PATTERN = re.compile(r"(-?\d+(?:[.,]\d+)*)\s*°")
    _NUM_PATTERN = re.compile(r"(?:(?<=^)|(?<=\s))-\d+(?:[.,]\d+)*|\b\d+(?:[.,]\d+)*")
    # Three or more dot-separated digit groups: IPv4 addresses ("10.85.0.65"),
    # version numbers ("1.3.1"), optionally followed by a CIDR prefix ("/23")
    # and - for IPv4 - by ":" plus digits ("10.85.0.15:13305").
    # Two groups ("1.000", "10.5") are left to _NUM_PATTERN. A sentence period
    # after the last group is not part of the match.
    _DOTTED_PATTERN = re.compile(
        r"(?<![\w.,])(\d+(?:\.\d+){2,})(?![\w]|[.,]\d)(?:/(\d{1,3})(?!\w))?(?::(\d+))?"
    )
    # IPv6 candidates: a run of hex digits and ":" containing at least two
    # ":", optionally followed by a prefix length. Whether it really is an
    # address is decided by ipaddress in _repl_ipv6, so "12:30:45" or a MAC
    # address ("00:11:22:33:44:55") are left alone.
    _IPV6_PATTERN = re.compile(
        r"(?<![\w:.])(?=[0-9a-fA-F:]*:[0-9a-fA-F:]*:)([0-9a-fA-F:]+)(?![\w:]|\.\d)(?:/(\d{1,3})(?!\w))?"
    )

    def __init__(self) -> None:
        self._del_table = str.maketrans("", "", self._chars_to_delete)

    def normalize(self, text: str) -> str:
        """Public entry point. Never raises - falls back to original text."""
        try:
            return self._normalize(text)
        except Exception as e:
            log.warning("de_norm failed, passing text through unmodified: %s", e)
            return text

    # --- internals -----------------------------------------------------

    def _normalize(self, text: str) -> str:
        if not text or not text.strip():
            return text

        parts = self._TAGS_PATTERN.split(text)
        out = []
        for part in parts:
            if not part:
                continue
            if self._TAGS_PATTERN.match(part):
                out.append(part)  # preserve <tag> verbatim
            else:
                out.append(self._pipeline(part))
        return re.sub(r"\s+", " ", " ".join(out)).strip()

    def _pipeline(self, text: str) -> str:
        if not text.strip():
            return text
        text = self._emoji_pattern.sub("", text)
        text = text.translate(self._del_table)
        text = text.replace("\n", " ").replace("\t", " ")
        text = self._IPV6_PATTERN.sub(self._repl_ipv6, text)
        text = self._DOTTED_PATTERN.sub(self._repl_dotted, text)
        text = self._TIME_PATTERN.sub(self._repl_time, text)
        text = self._PERCENT_PATTERN.sub(self._repl_percent, text)
        text = self._DEG_C_PATTERN.sub(self._repl_deg, text)
        text = self._DEG_PATTERN.sub(self._repl_deg, text)
        text = self._NUM_PATTERN.sub(self._repl_num, text)
        return re.sub(r"[ ]{2,}", " ", text).strip()

    @staticmethod
    def _say_int(value) -> str:
        return num2words(int(value), lang="de")

    def _num_token(self, s: str) -> str:
        # Handle negative sign
        is_negative = s.startswith("-") or s.startswith("−")
        if is_negative:
            s = s[1:]

        # Normalize delimiters (German vs English standards)
        if "," in s:
            # German standard: "1.000,5" -> remove dots, change comma to dot -> "1000.5"
            s = s.replace(".", "").replace(",", ".")
        elif "." in s:
            # If there are multiple dots, they are thousands separators.
            if s.count(".") > 1:
                s = s.replace(".", "")
            else:
                # Single dot check: e.g., "1.000" vs "10.0"
                # If there are exactly 3 digits after the dot and the integer part is not "0",
                # it is highly likely a thousand separator in German.
                parts = s.split(".")
                if len(parts) == 2 and len(parts[1]) == 3 and parts[0] != "0":
                    s = s.replace(".", "")

        if "." in s:
            a, b = s.split(".", 1)
            if a and b:
                if len(b) == 1:
                    frac = self._say_int(b)
                else:
                    frac = " ".join(self._say_int(d) for d in b)
                result = f"{self._say_int(a)} Komma {frac}"
            else:
                result = self._say_int(s.replace(".", "") or "0")
        else:
            result = self._say_int(s)

        return f"minus {result}" if is_negative else result

    def _repl_time(self, m: "re.Match") -> str:
        h, mm = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 23 and 0 <= mm <= 59):
            return m.group(0)
        if mm == 0:
            return f"{self._say_int(h)} Uhr"
        return f"{self._say_int(h)} Uhr {self._say_int(mm)}"

    def _repl_percent(self, m: "re.Match") -> str:
        return f" {self._num_token(m.group(1))} Prozent "

    def _repl_deg(self, m: "re.Match") -> str:
        return f" {self._num_token(m.group(1))} Grad "

    def _repl_num(self, m: "re.Match") -> str:
        return self._num_token(m.group(0))

    @staticmethod
    def _is_ipv4(addr: str) -> bool:
        try:
            ipaddress.IPv4Address(addr)
        except ValueError:
            return False
        return True

    @staticmethod
    def _is_cidr(addr: str, prefix: str) -> bool:
        try:
            ipaddress.ip_network(f"{addr}/{prefix}", strict=False)
        except ValueError:
            return False
        return True

    def _repl_dotted(self, m: "re.Match") -> str:
        addr, prefix, port = m.group(1), m.group(2), m.group(3)
        groups = addr.split(".")
        is_ipv4 = self._is_ipv4(addr)
        # German thousands grouping (1-3 digits, then groups of exactly 3):
        # "1.500.000" and "1.500.000.000" are still one number and left to
        # _NUM_PATTERN - unless the sequence is a valid IPv4 address such as
        # "192.168.178.100".
        if not is_ipv4 and len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]):
            return m.group(0)
        words = " Punkt ".join(self._say_int(g) for g in groups)
        if prefix is not None:
            if is_ipv4 and self._is_cidr(addr, prefix):
                words += f" Schrägstrich {self._say_int(prefix)}"
            else:
                words += f"/{prefix}"
        if port is not None:
            if is_ipv4:
                # "10.85.0.15:13305" - no port semantics: the colon is read as
                # "Doppelpunkt" and the digits after it are spelled one by one.
                words += " Doppelpunkt " + " ".join(self._say_int(d) for d in port)
            else:
                words += f":{port}"
        return words

    def _spell_char(self, ch: str) -> str:
        if ch.isdigit():
            return self._say_int(ch)
        if ch == ":":
            return "Doppelpunkt"
        return ch.lower()

    def _repl_ipv6(self, m: "re.Match") -> str:
        addr, prefix = m.group(1), m.group(2)
        try:
            ipaddress.IPv6Address(addr)
        except ValueError:
            return m.group(0)
        # Every group character by character (digits as number words, hex
        # letters as letters), "Doppelpunkt" for each ":" - so "::" becomes
        # "Doppelpunkt Doppelpunkt".
        words = " ".join(self._spell_char(ch) for ch in addr)
        if prefix is not None:
            if self._is_cidr(addr, prefix):
                words += f" Schrägstrich {self._say_int(prefix)}"
            else:
                words += f"/{prefix}"
        return words
