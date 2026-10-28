"""Czech text normalizer for Wyoming-Supertonic.

Converts digits, decimals, clock times, percentages, degrees, IPv4/IPv6
addresses and dotted version numbers into spoken Czech words so a fast local TTS
that cannot handle "," "." ":" inside numbers produces correct speech.

Design notes:
- fail-open: any internal error returns the ORIGINAL text unmodified.
- expression tags like <laugh> are preserved.
- Czech plural declension (1 / 2-4 / 5+) is applied to the few units we emit.
"""

import ipaddress
import logging
import re

from num2words import num2words

log = logging.getLogger(__name__)


class CzechTextNormalizer:

    _TAGS_PATTERN = re.compile(r"(<[a-zA-Z/]+>)")

    _chars_to_delete = "=#$“”„«»*\"‘’‚`"

    _emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF☀-⛿✀-➿\U0001F900-\U0001F9FF"
        "‍️"
        "]+",
        flags=re.UNICODE,
    )

    _TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(\s*(?:Uhr|hodin?)\.?)?\b")
    _PERCENT_PATTERN = re.compile(r"(-?\d+(?:[.,]\d+)*)\s*%")
    _DEG_C_PATTERN = re.compile(r"(-?\d+(?:[.,]\d+)*)\s*°\s*C\b")
    _DEG_PATTERN = re.compile(r"(-?\d+(?:[.,]\d+)*)\s*°")
    _NUM_PATTERN = re.compile(r"(?:(?<=^)|(?<=\s))-\d+(?:[.,]\d+)*|\b\d+(?:[.,]\d+)*")
    _DOTTED_PATTERN = re.compile(
        r"(?<![\w.,])(\d+(?:\.\d+){2,})(?![\w]|[.,]\d)(?:/(\d{1,3})(?!\w))?(?::(\d+))?"
    )
    _IPV6_PATTERN = re.compile(
        r"(?<![\w:.])(?=[0-9a-fA-F:]*:[0-9a-fA-F:]*:)([0-9a-fA-F:]+)(?![\w:]|\.\d)(?:/(\d{1,3})(?!\w))?"
    )

    def __init__(self) -> None:
        self._del_table = str.maketrans("", "", self._chars_to_delete)

    def normalize(self, text: str) -> str:
        try:
            return self._normalize(text)
        except Exception as e:
            log.warning("cs_norm failed, passing text through unmodified: %s", e)
            return text

    def _normalize(self, text: str) -> str:
        if not text or not text.strip():
            return text

        parts = self._TAGS_PATTERN.split(text)
        out = []
        for part in parts:
            if not part:
                continue
            if self._TAGS_PATTERN.match(part):
                out.append(part)
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
        v = str(value).lstrip("-").lstrip("−")
        return num2words(int(v), lang="cs")

    @staticmethod
    def _plural_form(value, singular: str, paucal: str, plural: str) -> str:
        """Czech 1 / 2-4 / 5+ (with 11-14 -> plural) unit declension."""
        s = str(value).replace(",", ".").split(".")[0].lstrip("-").lstrip("−")
        try:
            i = int(s) if s else 0
        except ValueError:
            i = 0
        last_two = abs(i) % 100
        if last_two == 1:
            return singular
        if 2 <= last_two <= 4:
            return paucal
        return plural

    def _num_token(self, s: str) -> str:
        is_negative = s.startswith("-") or s.startswith("−")
        if is_negative:
            s = s[1:]

        if "," in s:
            s = s.replace(".", "").replace(",", ".")
        elif "." in s:
            if s.count(".") > 1:
                s = s.replace(".", "")
            else:
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
                result = f"{self._say_int(a)} celá {frac}"
            else:
                result = self._say_int(s.replace(".", "") or "0")
        else:
            result = self._say_int(s)

        return f"mínus {result}" if is_negative else result

    def _repl_time(self, m: "re.Match") -> str:
        h, mm = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 23 and 0 <= mm <= 59):
            return m.group(0)
        hour_form = self._plural_form(h, "hodina", "hodiny", "hodin")
        if h == 1:
            hour_word = "jedna"
        elif h == 2:
            hour_word = "dvě"
        elif h == -1:
            hour_word = "mínus jedna"
        elif h == -2:
            hour_word = "mínus dvě"
        else:
            hour_word = self._say_int(h)
        if mm == 0:
            return f"{hour_word} {hour_form}"
        return f"{hour_word} {hour_form} {self._say_int(mm)}"

    def _repl_percent(self, m: "re.Match") -> str:
        num = m.group(1)
        form = self._plural_form(num, "procento", "procenta", "procent")
        if num == "1":
            word = "jedno"
        elif num == "2":
            word = "dvě"
        elif num == "-1":
            word = "mínus jedno"
        elif num == "-2":
            word = "mínus dvě"
        else:
            word = self._num_token(num)
        return f" {word} {form} "

    def _repl_deg(self, m: "re.Match") -> str:
        num = m.group(1)
        form = self._plural_form(num, "stupeň", "stupně", "stupňů")
        if num == "1":
            word = "jeden"
        elif num == "2":
            word = "dva"
        elif num == "-1":
            word = "mínus jeden"
        elif num == "-2":
            word = "mínus dva"
        else:
            word = self._num_token(num)
        return f" {word} {form} "

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
        if not is_ipv4 and len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]):
            return m.group(0)
        words = " tečka ".join(self._say_int(g) for g in groups)
        if prefix is not None:
            if is_ipv4 and self._is_cidr(addr, prefix):
                words += f" lomítko {self._say_int(prefix)}"
            else:
                words += f"/{prefix}"
        if port is not None:
            if is_ipv4:
                words += " dvojtečka " + " ".join(self._say_int(d) for d in port)
            else:
                words += f":{port}"
        return words

    def _spell_char(self, ch: str) -> str:
        if ch.isdigit():
            return self._say_int(ch)
        if ch == ":":
            return "dvojtečka"
        return ch.lower()

    def _repl_ipv6(self, m: "re.Match") -> str:
        addr, prefix = m.group(1), m.group(2)
        try:
            ipaddress.IPv6Address(addr)
        except ValueError:
            return m.group(0)
        words = " ".join(self._spell_char(ch) for ch in addr)
        if prefix is not None:
            if self._is_cidr(addr, prefix):
                words += f" lomítko {self._say_int(prefix)}"
            else:
                words += f"/{prefix}"
        return words