If you have specific text processing needs for your language, you can create a new normalization file and add the necessary actions to it that are applied to each sentence.

### Overview
The engine uses a modular architecture. To add a new language, you need to create a standalone normalizer script and register it in the main engine.

---

### Step 1: Create the Normalizer Script
Create a new file in the same directory as your engine, for example: `de_norm.py` (for German). 

Your class must implement a `normalize` method. It is highly recommended to use the `num2words` library for consistency.

**Template (`de_norm.py`):**
```python
import logging
import re
from num2words import num2words

# Initialize logger to report normalization errors safely
logger = logging.getLogger(__name__)

class GermanTextNormalizer:
    def __init__(self):
        # Add your character cleanup rules here
        self._chars_to_delete = "=#$“”„«»*\""
        self._translation_table = str.maketrans("", "", self._chars_to_delete)

    def normalize(self, text: str) -> str:
        if not text.strip():
            return text
            
        try:
            # 1. Basic cleanup
            cleaned_text = text.translate(self._translation_table)
            
            # 2. Convert numbers to words (example)
            cleaned_text = re.sub(
                r'\b\d+\b', 
                lambda m: num2words(int(m.group(0)), lang='de'), 
                cleaned_text
            )
            
            # 3. Final cleanup (ensure multiple spaces are removed)
            return " ".join(cleaned_text.split())
            
        except Exception as e:
            # Fail-open: if anything fails, log a warning and return original text
            logger.warning("Normalization failed, falling back to original text: %s", e)
            return text
```

The `de_norm.py` shipped in this repository is a more complete example: besides plain numbers it handles clock times, percentages, degrees, IPv4/IPv6 addresses with CIDR prefixes and dotted version numbers, and it comes with unit tests in `tests/test_de_norm.py` (`python3 -m unittest discover -s tests`, only `num2words` needed).

---

### Step 2: Register the Normalizer in `supertonic_engine.py`


The engine uses a dynamic registry to load normalizers. You only need to add a single line to tell the engine about your new script.

Open `supertonic_engine.py` and locate the `NORMALIZERS_REGISTRY` dictionary near the top of the file. Add your language code, the module name (the file name without `.py`, preceded by a dot), and your class name.

**Example:**
```python
# Registry of available normalizers: "lang_code": ("module_name", "ClassName")
NORMALIZERS_REGISTRY = {
    "ru": (".ru_norm", "RussianTextNormalizer"),
    "de": (".de_norm", "GermanTextNormalizer"),  # <-- Add your new language here
}
```
---

### Step 3: Test the Integration
1.  **Install dependencies**: If your new normalizer uses a specific library, ensure it's installed (`pip install num2words`).
2.  **Restart the server**: The engine will now automatically route any synthesis request with `lang_code="de"` through your `GermanTextNormalizer`.
3.  **Check logs**: Because of the logic in `synthesize`, you will see a debug log whenever the text is modified by your new normalizer:
    `DEBUG: [de] Text prepared for synthesis: einhundert ...`

### Key Requirements for New Normalizers:
*   **Language Codes**: Use 2-letter ISO codes (e.g., `es`, `fr`, `it`). The engine automatically converts incoming codes like `en-US` to `en`.
*   **Safety**: Always use `try-except` for imports so the server can still start even if a specific language dependency is missing.
*   **Tags**: If you use expression tags (like `<laugh>`), ensure your normalizer does not delete the `<` and `>` brackets.
