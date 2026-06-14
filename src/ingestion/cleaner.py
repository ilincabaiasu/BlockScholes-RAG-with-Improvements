from __future__ import annotations

import re
import unicodedata


def clean_page_text(raw_text: str) -> tuple[str, list[str]]:
    """Normalise *raw_text* and extract section headings.

    Returns:
        cleaned_text: normalised string.
        section_titles: section headings found in order of appearance.
    """
    # 1. Unicode NFC normalisation
    text = unicodedata.normalize("NFC", raw_text)

    # 2. Fix PDF hyphenation artefacts: "word-\nletter" → "wordletter"
    #    Only rejoin when the character before the hyphen is alphanumeric
    #    and the first character on the next line is a lowercase letter.
    text = re.sub(r"(?<=[A-Za-z0-9])-\n(?=[a-z])", "", text)

    # 3. Collapse runs of more than two newlines into exactly two newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Collapse runs of more than one horizontal space into a single space.
    #    Use [^\S\n] to match spaces/tabs but NOT newlines.
    text = re.sub(r"[^\S\n]{2,}", " ", text)

    # 5. Preserve: Markdown pipe syntax, numeric expressions ("25-delta",
    #    "1.5x", "-0.3 vega", "vol-of-vol"). The substitutions above already
    #    leave hyphens surrounded by alphanumerics intact because step 2 only
    #    targets line-end hyphens before lowercase letters.

    # --- section heading detection ---
    section_titles: list[str] = []
    lines = text.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Length check
        if not (5 <= len(stripped) <= 80):
            continue

        # Must not end with a period
        if stripped.endswith("."):
            continue

        # Style check: ALL CAPS, OR starts with digit+dot, OR Title Case
        is_all_caps = stripped == stripped.upper() and any(c.isalpha() for c in stripped)
        is_numbered = bool(re.match(r"^\d+\.\s+\S", stripped))
        is_title_case = stripped.istitle()

        if not (is_all_caps or is_numbered or is_title_case):
            continue

        # Next non-empty line must start with a lowercase letter or digit
        next_line = ""
        for j in range(i + 1, len(lines)):
            candidate = lines[j].strip()
            if candidate:
                next_line = candidate
                break

        if not next_line:
            continue

        if not (next_line[0].islower() or next_line[0].isdigit()):
            continue

        section_titles.append(stripped)

    return text, section_titles
