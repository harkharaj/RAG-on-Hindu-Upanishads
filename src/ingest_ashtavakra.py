"""Parse ASHTAVAKRA_GITA.pdf (Chinmaya Mission edition) into verse chunks.

The book interleaves, for every verse: Devanagari, Roman transliteration
(ending with the verse number, e.g. "... prabho. (1)"), a word-by-word
gloss, the English verse translation ("1. Teach me this, O Lord! ..."),
and then Swami Chinmayananda's prose commentary.

Only the verse translation is kept (the commentary is the commentator's
voice, not Ashtavakra's — and the Upanishad pipeline strips commentary
too, so the two corpora stay comparable). The translation is set in
Arial-Italic while the commentary is regular Arial, so the end of a
translation is detected by font: lines stay part of the verse while the
majority of their characters are italic.

Writes data/processed/ashtavakra.json with the same record shape as
verses.json plus a "source" field:
    {
      "id":         "ashtavakra_1-1",
      "source":     "ashtavakra",
      "upanishad":  "Ashtavakra",   # work name; key kept for pipeline compat
      "ref":        "1.1",          # chapter.verse
      "text":       "...",
      "translator": "Swami Chinmayananda"
    }
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "ASHTAVAKRA_GITA.pdf"
LINES_CACHE = ROOT / "data" / "raw" / "ashtavakra_lines.json"
OUT_PATH = ROOT / "data" / "processed" / "ashtavakra.json"

BODY_FROM_PAGE = 17  # chapter 1 starts here; earlier pages are front matter
CHAPTER = re.compile(r"^\s*Chapter\s*[–-]?\s*(\d+)\s*$")
TRANSLIT_END = re.compile(r"\.\s*\((\d+(?:-\d+)?)\)\s*$")
SPEAKER = re.compile(r"^(Janaka|A[sṣ][tṭ][aā]vakra)\s+said\s*:?\s*$", re.I)
ITALIC_MIN = 0.5  # a line is still verse translation while ≥50% italic
MIN_VERSE_CHARS = 20

# Canonical verse counts per chapter, used to validate the extraction.
EXPECTED = {1: 20, 2: 25, 3: 14, 4: 6, 5: 4, 6: 4, 7: 5, 8: 4, 9: 8, 10: 8,
            11: 8, 12: 8, 13: 7, 14: 4, 15: 20, 16: 11, 17: 20, 18: 100,
            19: 8, 20: 14}


def extract_lines() -> list[list[dict]]:
    """Per page, reading-order lines as {"text": str, "italic": fraction}."""
    if LINES_CACHE.exists():
        return json.loads(LINES_CACHE.read_text(encoding="utf-8"))
    from pypdf import PdfReader

    reader = PdfReader(str(PDF_PATH))
    pages = []
    for page in reader.pages:
        runs: list[tuple[float, float, str, bool]] = []

        def visitor(text, cm, tm, font_dict, font_size):
            if not text.strip():
                return
            font = str(font_dict.get("/BaseFont", "")) if font_dict else ""
            italic = "Italic" in font or "Oblique" in font
            runs.append((tm[5], tm[4], text, italic))

        page.extract_text(visitor_text=visitor)

        # Group runs into lines by y (PDF y grows upward), tolerance 2 units.
        lines: list[dict] = []
        current_y = None
        buf: list[tuple[float, str, bool]] = []

        def flush():
            if not buf:
                return
            buf.sort(key=lambda r: r[0])
            text = "".join(r[1] for r in buf)
            n_italic = sum(len(r[1]) for r in buf if r[2])
            n_total = sum(len(r[1]) for r in buf) or 1
            lines.append({"text": text.strip(), "italic": n_italic / n_total})

        for y, x, text, italic in runs:
            if current_y is None or abs(y - current_y) > 2:
                flush()
                buf = []
                current_y = y
            buf.append((x, text, italic))
        flush()
        pages.append(lines)

    LINES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LINES_CACHE.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_verses(pages: list[list[dict]]) -> list[dict]:
    # Flatten body pages, keeping chapter boundaries.
    lines: list[dict] = []
    for page in pages[BODY_FROM_PAGE:]:
        lines.extend(page)

    records = []
    chapter = 0
    pending = None  # verse number announced by the transliteration line
    i = 0
    while i < len(lines):
        text = lines[i]["text"]
        m = CHAPTER.match(text)
        if m:
            chapter, pending = int(m.group(1)), None
            i += 1
            continue
        m = TRANSLIT_END.search(text)
        if m:
            pending = m.group(1)
            i += 1
            continue
        if pending and chapter:
            # \s* not \s+ — some verses render as "4.The Self..." with no space
            m = re.match(rf"^{re.escape(pending)}\.\s*(.*)", text)
            if m:
                buf = [m.group(1)]
                if lines[i]["italic"] >= ITALIC_MIN:
                    while i + 1 < len(lines) and lines[i + 1]["italic"] >= ITALIC_MIN:
                        i += 1
                        buf.append(lines[i]["text"])
                else:
                    # A few verses are mis-set in roman type (e.g. 18.80);
                    # fall back to collecting until a sentence-ending line.
                    while (
                        not re.search(r"[.!?…][\"”')\]]*\s*$", buf[-1])
                        and i + 1 < len(lines)
                    ):
                        i += 1
                        buf.append(lines[i]["text"])
                verse = clean_text(" ".join(buf))
                prev = lines[i - len(buf)]["text"] if i >= len(buf) else ""
                speaker = SPEAKER.match(prev.strip())
                if speaker:
                    verse = f"{speaker.group(1)} said: {verse}"
                if len(verse) >= MIN_VERSE_CHARS:
                    records.append(
                        {
                            "id": f"ashtavakra_{chapter}-{pending}",
                            "source": "ashtavakra",
                            "upanishad": "Ashtavakra",
                            "ref": f"{chapter}.{pending}",
                            "text": verse,
                            "translator": "Swami Chinmayananda",
                        }
                    )
                pending = None
        i += 1
    return records


def main() -> None:
    pages = extract_lines()
    records = parse_verses(pages)

    counts: dict[int, int] = {}
    for r in records:
        counts[int(r["ref"].split(".")[0])] = counts.get(int(r["ref"].split(".")[0]), 0) + 1
    ok = True
    for ch in sorted(EXPECTED):
        got = counts.get(ch, 0)
        flag = "" if got == EXPECTED[ch] else f"  <-- expected {EXPECTED[ch]}"
        if flag:
            ok = False
        print(f"  Chapter {ch:>2}: {got:>3} verses{flag}")
    if not ok:
        print("WARNING: verse counts differ from the canonical 298", file=sys.stderr)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nWrote {len(records)} chunks -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
