"""Parse Srimad-Bhagavad-Gita-Shankara-Bhashya-English.pdf into verse chunks.

Gambhirananda's translation (via Project Gutenberg). For every verse the
book prints: the verse translation ("2.20 Never is this One born..."),
then a separator line ("English Translation of Sri Sankaracharya's
Sanskrit Commentary - Swami Gambhirananda"), then Sankara's commentary,
which opens with the same verse ref and runs until the next verse's
translation.

Only the verse translation is kept (the commentary is Sankara's voice,
not the Gita's — same policy as the other corpora). Note: this edition
covers chapters 2-18 starting at 2.10, because Sankara wrote no
commentary on chapter 1 and 2.1-2.9.

Writes data/processed/gita.json, same record shape as the others:
    {
      "id":         "gita_2-20",
      "source":     "gita",
      "upanishad":  "Gita",        # work name; key kept for pipeline compat
      "ref":        "2.20",        # chapter.verse, or e.g. "2.42-44"
      "text":       "...",
      "translator": "Swami Gambhirananda"
    }
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "Srimad-Bhagavad-Gita-Shankara-Bhashya-English.pdf"
PAGES_CACHE = ROOT / "data" / "raw" / "gita_pages.json"
OUT_PATH = ROOT / "data" / "processed" / "gita.json"

# "2.20", or a grouped ref like "2.42-2.43" / "5.8-9"
REF = re.compile(r"^\s*(\d{1,2}\.\d{1,3}(?:\s*-\s*(?:\d{1,2}\.)?\d{1,3})?)\s+(.*)")
SEPARATOR = "English Translation of Sri Sankaracharya"
CHAPTER_LINE = re.compile(r"^\s*Chapter\s+\d+\s*$")
PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
MIN_VERSE_CHARS = 20

# Verses per chapter in the vulgate text, for a coverage report.
CANONICAL = {2: 72, 3: 43, 4: 42, 5: 29, 6: 47, 7: 30, 8: 28, 9: 34, 10: 42,
             11: 55, 12: 20, 13: 35, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78}


def load_pages() -> list[str]:
    if PAGES_CACHE.exists():
        return json.loads(PAGES_CACHE.read_text(encoding="utf-8"))
    from pypdf import PdfReader

    reader = PdfReader(str(PDF_PATH))
    pages = [(p.extract_text() or "") for p in reader.pages]
    PAGES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PAGES_CACHE.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ref_bounds(ref: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """("2.42-2.43" | "2.42-43" | "2.42") -> ((2, 42), (2, 43))."""
    parts = [p.strip() for p in ref.split("-")]
    ch, verse = (int(x) for x in parts[0].split("."))
    start = (ch, verse)
    if len(parts) == 1:
        return start, start
    end = parts[1].split(".")
    return start, (int(end[0]), int(end[1])) if len(end) == 2 else (ch, int(end[0]))


def parse_verses(pages: list[str]) -> list[dict]:
    lines = []
    for page in pages:
        for raw in page.split("\n"):
            s = raw.strip()
            if not s or PAGE_NUMBER.match(s) or CHAPTER_LINE.match(s):
                continue
            lines.append(s)

    records: list[dict] = []
    buf: list[str] | None = None  # translation lines being collected
    cur_ref = ""
    last_end = (0, 0)  # end of the last translated ref (range-aware)
    in_commentary = False

    def close():
        nonlocal buf
        if buf is None:
            return
        text = clean_text(" ".join(buf))
        if len(text) >= MIN_VERSE_CHARS:
            records.append(
                {
                    "id": f"gita_{cur_ref.replace('.', '-')}",
                    "source": "gita",
                    "upanishad": "Gita",
                    "ref": cur_ref,
                    "text": text,
                    "translator": "Swami Gambhirananda",
                }
            )
        buf = None

    for line in lines:
        if SEPARATOR in line:
            close()
            in_commentary = True
            continue
        m = REF.match(line)
        if m:
            start, end = ref_bounds(m.group(1))
            if start <= last_end:
                # The commentary opener repeats the ref just translated
                # (possibly one verse of a grouped ref, e.g. "2.43" after
                # "2.42-43"), and quoted refs inside commentary point
                # backward: not a new verse. If it happens while still
                # collecting a translation, the separator line was mangled —
                # this is the commentary opener, so close the verse (e.g. 2.69).
                if buf is not None and not in_commentary:
                    close()
                    in_commentary = True
                continue
            # A forward ref = the next verse's translation.
            close()
            in_commentary = False
            cur_ref = re.sub(r"\s*-\s*(?:\d{1,2}\.)?", "-", m.group(1))
            last_end = end
            buf = [m.group(2)]
            continue
        if buf is not None and not in_commentary:
            buf.append(line)
    close()
    return records


def main() -> None:
    records = parse_verses(load_pages())

    by_chapter: dict[int, int] = {}
    for r in records:
        ch = int(r["ref"].split(".")[0])
        n = len(r["ref"].split("-"))  # grouped refs cover several verses
        by_chapter[ch] = by_chapter.get(ch, 0) + (1 if n == 1 else n)
    print("Coverage (verses incl. grouped, vs vulgate count):")
    for ch in sorted(CANONICAL):
        print(f"  Chapter {ch:>2}: {by_chapter.get(ch, 0):>3} / {CANONICAL[ch]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nWrote {len(records)} chunks -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
