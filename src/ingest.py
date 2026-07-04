"""Parse 108upanishads.pdf into verse-level chunks with metadata.

Reads the PDF (or a cached page dump at data/raw/pages.json), detects each
Upanishad section, strips the commentary sections and the trailing glossary,
splits the translation into verses using whichever verse-numbering style the
text uses, and writes data/processed/verses.json.

Each output record:
    {
      "id":         "katha_1-I-3",
      "upanishad":  "Katha",
      "ref":        "1-I-3",          # verse reference as printed in the text
      "text":       "...",
      "translator": "Vidyavachaspati V. Panoli"
    }
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "108upanishads.pdf"
PAGES_CACHE = ROOT / "data" / "raw" / "pages.json"
OUT_PATH = ROOT / "data" / "processed" / "verses.json"

SECTION_HEADER = re.compile(r"^\s*([A-Z][A-Za-z'\- ]+? Upanishad)\s*\*?\s*$")
GLOSSARY_HEADER = "A Brief Sanskrit Glossary"
COMMENTARY = re.compile(r"Upanishad Commentary|^Commentary on the ", re.M)

# Verse-numbering styles, most specific first. A style is accepted only if it
# matches at least MIN_STRUCTURED times, so stray numbered lists inside prose
# don't get mistaken for verse numbers.
VERSE_STYLES = [
    ("part-canto-verse", re.compile(r"(?m)^\s*(\d+-[IVX]+-\d+(?:-\d+)?)\.\s*")),      # Katha: 1-I-1.
    ("ch-sec-verse", re.compile(r"(?m)^\s*([IVX]+-[ivxIVX]+-\d+(?:-\d+)?)[:.]\s*")),  # Mundaka: I-i-1:
    ("ch-verse", re.compile(r"(?m)^\s*([IVX]+-\d+(?:-\d+)?)[:.]\s*")),                # Prasna: I-1:
    ("plain", re.compile(r"(?m)^\s*(\d+(?:-\d+)?)\.\s+")),                            # Isa: 1.
]
MIN_STRUCTURED = 5
MIN_PLAIN = 3
MIN_VERSE_CHARS = 25


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
    # The PDF's em-dashes and curly quotes come out as U+FFFD. Restore
    # apostrophes inside contractions; render the rest as an em-dash.
    text = re.sub(r"(?<=\w)�(?=(?:t|s|d|m|ll|re|ve)\b)", "'", text)
    text = text.replace("�", " — ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_sections(pages: list[str]) -> list[tuple[str, int, int]]:
    """Return (name, start_page_idx, end_page_idx_exclusive) per Upanishad."""
    starts = []
    glossary_at = len(pages)
    for i, page in enumerate(pages):
        head_lines = page.split("\n")[:6]
        for line in head_lines:
            if GLOSSARY_HEADER in line and glossary_at == len(pages):
                glossary_at = i
            m = SECTION_HEADER.match(line)
            if m:
                starts.append((m.group(1).strip(), i))
                break
    sections = []
    for j, (name, start) in enumerate(starts):
        end = starts[j + 1][1] if j + 1 < len(starts) else glossary_at
        sections.append((name, start, min(end, glossary_at)))
    return sections


def strip_front_matter(lines: list[str]) -> tuple[list[str], str]:
    """Drop title/translator/publisher lines; return remaining lines + translator."""
    translator = ""
    body_start = 0
    for i, line in enumerate(lines[:6]):
        s = line.strip()
        if SECTION_HEADER.match(s):
            body_start = i + 1
        elif s.startswith(("Translated by", "Translaetd by")):  # sic, typo in PDF
            translator = clean_text(s.split("by", 1)[1])
            body_start = i + 1
        elif s.startswith("Published by"):
            body_start = i + 1
    return lines[body_start:], translator


def pick_style(text: str):
    for style, rx in VERSE_STYLES:
        n = len(rx.findall(text))
        if n >= (MIN_PLAIN if style == "plain" else MIN_STRUCTURED):
            return style, rx
    return None, None


def split_verses(name: str, text: str) -> list[dict]:
    style, rx = pick_style(text)
    records = []

    def add(ref: str, chunk: str):
        chunk = clean_text(chunk)
        if len(chunk) < MIN_VERSE_CHARS:
            return
        # Split over-long verses at sentence boundaries so each piece fits the
        # embedding model's window; they keep the same ref for citation.
        if len(chunk) > 1200:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            buf: list[str] = []
            for s in sentences:
                buf.append(s)
                if sum(len(x) for x in buf) > 800:
                    records.append(
                        {"upanishad": name, "ref": ref, "text": " ".join(buf)}
                    )
                    buf = []
            if buf:
                records.append({"upanishad": name, "ref": ref, "text": " ".join(buf)})
        else:
            records.append({"upanishad": name, "ref": ref, "text": chunk})

    if rx is None:
        # Prose text with no verse numbers: chunk by sentence groups.
        sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
        buf, n = [], 1
        for s in sentences:
            buf.append(s)
            if sum(len(x) for x in buf) > 600:
                add(f"passage {n}", " ".join(buf))
                buf, n = [], n + 1
        if buf:
            add(f"passage {n}", " ".join(buf))
        return records

    matches = list(rx.finditer(text))
    invocation = text[: matches[0].start()]
    add("Invocation", invocation)
    for m, nxt in zip(matches, matches[1:] + [None]):
        end = nxt.start() if nxt else len(text)
        add(m.group(1), text[m.end():end])
    return records


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower().replace(" upanishad", "")).strip("-")


def main() -> None:
    pages = load_pages()
    sections = find_sections(pages)
    print(f"Found {len(sections)} Upanishad sections")

    all_records = []
    for name, start, end in sections:
        text = "\n".join(pages[start:end])
        cut = COMMENTARY.search(text)
        if cut:
            text = text[: cut.start()]
        lines, translator = strip_front_matter(text.split("\n"))
        short_name = name.replace(" Upanishad", "")
        records = split_verses(short_name, "\n".join(lines))
        slug = slugify(name)
        for k, r in enumerate(records):
            r["id"] = f"{slug}_{re.sub(r'[^A-Za-z0-9-]+', '-', r['ref'])}_{k}"
            r["translator"] = translator
        all_records.extend(records)
        print(f"  {short_name:<35} {len(records):>4} chunks")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nWrote {len(all_records)} chunks -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
