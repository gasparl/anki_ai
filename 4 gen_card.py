#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Tagalog Intro — Anki pack builder

What it does:
- Reads json/tl_cards.json
- Looks for readings/{id}_Angelo.wav and readings/{id}_Blessica.wav
- Sorts items by integer id
- Creates an .apkg next to the script with a minute-level timestamp
- Adds BOTH directions for each 30-item block:
    1) V1 TL->EN for the block  (tag: v1)
    2) V2 EN->TL for the same   (tag: v2)

Just run:
    python build_anki_from_json.py

Requires:
    pip install genanki
"""

import json
from pathlib import Path
from datetime import datetime
import random
import re
from typing import List, Tuple, Dict, Any

import genanki

# ======= Constants =======
DECK_NAME = "AI Tagalog Intro"
DECK_ID = 1765432109
BLOCK_SIZE = 30
MALE_SUFFIX = "Angelo"
FEMALE_SUFFIX = "Blessica"
AUDIO_EXT = ".wav"

def project_paths() -> Tuple[Path, Path, Path, Path]:
    base = Path(__file__).resolve().parent
    json_path = base / "json" / "tl_cards.json"
    readings_dir = base / "readings"
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = base / f"{DECK_NAME.replace(' ', '')}_{ts}.apkg"
    return base, json_path, readings_dir, out_path

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", (s or "")).strip("-").lower()

def norm(s: str) -> str:
    return " ".join((s or "").split())

def render_word_breakdown(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        items = "".join(f"<li>{str(x)}</li>" for x in value)
        return f"<ul class='wb'>{items}</ul>"
    return str(value)

def build_models() -> Tuple[genanki.Model, genanki.Model]:
    css = """
.card { font-size: 20px; text-align: left; }
.front { 
  font-size: 24px; 
  margin: 30px 0;
  text-align: center;
}
.audio {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin: 8px auto 16px;
  width: 100%;
  text-align: center;
}
.btn {
  font-weight: 600;
  padding: 2px 8px;
  border: 1px solid;
  border-radius: 6px;
}
.meta  { margin-top: 10px; }
.meta .label { font-weight: 600; }
.wb { padding-left: 22px; }
hr { border: none; border-top: 1px solid; opacity: .25; margin: 14px 0; }
"""

    fields = [
        {"name": "ID"},
        {"name": "SentenceTL"},
        {"name": "SentenceEN"},
        {"name": "ExplanationEN"},
        {"name": "IPA"},
        {"name": "WordBreakdown"},
        {"name": "MaleAudio"},
        {"name": "FemaleAudio"},
        {"name": "Stage"},
        {"name": "TargetWord"},
        {"name": "GrammarID"},
        {"name": "GrammarTitle"},
    ]

    # Fixed templates - properly escaped for Anki
    tl_to_en = genanki.Model(
        model_id=1785632491,
        name="AI Tagalog Intro — TL->EN (dual audio)",
        fields=fields,
        templates=[{
            "name": "V1 TL->EN",
            "qfmt": """
<div class="front">{{SentenceTL}}</div>
<div class="audio">{{MaleAudio}}<span class="btn">A</span> {{FemaleAudio}}<span class="btn">B</span></div>
""",
            "afmt": """
{{FrontSide}}
<hr id="answer">
<div class="meta"><span class="label">English:</span> {{SentenceEN}}</div>
<div class="meta"><span class="label">Explanation:</span> {{ExplanationEN}}</div>
<div class="meta"><span class="label">IPA:</span> {{IPA}}</div>
<div class="meta"><span class="label">Word breakdown:</span> {{WordBreakdown}}</div>
<hr>
""",
        }],
        css=css,
    )

    en_to_tl = genanki.Model(
        model_id=1785632492,
        name="AI Tagalog Intro — EN->TL (dual audio)",
        fields=fields,
        templates=[{
            "name": "V2 EN->TL",
            "qfmt": """
<div class="front">{{SentenceEN}}</div>
<div class="audio">{{MaleAudio}}<span class="btn">A</span> {{FemaleAudio}}<span class="btn">B</span></div>
""",
            "afmt": """
{{FrontSide}}
<hr id="answer">
<div class="meta"><span class="label">Tagalog:</span> {{SentenceTL}}</div>
<div class="meta"><span class="label">Explanation:</span> {{ExplanationEN}}</div>
<div class="meta"><span class="label">IPA:</span> {{IPA}}</div>
<div class="meta"><span class="label">Word breakdown:</span> {{WordBreakdown}}</div>
<hr>
""",
        }],
        css=css,
    )

    return tl_to_en, en_to_tl

def note_guid(deck_ns: int, note_id: int, direction: str) -> int:
    random.seed(f"{deck_ns}-{note_id}-{direction}")
    return random.getrandbits(63)

def get_audio_fields(item_id: int, readings_dir: Path, media_files: List[str], seen_media: set) -> Tuple[str, str]:
    male_name = f"{item_id}_{MALE_SUFFIX}{AUDIO_EXT}"
    fem_name = f"{item_id}_{FEMALE_SUFFIX}{AUDIO_EXT}"
    
    male_path = readings_dir / male_name
    fem_path = readings_dir / fem_name

    # Track media files
    for path in (male_path, fem_path):
        if path.exists() and path.as_posix() not in seen_media:
            media_files.append(str(path))
            seen_media.add(path.as_posix())

    male_field = f"[sound:{male_name}]" if male_path.exists() else ""
    female_field = f"[sound:{fem_name}]" if fem_path.exists() else ""
    
    return male_field, female_field

def build_note_fields(item: Dict[str, Any], male_field: str, female_field: str) -> List[str]:
    return [
        str(item.get("id", "")),
        norm(item.get("sentence_tl", "")),
        norm(item.get("sentence_en", "")),
        norm(item.get("explanation_en", "")),
        norm(item.get("ipa", "")),
        render_word_breakdown(item.get("word_breakdown")),
        male_field,
        female_field,
        norm(item.get("stage", "")),
        norm(item.get("target_word", "")),
        norm(item.get("grammar_id", "")),
        norm(item.get("grammar_title", "")),
    ]

def add_note(deck, model, item, idx_for_guid: int, direction: str,
             readings_dir: Path, media_files: List[str], seen_media: set):
    item_id = item.get("id", idx_for_guid)
    
    # Validate front field based on direction
    front_field = "sentence_tl" if direction == "V1" else "sentence_en"
    if not norm(item.get(front_field, "")):
        return

    male_field, female_field = get_audio_fields(item_id, readings_dir, media_files, seen_media)
    fields = build_note_fields(item, male_field, female_field)

    guid_seed_id = int(item_id) if str(item_id).isdigit() else idx_for_guid
    base_tags = [slugify(item.get("stage", ""))] if item.get("stage") else []
    version_tag = ["v1"] if direction == "V1" else ["v2"]

    note = genanki.Note(
        model=model,
        fields=fields,
        guid=note_guid(DECK_ID, guid_seed_id, direction),
        tags=base_tags + version_tag,
    )
    deck.add_note(note)

def process_block(deck, block, start_idx: int, direction: str, model, 
                 readings_dir: Path, media_files: List[str], seen_media: set):
    for i, item in enumerate(block):
        idx = start_idx + i
        add_note(deck, model, item, idx, direction, readings_dir, media_files, seen_media)

def main():
    base, json_path, readings_dir, out_path = project_paths()

    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found at {json_path}")

    readings_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of card objects.")

    data.sort(key=lambda x: int(x.get("id", 0)))
    tl_to_en_model, en_to_tl_model = build_models()
    deck = genanki.Deck(DECK_ID, DECK_NAME)
    media_files = []
    seen_media = set()

    # Process in blocks: V1 then V2 for each block
    for start in range(0, len(data), BLOCK_SIZE):
        block = data[start:start + BLOCK_SIZE]
        process_block(deck, block, start, "V1", tl_to_en_model, readings_dir, media_files, seen_media)
        process_block(deck, block, start, "V2", en_to_tl_model, readings_dir, media_files, seen_media)

    # Package and write
    pkg = genanki.Package(deck)
    pkg.media_files = media_files
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pkg.write_to_file(str(out_path))

    print(f"✅ Wrote {out_path.name} with {len(deck.notes)} notes and {len(media_files)} media files.")
    print(f"   Location: {out_path}")

if __name__ == "__main__":
    main()