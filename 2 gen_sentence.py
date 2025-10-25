#!/usr/bin/env python3
"""
Tagalog card generator (Anki-ready JSON) - Fixed Wiktionary version
"""

from __future__ import annotations

import os
import time
import re
import json
import types
import logging
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime
import requests
import html

# --------------------------- easy testing toggle ---------------------------
TEST_MODE = True

TEST_LIMITS = {
    "greetings_items": 2,
    "expressions_items": 2,
    "grammar_drills_items": 2,
    "grammar_points": 2,
    "greet_per_word": 1,
    "expr_per_word": 1,
    "grammar_examples_per_point": 1,
}

# --------------------------- model ---------------------------

MODEL_NAME = "gpt-4o"

# --------------------------- logging ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
for noisy in ("openai", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("tl_cards")

# --------------------------- paths & config ---------------------------
HERE = Path(__file__).resolve().parent
CWD = Path.cwd()

CONFIG_CANDIDATES = [CWD / "config.json", HERE / "config.json"]
WORD_LIST_CANDIDATES = [CWD / "word_list.py", HERE / "word_list.py"]
GRAMMAR_LIST_CANDIDATES = [CWD / "grammar_list.py", HERE / "grammar_list.py"]

OUT_DIR = HERE / "json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_JSON = OUT_DIR / f"tl_cards_{TIMESTAMP}.json"
PARTIAL_JSON = OUT_DIR / f"tl_cards_{TIMESTAMP}.partial.json"
USAGE_JSON = OUT_DIR / f"openai_usage_summary_{TIMESTAMP}.json"

CHECKPOINT_EVERY = 10
if TEST_MODE:
    CHECKPOINT_EVERY = 1

# --------------------------- utilities ---------------------------
_token_re = re.compile(r"[\w'\-]+", re.UNICODE)


COMMON_TL_FUNCTION_WORDS = {
    "ang", "ng", "nang", "na", "ay", "sa", "si", "sina", "kay", "kina",
    "ko", "mo", "niya", "namin", "natin", "ninyo", "nila", "ito", "iyan",
    "iyon", "ba", "po", "ho", "rin", "din", "pa", "lang", "daw", "raw",
    "mga", "may", "wala", "meron", "hindi", "oo", "opo", "huwag", "salamat",
    "kamusta", "umaga", "gabi", "maganda", "masama", "mabuti", "sige", "pwede"
}

# Consolidated affix sets (removing duplicates from previous definitions)
TL_PREFIXES = {
    "pinaka", "pakiki", "pagpapa", "ipinag", "pinag", "nakiki", "makipag",
    "mapag", "mapa", "nakipag", "maki", "maka", "mag", "nag", "pag",
    "ipa", "ika", "i", "ma", "na", "ka", "pa"
}
TL_INFIXES = {"um", "in"}
TL_SUFFIXES = {"han", "hin", "in", "an", "nan", "nin"}

# For lemma detection - use sorted by length for priority
LEMMA_PREFIXES = sorted(TL_PREFIXES, key=len, reverse=True)
LEMMA_INFIXES = tuple(TL_INFIXES)
LEMMA_SUFFIXES = tuple(TL_SUFFIXES)

# Simple CV pattern for reduplication detection
CV_PATTERN = r"[bcdfghjklmnpqrstvwxyz]?[aeiou]"


# Progress tracking
class ProgressTracker:
    def __init__(self, total_expected: int = 0):
        self.total_expected = total_expected
        self.completed = 0
        self.last_update = time.time()
    
    def update(self, count: int = 1):
        self.completed += count
        current_time = time.time()
        if current_time - self.last_update > 0.5:
            print(".", end="", flush=True)
            self.last_update = current_time
    
    def finish(self):
        print()

# --------------------------- data loading ---------------------------
# Your existing data loading functions remain the same...
def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Failed to load %s: %s", path, e)
    return None

def import_module_from_path(path: Path, module_name: str) -> Optional[types.ModuleType]:
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_config() -> Dict[str, Any]:
    cfg = {}
    for p in CONFIG_CANDIDATES:
        c = load_json_if_exists(p)
        if c:
            cfg.update(c)
            break
    
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and not cfg.get("OPENAI_API_KEY"):
        cfg["OPENAI_API_KEY"] = env_key
    if cfg.get("openai_api_key") and not cfg.get("OPENAI_API_KEY"):
        cfg["OPENAI_API_KEY"] = cfg["openai_api_key"]
    
    return cfg

def dedupe_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def load_word_buckets() -> Dict[str, List[str]]:
    mod = None
    for p in WORD_LIST_CANDIDATES:
        mod = import_module_from_path(p, "word_list")
        if mod:
            break
    
    if not mod or not hasattr(mod, "WORD_BUCKETS"):
        raise RuntimeError("word_list.py with WORD_BUCKETS not found")
    
    wb = dict(mod.WORD_BUCKETS)
    for k in ("greetings", "expressions", "grammar_drills"):
        wb[k] = dedupe_preserve([str(x).strip() for x in wb.get(k, []) if str(x).strip()])
    
    return wb

def load_grammar_points() -> List[Dict[str, Any]]:
    mod = None
    for p in GRAMMAR_LIST_CANDIDATES:
        mod = import_module_from_path(p, "grammar_list")
        if mod:
            break
    
    if not mod or not hasattr(mod, "GRAMMAR_POINTS"):
        raise RuntimeError("grammar_list.py with GRAMMAR_POINTS not found")
    
    gps = []
    seen_ids, seen_titles = set(), set()
    for g in mod.GRAMMAR_POINTS:
        gid = str(g.get("id", "")).strip()
        title = str(g.get("title", "")).strip()
        note = str(g.get("note", "")).strip()
        
        if not gid or not title:
            continue
        if gid in seen_ids or title in seen_titles:
            continue
        
        gps.append({"id": gid, "title": title, "note": note})
        seen_ids.add(gid)
        seen_titles.add(title)
    
    return gps

# --------------------------- improved tokenization & heuristics ---------------------------
# Your existing tokenization functions remain the same...
def tokenize_simple(text: str) -> List[str]:
    return _token_re.findall(text.lower())

def looks_tl(text: str) -> bool:
    t = text.lower().strip()
    
    if not t or len(t) < 2:
        return False
    
    words = tokenize_simple(t)
    if not words:
        return False
    
    score = 0
    
    for w in COMMON_TL_FUNCTION_WORDS:
        if any(word == w for word in words):
            score += 2
            break
    
    for word in words:
        for prefix in TL_PREFIXES:
            if word.startswith(prefix) and len(word) > len(prefix) + 1:
                score += 1
                break
        
        for suffix in TL_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                score += 1
                break
        
        if word.endswith(('ad', 'id', 'od', 'ud', 'ag', 'ig', 'og', 'ug')):
            score += 0.5
    
    english_indicators = 0
    common_english = {'the', 'and', 'to', 'of', 'is', 'are', 'you', 'your', 'this', 'that'}
    for word in words:
        if word in common_english:
            english_indicators += 1
    
    if score >= 1 and english_indicators <= len(words) * 0.3:
        return True
    
    tl_word_count = sum(1 for word in words if any(word.startswith(p) for p in TL_PREFIXES) or 
                       any(word.endswith(s) for s in TL_SUFFIXES) or word in COMMON_TL_FUNCTION_WORDS)
    
    if tl_word_count >= max(2, len(words) * 0.4):
        return True
    
    if english_indicators == 0 and len(words) > 0:
        return True
    
    return False

def try_lemma(word: str, vocab: Set[str]) -> str:
    """Unified lemma detection for Tagalog morphology"""
    w = word.lower().strip()
    
    if len(w) < 2:
        return w
    
    # 0) Function words and exact matches stay as-is
    if w in COMMON_TL_FUNCTION_WORDS or w in vocab:
        return w
    
    # 1) Strip prefixes (longest first)
    for prefix in LEMMA_PREFIXES:
        if w.startswith(prefix):
            base = w[len(prefix):]
            if base and base in vocab:
                return base
    
    # 2) Remove infixes after first vowel
    w_inf_removed = w
    first_vowel = re.search(r"[aeiou]", w)
    if first_vowel:
        pos = first_vowel.start() + 1
        for infix in LEMMA_INFIXES:
            if pos + len(infix) <= len(w) and w[pos:pos+len(infix)] == infix:
                base = w[:pos] + w[pos+len(infix):]
                if base in vocab:
                    return base
                w_inf_removed = base
                break
    
    # 3) Strip suffixes
    def strip_suffixes(x: str) -> Optional[str]:
        for suffix in LEMMA_SUFFIXES:
            if x.endswith(suffix) and len(x) > len(suffix) + 1:
                base = x[:-len(suffix)]
                if base in vocab:
                    return base
        return None
    
    # Try both original and infix-removed versions
    for candidate in (w, w_inf_removed):
        result = strip_suffixes(candidate)
        if result:
            return result
    
    # 4) Handle reduplication
    for candidate in (w, w_inf_removed):
        match = re.match(rf"^({CV_PATTERN})\1(.+)$", candidate)
        if match:
            base = match.group(1) + match.group(2)
            if base in vocab:
                return base
    
    # 5) Combined prefix + reduplication/suffix
    candidate = w_inf_removed
    for prefix in LEMMA_PREFIXES:
        if candidate.startswith(prefix):
            remaining = candidate[len(prefix):]
            # Try reduplication
            match = re.match(rf"^({CV_PATTERN})\1(.+)$", remaining)
            if match:
                base = match.group(1) + match.group(2)
                if base in vocab:
                    return base
            # Try suffixes
            result = strip_suffixes(remaining)
            if result:
                return result
    
    # 6) Return original if nothing works
    return w

def ensure_target_present(sentence_tl: str, target: Optional[str], vocab: Set[str]) -> bool:
    if not target:
        return True
    
    toks = tokenize_simple(sentence_tl)
    tbase = target.lower()
    
    for tok in toks:
        if try_lemma(tok, vocab) == tbase:
            return True
    
    return False

# --------------------------- FIXED Wiktionary Implementation ---------------------------

def fetch_wiktionary_tl_direct(word: str) -> Dict[str, Any]:
    """
    Direct Wiktionary API implementation that actually works
    Uses the MediaWiki API to get page content and parses it directly
    """
    WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"
    
    time.sleep(0.1)
    
    # Clean the word for lookup
    word_clean = word.strip().lower()
    
    # Define a proper User-Agent header
    headers = {
        'User-Agent': 'TagalogCardGenerator/1.0 (Tagalog language learning project; gaspar@ephs.aoyama.ac.jp)'
    }
    
    try:
        # First, get the page content
        params = {
            'action': 'parse',
            'page': word_clean,
            'prop': 'wikitext',
            'format': 'json'
        }
        
        # Pass the headers to the request
        response = requests.get(WIKTIONARY_API, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            return {
                "ipa": None,
                "ipa_source": "none", 
                "etymology": [],
                "etymology_source": "none",
                "definitions": [],
                "word_found": False,
                "lookup_details": f"word_not_found: {data['error'].get('info', 'unknown_error')}"
            }
        
        wikitext = data['parse']['wikitext']['*']
        
        # Look for Tagalog section
        tagalog_section = extract_tagalog_section(wikitext)
        if not tagalog_section:
            return {
                "ipa": None,
                "ipa_source": "none",
                "etymology": [],
                "etymology_source": "none", 
                "definitions": [],
                "word_found": False,
                "lookup_details": "no_tagalog_section_found"
            }
        
        # Extract data from Tagalog section
        ipa = extract_ipa(tagalog_section)
        etymology = extract_etymology(tagalog_section)
        definitions = extract_definitions(tagalog_section)
        
        lookup_details = f"success: ipa_found={ipa is not None}, etymology_found={len(etymology) > 0}, definitions_found={len(definitions)}"
        
        return {
            "ipa": ipa,
            "ipa_source": "wiktionary" if ipa else "none",
            "etymology": etymology,
            "etymology_source": "wiktionary" if etymology else "none",
            "definitions": definitions,
            "word_found": True,
            "lookup_details": lookup_details
        }
        
    except Exception as e:
        log.warning(f"Direct Wiktionary fetch failed for '{word}': {e}")
        return {
            "ipa": None,
            "ipa_source": "none",
            "etymology": [],
            "etymology_source": "none",
            "definitions": [],
            "word_found": False,
            "lookup_details": f"api_error: {str(e)}"
        }

def extract_tagalog_section(wikitext: str) -> Optional[str]:
    """Extract the Tagalog language section from wikitext"""
    # Look for ==Tagalog== or ===Tagalog=== section
    tagalog_patterns = [
        r'==\s*Tagalog\s*==(.*?)(?=\n==[^=]|$)',
        r'===\s*Tagalog\s*===(.*?)(?=\n==[^=]|$)',
        r'==\s*Filipino\s*==(.*?)(?=\n==[^=]|$)'
    ]
    
    for pattern in tagalog_patterns:
        match = re.search(pattern, wikitext, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def extract_ipa(section_text: str) -> Optional[str]:
    """Extract IPA pronunciation from section"""
    try:
        # log.warning("IPA extraction - section: %.100s", section_text)
        
        # Try tl-pr template first (most common)
        tl_pr_match = re.search(r'\{\{tl-pr\|([^}]+)\}\}', section_text)
        if tl_pr_match:
            ipa = tl_pr_match.group(1).split('|')[0].strip()
            if ipa and len(ipa) > 1:
                # log.warning("IPA extraction - found: %s", ipa)
                return f"/{ipa}/"
        
        # Fallback to other IPA patterns
        ipa_patterns = [
            r'\{\{tl-ipa\|([^}]+)\}\}',
            r'\{\{ipa\|tl\|([^}]+)\}\}',
        ]
        
        for pattern in ipa_patterns:
            match = re.search(pattern, section_text)
            if match:
                ipa = match.group(1).split('|')[0].strip()
                if ipa and len(ipa) > 1:
                    log.warning("IPA extraction - found: %s", ipa)
                    return f"/{ipa}/"
        
        log.warning("IPA extraction - no IPA found")
        return None
        
    except Exception as e:
        log.warning("IPA extraction failed: %s", e)
        return None

def extract_etymology(section_text: str) -> List[str]:
    """Extract etymology from section"""
    try:
        etymologies = []
        etym_patterns = [
            r'===Etymology\s+\d+===(.*?)(?=\n===|\n==|$)',
            r'===Etymology===(.*?)(?=\n===|\n==|$)'
        ]
        
        for pattern in etym_patterns:
            matches = re.findall(pattern, section_text, re.DOTALL)
            for match in matches:
                cleaned = re.sub(r'\{\{([^}]*)\}\}', lambda m: ' '.join(m.group(1).split('|')[1:]), match)
                cleaned = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', cleaned)
                cleaned = re.sub(r'\[\[([^\]]+)\]\]', r'\1', cleaned)
                cleaned = re.sub(r'<[^>]+>', '', cleaned)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                
                if len(cleaned) > 20:
                    etymologies.append(cleaned)
        
        return etymologies
        
    except Exception as e:
        log.warning("Etymology extraction failed: %s", e)
        return []



def extract_definitions(section_text: str) -> List[str]:
    """Extract definitions from section"""
    definitions = []
    
    # Remove templates first (multi-line safe)
    cleaned_section = re.sub(r'\{\{.*?\}\}', '', section_text, flags=re.DOTALL)
    
    # Look for numbered definitions at line starts - stop at any heading level
    def_matches = re.findall(r'^\s*#\s*(.*?)(?=^\s*#|^\s*={2,}|$)', 
                           cleaned_section, flags=re.MULTILINE | re.DOTALL)
    
    for match in def_matches:
        # Clean the definition text
        cleaned = match.strip()
        
        # Handle [[links]] and [[link|text]] formats properly
        cleaned = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', cleaned)  # [[link|text]] → text
        cleaned = re.sub(r'\[\[([^\]]+)\]\]', r'\1', cleaned)             # [[link]] → link
        
        # Remove any remaining HTML tags and normalize whitespace
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Only keep substantial definitions
        if len(cleaned) > 5 and not cleaned.startswith(':'):
            definitions.append(cleaned)
    
    return definitions[:3]


def clean_wikitext(text: str) -> str:
    """Clean wikitext by removing templates and formatting"""
    # Remove templates
    text = re.sub(r'\{\{.*?\}\}', '', text)
    # Remove links but keep text
    text = re.sub(r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Unescape HTML entities
    text = html.unescape(text)
    
    return text.strip()

# --------------------------- OpenAI client wrapper ---------------------------
# Your existing OpenAI client remains the same...
class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self.legacy = False
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.legacy = False
        except Exception:
            import openai
            openai.api_key = api_key
            self.client = openai
            self.legacy = True
        
        self.supports_json_object = not (str(self.model).startswith("o") or str(self.model).startswith("gpt-4.1"))
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    def _add_usage(self, usage_obj: Any) -> None:
        if not usage_obj:
            return
        
        try:
            if hasattr(usage_obj, 'prompt_tokens'):
                pt = usage_obj.prompt_tokens
                ct = usage_obj.completion_tokens
                tt = usage_obj.total_tokens
            elif isinstance(usage_obj, dict):
                pt = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens")
                ct = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens")
                tt = usage_obj.get("total_tokens")
            else:
                return
            
            if pt is not None:
                self.usage["prompt_tokens"] += int(pt)
            if ct is not None:
                self.usage["completion_tokens"] += int(ct)
            if tt is not None:
                self.usage["total_tokens"] += int(tt)
                
        except Exception:
            pass

    def _call(self, messages: List[Dict[str, str]], schema: Optional[Dict[str, Any]] = None, timeout: int = 60) -> str:
        if not self.legacy:
            try:
                kwargs = {"response_format": {"type": "json_object"}} if (schema and self.supports_json_object) else {}
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.4,
                    **kwargs,
                )
                self.usage["calls"] += 1
                self._add_usage(resp.usage)
                return resp.choices[0].message.content or ""
            except Exception:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.5,
                )
                self.usage["calls"] += 1
                self._add_usage(resp.usage)
                return resp.choices[0].message.content or ""
        else:
            resp = self.client.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
            )
            self.usage["calls"] += 1
            self._add_usage(resp.get("usage"))
            return resp["choices"][0]["message"]["content"]

# --------------------------- Enhanced LLM prompts ---------------------------
# Your existing prompts remain the same...
GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "sentence_tl": {"type": "string"},
        "sentence_en": {"type": "string"},
        "explanation_en": {"type": "string"}
    },
    "required": ["sentence_tl", "sentence_en"],
    "additionalProperties": False,
}

def sys_prompt_generate(stage: str) -> str:
    base_json_instr = (
        "Return ONLY a JSON object with keys 'sentence_tl', 'sentence_en', 'explanation_en': "
        "'sentence_tl' contains the Tagalog sentence, "
        "'sentence_en' is a faithful translation, and 'explanation_en' concisely explains any relevant grammar or nuance.\n"
        "No extra keys, no markdown, no code fences, no comments.\n"
        "Uncertainty policy: If you are not fully confident about any detail, explicitly write "
        "\"Uncertain: <short note>\" in 'explanation_en'.\n"
        "Ground in the inputs and standard, widely taught Tagalog usage.\n"
        "All text must be valid UTF-8 and valid JSON.\n"
        "Example:\n"
        "{\n"
        '  "sentence_tl": "Magandang umaga.",\n'
        '  "sentence_en": "Good morning.",\n'
        '  "explanation_en": "Common polite greeting; neutral formality."\n'
        "}"
    )

    shared_rules = (
        "Quality rules:\n"
        "- Use only authentic Tagalog grammar and common vocabulary.\n"
        "- Prefer high-frequency, idiomatic forms; avoid rare/archaic forms.\n"
        "- Never fabricate grammatical rules, history, or cultural claims.\n"
        "- The provided target MUST appear (base or inflected) in 'sentence_tl'.\n"
    )

    if stage == "greeting":
        return (
            "ROLE: Generate a natural, high-frequency Tagalog greeting.\n"
            "Constraints: exactly one Tagalog sentence (1–5 words; only 1 word is okay too).\n"
            + shared_rules
            + base_json_instr
        )

    if stage == "expression":
        return (
            "ROLE: Generate a natural, high-frequency Tagalog expression.\n"
            "Constraints: exactly one short Tagalog sentence (preferably 2-4 words, max about 7 words).\n"
            + shared_rules
            + base_json_instr
        )

    # grammar stage
    return (
        "ROLE: Generate one Tagalog sentence that clearly exemplifies a single grammar point.\n"
        "Inputs: 'grammar_title', 'note', and 'drill_target'.\n"
        "Constraints: one short Tagalog sentence (preferably 2-5 words, max about 10 words), include 'drill_target' (base or inflected).\n"
        + shared_rules
        + base_json_instr
    )

def sys_prompt_breakdown() -> str:
    return (
        "You are a Tagalog linguistics expert. You are given: (a) a Tagalog sentence, "
        "(b) its English translation, and (c) per-token Wiktionary lookup data.\n\n"
        "GROUNDING:\n"
        "- Use the provided Wiktionary data as primary source, but apply common sense for basic Tagalog words.\n"
        "- For very common words (like 'ay', 'ako', basic verbs), provide reasonable information even if Wiktionary data is incomplete or unclear.\n"
        "- Ignore irrelevant or incorrect definitions.\n"
        "- If something is unclear or missing, state it explicitly as 'Uncertain: <short note>'.\n"
        "- When Wiktionary lists multiple meanings or etymologies, choose the one that makes sense in the sentence context.\n\n"
        "FORMAT (STRICT JSON ONLY):\n"
        "- Return a JSON object with exactly these keys:\n"
        "  'breakdown'  : array of strings formatted 'word — \"meaning(s)\" (notes, brief explanations, including etymology, if known)'\n"
        "  'ipa'        : IPA for the entire sentence. Provide best-effort IPA even if uncertain.\n"
        "  'wiktionary_summary': brief stats on lookup coverage (e.g., 'entries found: 5/7; lemmas normalized').\n"
        "No extra keys, no markdown, no code fences.\n\n"
        "CONTENT GUIDELINES:\n"
        "- Include multiple senses or nuances if attested in the provided data or otherwise certain.\n"
        "- Note morphology (affixes, reduplication) and salient grammar.\n"
    )

BREAKDOWN_SCHEMA = {
    "type": "object",
    "properties": {
        "breakdown": {
            "type": "array",
            "items": {"type": "string"}
        },
        "ipa": {"type": "string"},
        "wiktionary_summary": {"type": "string"}
    },
    "required": ["breakdown"],
    "additionalProperties": False,
}

# --------------------------- validators ---------------------------
def length_ok(stage: str, n_tokens: int) -> bool:
    if stage == "greeting":
        return 1 <= n_tokens <= 7
    if stage == "expression":
        return 2 <= n_tokens <= 7
    if stage == "grammar":
        return 2 <= n_tokens <= 12
    return True

# --------------------------- Enhanced main builder ---------------------------
class Builder:
    def __init__(self, client: OpenAIClient, word_buckets: Dict[str, List[str]], grammar_points: List[Dict[str, Any]]):
        self.client = client
        
        if TEST_MODE:
            log.info("TEST_MODE enabled — generating a tiny sample.")
            wb = dict(word_buckets)
            wb["greetings"] = wb.get("greetings", [])[:TEST_LIMITS["greetings_items"]]
            wb["expressions"] = wb.get("expressions", [])[:TEST_LIMITS["expressions_items"]]
            wb["grammar_drills"] = wb.get("grammar_drills", [])[:TEST_LIMITS["grammar_drills_items"]]
            gps = grammar_points[:TEST_LIMITS["grammar_points"]]
            self.word_buckets = wb
            self.grammar_points = gps
            self.greet_per_word = TEST_LIMITS["greet_per_word"]
            self.expr_per_word = TEST_LIMITS["expr_per_word"]
            self.grammar_examples_per_point = TEST_LIMITS["grammar_examples_per_point"]
        else:
            self.word_buckets = word_buckets
            self.grammar_points = grammar_points
            self.greet_per_word = 2
            self.expr_per_word = 2
            self.grammar_examples_per_point = 2
        
        all_vocab = (
            self.word_buckets.get("greetings", []) + 
            self.word_buckets.get("expressions", []) + 
            self.word_buckets.get("grammar_drills", [])
        )
        self.vocab_set = {v.lower() for v in all_vocab}
        
        total_greetings = len(self.word_buckets.get("greetings", [])) * self.greet_per_word
        total_expressions = len(self.word_buckets.get("expressions", [])) * self.expr_per_word
        total_grammar = len(self.grammar_points) * self.grammar_examples_per_point
        self.expected_total = total_greetings + total_expressions + total_grammar
        
        self.card_counter = 0 
        
        log.info("Expected cards: %d greetings + %d expressions + %d grammar = %d total", 
                total_greetings, total_expressions, total_grammar, self.expected_total)

    def gen_one(self, stage: str, payload: Dict[str, Any], progress: ProgressTracker) -> Optional[Dict[str, str]]:
        user_parts = {"stage": stage}
        user_parts.update(payload)
        
        messages = [
            {"role": "system", "content": sys_prompt_generate(stage)},
            {"role": "user", "content": json.dumps(user_parts, ensure_ascii=False)},
        ]
        
        for attempt in range(3):
            try:
                text = self.client._call(messages, schema=GEN_SCHEMA)
            except Exception as e:
                es = str(e)
                status = getattr(e, "status_code", None)
                
                if "insufficient_quota" in es or "You exceeded your current quota" in es:
                    log.error("OpenAI API error: insufficient quota.")
                    raise SystemExit("OpenAI API insufficient quota.")
                
                if (status in (429, 500, 502, 503, 504)) or ("RateLimit" in type(e).__name__) or ("429" in es):
                    backoff = min(0.6 * (2 ** attempt), 8.0)
                    time.sleep(backoff)
                    continue
                raise
            
            fail_reason = ""
            try:
                data = json.loads(text)
                tl = (data.get("sentence_tl") or "").strip()
                en = (data.get("sentence_en") or "").strip()
                ge = (data.get("explanation_en") or "").strip()
            except Exception:
                tl, en = "", ""
                m = re.search(r'"sentence_tl"\s*:\s*"(.*?)"', text, re.S)
                if m:
                    tl = m.group(1)
                m = re.search(r'"sentence_en"\s*:\s*"(.*?)"', text, re.S)
                if m:
                    en = m.group(1)
            
            if not tl or not en:
                fail_reason = "missing sentence fields"
                continue
            
            n_tok = len(tokenize_simple(tl))
            if not looks_tl(tl):
                target = payload.get("target") or payload.get("drill_target")
                if target and ensure_target_present(tl, target, self.vocab_set):
                    log.warning("Accepted sentence despite weak Tagalog detection (contains target): %s", tl)
                else:
                    fail_reason = f"not detected as Tagalog: '{tl}'"
                    continue
            
            target = payload.get("target")
            if stage in {"greeting", "expression"}:
                if not ensure_target_present(tl, target, self.vocab_set):
                    fail_reason = f"target not present: {target}"
                    continue
            elif stage == "grammar":
                drill = payload.get("drill_target")
                if drill and not ensure_target_present(tl, drill, self.vocab_set):
                    fail_reason = f"drill not present: {drill}"
                    continue
            
            if not length_ok(stage, n_tok):
                fail_reason = f"length {n_tok} out of range for {stage}"
                continue
            
            progress.update()
            return {"sentence_tl": tl, "sentence_en": en, "explanation_en": ge}
        
        log.warning("Failed to generate %s. Last reason: %s", stage, fail_reason or "unknown")
        return None

    def generate_enhanced_breakdown(self, sentence_tl: str, sentence_en: str, progress: ProgressTracker) -> Dict[str, Any]:
        """Generate enhanced word breakdown with direct Wiktionary API"""
        tokens = tokenize_simple(sentence_tl)
        
        # Collect Wiktionary data for all tokens using direct API
        wiktionary_data = {}
        for token in tokens:
            lemma = try_lemma(token, self.vocab_set)
            wiktionary_data[token] = fetch_wiktionary_tl_direct(lemma)
        
        # Prepare detailed Wiktionary info for LLM
        wiktionary_info = {}
        for token, data in wiktionary_data.items():
            # Prefix etymologies with numbers for clarity
            prefixed_etymologies = []
            for i, etym in enumerate(data["etymology"], 1):
                prefixed_etymologies.append(f"Etymology {i}: {etym}")
            
            wiktionary_info[token] = {
                "found": data["word_found"],
                "definitions": data["definitions"][:3],
                "etymology": prefixed_etymologies,  # Now with clear numbering
                "ipa": data["ipa"],
                "lookup_details": data["lookup_details"]
            }
        
        # log.warning("Raw Wiktionary data for '%s': %s", sentence_tl, json.dumps(wiktionary_data, ensure_ascii=False, indent=2))

        user_payload = {
            "sentence_tl": sentence_tl,
            "sentence_en": sentence_en,
            "wiktionary_data": wiktionary_info
        }
        
        messages = [
            {"role": "system", "content": sys_prompt_breakdown()},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ]
        
        try:
            text = self.client._call(messages, schema=BREAKDOWN_SCHEMA)
            data = json.loads(text)
            breakdown = data.get("breakdown", [])
            ipa = data.get("ipa", "")
            
            found_count = sum(1 for data in wiktionary_data.values() if data["word_found"])
            total_count = len(wiktionary_data)
            actual_summary = f"Wiktionary lookup: {found_count}/{total_count} words found"
            
            progress.update()
            return {
                "breakdown": breakdown,
                "ipa": ipa,
                "wiktionary_summary": actual_summary,
                # Minimal trace - just success stats
                "wiktionary_trace": {
                    "total_words": total_count,
                    "words_found": found_count,
                    "success_rate": f"{(found_count/total_count)*100:.1f}%"
                }
            }
        except Exception as e:
            log.warning("Failed to generate enhanced breakdown: %s", e)
            found_count = sum(1 for data in wiktionary_data.values() if data["word_found"])
            total_count = len(wiktionary_data)
            
            progress.update()
            return {
                "breakdown": [f"{token} — (analysis unavailable)" for token in tokens],
                "ipa": "",
                "wiktionary_summary": f"Wiktionary lookup: {found_count}/{total_count} words found",
                # Minimal trace - just success stats
                "wiktionary_trace": {
                    "total_words": total_count,
                    "words_found": found_count,
                    "success_rate": f"{(found_count/total_count)*100:.1f}%"
                }
            }

    def build_vocab_stage(self, stage: str, items: List[str], per_word: int, progress: ProgressTracker) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        
        for x in items:
            for _ in range(per_word):
                res = self.gen_one(stage, {"target": x}, progress)
                if not res:
                    continue
                
                breakdown_data = self.generate_enhanced_breakdown(res["sentence_tl"], res["sentence_en"], progress)
                
                card = {
                    "id": self.card_counter,
                    "sentence_tl": res["sentence_tl"],
                    "sentence_en": res["sentence_en"],
                    "stage": stage,
                    "target_word": x,
                    "grammar_id": None,
                    "grammar_title": None,
                    "explanation_en": res.get("explanation_en", ""),
                    "ipa": breakdown_data["ipa"],
                    "word_breakdown": breakdown_data["breakdown"],
                    "wiktionary_trace": breakdown_data["wiktionary_trace"]
                }
                self.card_counter += 1
                cards.append(card)
        
        return cards

    def build_greetings(self, progress: ProgressTracker) -> List[Dict[str, Any]]:
        return self.build_vocab_stage("greeting", self.word_buckets.get("greetings", []), self.greet_per_word, progress)

    def build_expressions(self, progress: ProgressTracker) -> List[Dict[str, Any]]:
        items = self.word_buckets.get("expressions", [])
        return self.build_vocab_stage("expression", items, self.expr_per_word, progress)


    def build_grammar(self, progress: ProgressTracker) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        drills = self.word_buckets.get("grammar_drills", [])
        
        if not drills:
            log.warning("No grammar_drills provided; grammar sentences may not meet pairing requirement.")
        
        if drills:
            # Track usage count for each drill to distribute evenly
            drills_usage_count = {drill: 0 for drill in drills}
            
            # Assign drills to grammar points
            for idx_gp, gp in enumerate(self.grammar_points):
                gid, gtitle, gnote = gp["id"], gp["title"], gp.get("note", "")
                
                for ex_i in range(self.grammar_examples_per_point):
                    # Always pick the least used drill (this ensures fair distribution)
                    drill = min(drills, key=lambda d: drills_usage_count[d])
                    drills_usage_count[drill] += 1
                    
                    payload = {"grammar_id": gid, "grammar_title": gtitle, "note": gnote, "drill_target": drill}
                    
                    res = self.gen_one("grammar", payload, progress)
                    if not res:
                        continue
                    
                    breakdown_data = self.generate_enhanced_breakdown(res["sentence_tl"], res["sentence_en"], progress)
                    
                    card = {
                        "id": self.card_counter,
                        "sentence_tl": res["sentence_tl"],
                        "sentence_en": res["sentence_en"],
                        "stage": "grammar",
                        "target_word": drill,
                        "grammar_id": gid,
                        "grammar_title": gtitle,
                        "explanation_en": res.get("explanation_en", ""),
                        "ipa": breakdown_data["ipa"],
                        "word_breakdown": breakdown_data["breakdown"],
                                "wiktionary_trace": breakdown_data["wiktionary_trace"]
                    }
                    self.card_counter += 1
                    cards.append(card)
            log.info("Grammar drill usage distribution: %s", dict(drills_usage_count))

        else:
            # Fallback: no drills available
            for idx_gp, gp in enumerate(self.grammar_points):
                gid, gtitle, gnote = gp["id"], gp["title"], gp.get("note", "")
                
                for ex_i in range(self.grammar_examples_per_point):
                    payload = {"grammar_id": gid, "grammar_title": gtitle, "note": gnote}
                    
                    res = self.gen_one("grammar", payload, progress)
                    if not res:
                        continue
                    
                    breakdown_data = self.generate_enhanced_breakdown(res["sentence_tl"], res["sentence_en"], progress)
                    
                    card = {
                        "id": self.card_counter,
                        "sentence_tl": res["sentence_tl"],
                        "sentence_en": res["sentence_en"],
                        "stage": "grammar",
                        "target_word": None,
                        "grammar_id": gid,
                        "grammar_title": gtitle,
                        "explanation_en": res.get("explanation_en", ""),
                        "ipa": breakdown_data["ipa"],
                        "word_breakdown": breakdown_data["breakdown"],
                        "wiktionary_trace": breakdown_data["wiktionary_trace"]
                    }
                    self.card_counter += 1
                    cards.append(card)
        
        return cards



# --------------------------- main execution ---------------------------
def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def checkpoint_write(cards: List[Dict[str, Any]]) -> None:
    try:
        write_json(PARTIAL_JSON, cards)
        log.info("Checkpoint saved: %s (%d cards)", PARTIAL_JSON, len(cards))
    except Exception as e:
        log.warning("Failed to write checkpoint: %s", e)

def build_all_cards(client: OpenAIClient, word_buckets: Dict[str, List[str]], grammar_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    builder = Builder(client, word_buckets, grammar_points)
    all_cards: List[Dict[str, Any]] = []
    
    progress = ProgressTracker(builder.expected_total)

    log.info("Generating greeting cards…")
    for card in builder.build_greetings(progress):
        all_cards.append(card)
        if len(all_cards) % CHECKPOINT_EVERY == 0:
            checkpoint_write(all_cards)

    log.info("Generating expression cards…")
    for card in builder.build_expressions(progress):
        all_cards.append(card)
        if len(all_cards) % CHECKPOINT_EVERY == 0:
            checkpoint_write(all_cards)

    log.info("Generating grammar cards…")
    for card in builder.build_grammar(progress):
        all_cards.append(card)
        if len(all_cards) % CHECKPOINT_EVERY == 0:
            checkpoint_write(all_cards)

    progress.finish()
    
    gid_order = {gp["id"]: idx for idx, gp in enumerate(builder.grammar_points)}
    
    def order_key(card: Dict[str, Any]) -> Tuple[int, int]:
        stage_rank = {"greeting": 0, "expression": 1, "grammar": 2}.get(card.get("stage") or "other", 3)
        if stage_rank == 2:
            return (stage_rank, gid_order.get(card.get("grammar_id"), 10**9))
        return (stage_rank, 0)

    all_cards = sorted(all_cards, key=order_key)
    return all_cards

def main():
    log.info("Tagalog Card Generator starting at %s", TIMESTAMP)
    log.info("Loading configuration…")
    config = load_config()
    api_key = config.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No OPENAI_API_KEY found in config.json or environment")
    
    log.info("Loading word buckets…")
    word_buckets = load_word_buckets()
    log.info("Loading grammar points…")
    grammar_points = load_grammar_points()
    
    log.info("Initializing OpenAI client…")
    client = OpenAIClient(api_key=api_key, model=MODEL_NAME)
    
    log.info("Starting card generation…")
    all_cards = build_all_cards(client, word_buckets, grammar_points)
    
    log.info("Writing final output…")
    write_json(OUTPUT_JSON, all_cards)
    write_json(USAGE_JSON, client.usage)
    
    total_words = sum(card.get('wiktionary_trace', {}).get('total_words', 0) for card in all_cards)
    found_words = sum(card.get('wiktionary_trace', {}).get('words_found', 0) for card in all_cards)
    
    log.info("Wiktionary Statistics: %d/%d words found (%.1f%%)", 
             found_words, total_words, (found_words/total_words)*100 if total_words > 0 else 0)
    
    try:
        if PARTIAL_JSON.exists():
            PARTIAL_JSON.unlink()
    except Exception:
        pass
    
    log.info("Successfully generated %d cards", len(all_cards))
    log.info("Output: %s", OUTPUT_JSON)
    log.info("Usage: %d total tokens", client.usage.get("total_tokens", 0))
    
    if TEST_MODE:
        log.info("TEST_MODE completed successfully")

if __name__ == "__main__":
    main()