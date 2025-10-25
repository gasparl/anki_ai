import json, subprocess, pathlib, re, time, sys, os, random

# ---------- CONFIG ----------
BAL4WEB = r"C:\Program Files (x86)\Balabolka\bal4web\bal4web.exe"
JSON_NAME = "tl_cards.json"   # expected JSON filename next to this script
SAMPLE_RATE_DEFAULT = "24"     # preferred sample rate; probe may try 48 too
SPEED = "1.0"                  # 0.10..3.00
PITCH = "0"                    # -10..+10
SERVICE = ["-s", "microsoft"]
LANG_CODE = "fil-PH"           # keep to avoid name collisions

VOICE_CANDIDATES = {
    "Blessica": ["Blessica", "fil-PH Blessica", "fil-PH-BlessicaNeural"],
    "Angelo":   ["Angelo",   "fil-PH Angelo",   "fil-PH-AngeloNeural"],
}

# ---------- RETRY / PROBE TUNABLES ----------
PROBE_COMBOS = [(True,"24"), (False,"24"), (True,"48"), (False,"48")]
PROBE_ATTEMPTS_PER_COMBO = 3
SYNTH_RETRIES = 4
BACKOFF_BASE = 0.8
BACKOFF_MULT = 1.7
JITTER = 0.25

# ---------- PATHS ----------
try:
    BASE_DIR = pathlib.Path(__file__).resolve().parent
except NameError:
    BASE_DIR = pathlib.Path.cwd()

READINGS_DIR = BASE_DIR / "readings"
JSON_PATH = BASE_DIR / "json" / "tl_cards.json"

# ---------- UTILS ----------
def safe_name(s, maxlen=120):
    s = re.sub(r'[\\/:*?"<>|]+', "_", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return (s or "untitled")[:maxlen]

def _run(text, outpath, voice_name, use_lang=True, fr=SAMPLE_RATE_DEFAULT):
    args = [BAL4WEB, *SERVICE]
    if use_lang:
        args += ["-l", LANG_CODE]
    args += ["-n", voice_name, "-r", SPEED, "-p", PITCH, "-fr", fr,
             "-enc", "utf8", "-t", text, "-w", str(outpath)]
    try:
        subprocess.run(args, check=True, capture_output=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode("utf-8", "ignore")
        return False, msg

def synth_with_retries(text, outpath, voice_params):
    name, use_lang, fr = voice_params["name"], voice_params["use_lang"], voice_params["fr"]
    delay = BACKOFF_BASE
    for attempt in range(1, SYNTH_RETRIES + 1):
        ok, err = _run(text, outpath, name, use_lang=use_lang, fr=fr)
        if ok:
            return True, ""
        if attempt < SYNTH_RETRIES:
            time.sleep(delay + random.uniform(-JITTER, JITTER))
            delay *= BACKOFF_MULT
    return False, err

def list_microsoft_voices():
    try:
        out = subprocess.run([BAL4WEB, "-s", "microsoft", "-m"],
                             check=True, capture_output=True)
        return (out.stdout or b"").decode("utf-8", "ignore")
    except Exception:
        return ""

def probe_voice(display_label, candidates):
    """
    Return dict {'name': ..., 'use_lang': bool, 'fr': '24'|'48'} or None.
    Ensures we never bind Blessica->Angelo or Angelo->Blessica.
    """
    READINGS_DIR.mkdir(parents=True, exist_ok=True)
    probe = READINGS_DIR / f"_probe_{display_label}.wav"

    # Only augment with THIS label (prevents cross-contamination).
    live = list_microsoft_voices()
    if live and display_label not in candidates and display_label.lower() in live.lower():
        candidates = [display_label] + list(candidates)

    # Try candidates across combos with retries
    for name in candidates:
        for use_lang, fr in PROBE_COMBOS:
            delay = BACKOFF_BASE
            for attempt in range(1, PROBE_ATTEMPTS_PER_COMBO + 1):
                ok, _ = _run("Test", probe, name, use_lang=use_lang, fr=fr)
                if ok:
                    # sanity guard: if the returned token clearly belongs to the other voice, skip
                    other = "Angelo" if display_label == "Blessica" else "Blessica"
                    if other.lower() in name.lower() and display_label.lower() not in name.lower():
                        # treat as mismatch; continue probing
                        break
                    try: probe.unlink(missing_ok=True)
                    except Exception: pass
                    return {"name": name, "use_lang": use_lang, "fr": fr}
                if attempt < PROBE_ATTEMPTS_PER_COMBO:
                    time.sleep(delay + random.uniform(-JITTER, JITTER))
                    delay *= BACKOFF_MULT
    try: probe.unlink(missing_ok=True)
    except Exception: pass
    return None

def load_sentences():
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, encoding="utf-8") as f:
                data = json.load(f)
            
            pairs = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "sentence_tl" in item:
                        # Use the id as the filename prefix, and sentence_tl as the text
                        item_id = item.get("id", "unknown")
                        text = str(item["sentence_tl"]).strip()
                        if text:
                            pairs.append((str(item_id), text))
            
            if pairs:
                return pairs
        except Exception as e:
            print(f"[WARN] Failed to read {JSON_PATH.name}: {e}. Falling back to dummy data.")
    
    # Fallback data if JSON loading fails
    return [
        ("0", "Salamat sa iyong tulong!"),
        ("1", "Magandang araw po!"),
    ]

# ---------- MAIN ----------
def main():
    if not pathlib.Path(BAL4WEB).exists():
        print(f"[ERROR] bal4web.exe not found at:\n{BAL4WEB}")
        print('Fix the path above or run: bal4web.exe -s microsoft -m   to verify installation.')
        sys.exit(1)

    READINGS_DIR.mkdir(parents=True, exist_ok=True)

    blessica_params = probe_voice("Blessica", VOICE_CANDIDATES["Blessica"][:])
    angelo_params   = probe_voice("Angelo",   VOICE_CANDIDATES["Angelo"][:])

    print("Detected voices:")
    print(f"  Blessica -> {blessica_params['name'] if blessica_params else 'NOT AVAILABLE (will skip)'}")
    print(f"  Angelo   -> {angelo_params['name']   if angelo_params   else 'NOT AVAILABLE (will skip)'}")

    if not blessica_params and not angelo_params:
        print('[ERROR] Neither Tagalog voice accepted by bal4web in this mode.')
        print(f'List available voices with:\n"{BAL4WEB}" -s microsoft -m')
        sys.exit(2)

    pairs = load_sentences()
    print(f"Processing {len(pairs)} sentences...")

    failures = 0
    for sid, text in pairs:
        for label, params in (("Blessica", blessica_params), ("Angelo", angelo_params)):
            if not params:
                continue
            # Format: {id}_{voice_name}.wav (e.g., "0_Blessica.wav", "1_Angelo.wav")
            out_wav = READINGS_DIR / f"{sid}_{label}.wav"
            ok, err = synth_with_retries(text, out_wav, params)
            if not ok:
                failures += 1
                print(f"[WARN] {sid} ({label}) failed.\n{err}\n")

    print("Done." + (f" {failures} failures." if failures else ""))

if __name__ == "__main__":
    main()