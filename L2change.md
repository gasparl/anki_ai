You are a careful code editor.

Goal:
Update the script {{SCRIPT_NAME}} from {{SOURCE_LANGUAGE}} to {{TARGET_LANGUAGE}}.

Input:
- Full original script (pasted below).

Tasks:
1) Identify all language-specific constants/resources (e.g., locale codes, voices, tokenizers, IPA/transliteration, stopwords, example text).
2) Replace them for {{TARGET_LANGUAGE}} while preserving I/O, file structure, function names, and CLI args.
3) Keep behavior identical except where language adaptation is required.

Requirements:
- UTF-8 only. No breaking changes to JSON/CSV schemas or field names.
- Keep comments/docstrings concise and correct for {{TARGET_LANGUAGE}}.
- If a resource is unknown, add a clearly marked TODO placeholder with a sensible default.

Output:
- Return ONLY the complete updated script (no prose).
- At the top of the file, include a brief commented “CHANGELOG” (≤5 lines) summarizing what changed.
