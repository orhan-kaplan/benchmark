#!/usr/bin/env python3
"""
Complete scoring of all 432 responses.
Uses full response text (including thinking) for truncated responses.
Outputs final table.
"""
import json, re

with open("_all_responses.json", encoding="utf-8") as f:
    all_responses = json.load(f)

models = sorted(all_responses.keys())

def clean(text):
    return text.replace("Ġ", " ").replace("Ċ", "\n")

def get_answer(text):
    c = clean(text)
    idx = c.find("</think>")
    if idx >= 0:
        return c[idx+8:].strip()
    return c.strip()

def get_full(text):
    """Full text including thinking - for checking truncated responses"""
    return clean(text)

def has_in_full(text, *terms):
    """Check if any term appears in full response (including thinking)"""
    full = get_full(text).lower()
    return any(t.lower() in full for t in terms)

def word_count(text):
    return len(text.split())

# ============================================================
# SCORING ALL CATEGORIES
# ============================================================

scores = {m: {"coding": [], "creative": [], "general": [], 
              "reasoning": [], "translation": []} for m in models}

# ============================================================
# REASONING (12 prompts)
# ============================================================
# reason-001: $129.60
for m in models:
    full = get_full(all_responses[m]["reason-001"])
    ans = get_answer(all_responses[m]["reason-001"])
    correct = "129.60" in full or "129.6" in full
    has_steps = "120" in full and ("9.60" in full or "9.6" in full)
    s = 0
    if correct: s += 6
    if has_steps: s += 2
    s += 2  # all show work
    scores[m]["reasoning"].append(min(s, 10))

# reason-002: 40 meters
for m in models:
    full = get_full(all_responses[m]["reason-002"])
    correct = bool(re.search(r'\b40\b', full))
    geometric = "geometric" in full.lower() or "series" in full.lower()
    s = 0
    if correct: s += 5
    if geometric: s += 3
    s += 2
    scores[m]["reasoning"].append(min(s, 10))

# reason-003: ~70.6%
for m in models:
    full = get_full(all_responses[m]["reason-003"])
    correct = bool(re.search(r'70\.6|0\.706|70\.63', full))
    complement = "1 -" in full or "complement" in full.lower()
    s = 0
    if correct: s += 4
    if complement: s += 3
    if "365" in full: s += 2
    s += 1
    scores[m]["reasoning"].append(min(s, 10))

# reason-004: 8 arrangements
for m in models:
    full = get_full(all_responses[m]["reason-004"])
    ans = get_answer(all_responses[m]["reason-004"])
    has_8 = bool(re.search(r'\b8\b', full))
    # Count listed arrangements
    arr_lines = len(re.findall(r'(?:Eve|Alice|Bob|Carol|Dave).*-.*(?:Eve|Alice|Bob|Carol|Dave)', full))
    s = 0
    if has_8: s += 3
    if arr_lines >= 8: s += 4
    elif arr_lines >= 4: s += 2
    s += 2  # systematic approach
    scores[m]["reasoning"].append(min(s, 10))

# reason-005: 1 draw
for m in models:
    full = get_full(all_responses[m]["reason-005"])
    correct = bool(re.search(r'\b1\b.*draw|one.*draw|single.*draw', full.lower()))
    mixed_key = "mixed" in full.lower() and "draw" in full.lower()
    s = 0
    if correct: s += 4
    if mixed_key: s += 3
    s += 3  # deduction chain
    scores[m]["reasoning"].append(min(s, 10))

# reason-006: pirate (98, 0, 1, 0, 1)
for m in models:
    full = get_full(all_responses[m]["reason-006"])
    has_98 = "98" in full
    backward = "backward" in full.lower() or "work back" in full.lower() or "reverse" in full.lower()
    s = 0
    if has_98: s += 4
    if backward: s += 3
    s += 3  # reasoning shown
    scores[m]["reasoning"].append(min(s, 10))

# reason-007: micrometers
for m in models:
    full = get_full(all_responses[m]["reason-007"])
    micro = any(t in full.lower() for t in ["micrometer", "micron", "μm"])
    tiny = any(t in full.lower() for t in ["dust", "grain", "pinhead", "blood cell", "speck"])
    nucleus = "nucleus" in full.lower() or "nuclei" in full.lower()
    s = 0
    if micro or tiny: s += 4
    if nucleus: s += 3
    if "empty" in full.lower(): s += 2
    s += 1
    scores[m]["reasoning"].append(min(s, 10))

# reason-008: Rayleigh scattering
for m in models:
    full = get_full(all_responses[m]["reason-008"])
    rayleigh = "rayleigh" in full.lower()
    path = "path" in full.lower() and ("long" in full.lower() or "short" in full.lower())
    s = 0
    if rayleigh: s += 4
    if path: s += 3
    s += 3
    scores[m]["reasoning"].append(min(s, 10))

# reason-009: $113.85
for m in models:
    full = get_full(all_responses[m]["reason-009"])
    correct = "113.85" in full
    round_trip = any(t in full.lower() for t in ["round trip", "× 2", "×2", "times 2", "* 2", "both ways"])
    s = 0
    if correct: s += 4
    if "150" in full and "350" in full: s += 2
    if "15%" in full or "1.15" in full: s += 2
    if round_trip or "75.9" in full: s += 2
    scores[m]["reasoning"].append(min(s, 10))

# reason-010: 12-coin problem
for m in models:
    full = get_full(all_responses[m]["reason-010"])
    has_4v4 = "4 vs 4" in full or "4 against 4" in full.lower()
    heavier_lighter = "heavier" in full.lower() and "lighter" in full.lower()
    cases = "case" in full.lower() or "balanced" in full.lower()
    s = 0
    if has_4v4: s += 3
    if heavier_lighter: s += 2
    if cases: s += 3
    s += 2
    scores[m]["reasoning"].append(min(s, 10))

# reason-011: 99 survivors, parity
for m in models:
    full = get_full(all_responses[m]["reason-011"])
    has_99 = bool(re.search(r'\b99\b', full))
    parity = "parity" in full.lower() or ("odd" in full.lower() and "even" in full.lower())
    s = 0
    if has_99: s += 3
    if parity: s += 4
    s += 3
    scores[m]["reasoning"].append(min(s, 10))

# reason-012: snail day 28 and day 14
for m in models:
    full = get_full(all_responses[m]["reason-012"])
    has_28 = bool(re.search(r'\b28\b', full))
    has_14 = bool(re.search(r'\b14\b', full))
    s = 0
    if has_28: s += 4
    if has_14: s += 4
    s += 2
    scores[m]["reasoning"].append(min(s, 10))

# ============================================================
# GENERAL (12 prompts)
# ============================================================
# general-001: fission vs fusion
for m in models:
    full = get_full(all_responses[m]["general-001"])
    fission = "fission" in full.lower() and "split" in full.lower()
    fusion = "fusion" in full.lower() and ("combin" in full.lower() or "hydrogen" in full.lower())
    sun_fusion = "sun" in full.lower() and "fusion" in full.lower()
    challenge = "temperature" in full.lower() or "confine" in full.lower()
    s = 0
    if fission: s += 2
    if fusion: s += 2
    if sun_fusion: s += 2
    if challenge: s += 3
    s += 1
    scores[m]["general"].append(min(s, 10))

# general-002: transformers
for m in models:
    full = get_full(all_responses[m]["general-002"])
    self_attn = "self-attention" in full.lower() or "self attention" in full.lower()
    multi = "multi-head" in full.lower() or "multi head" in full.lower()
    pos = "positional" in full.lower()
    rnn = "rnn" in full.lower() or "recurrent" in full.lower()
    s = 0
    if self_attn: s += 3
    if multi: s += 2
    if pos: s += 2
    if rnn: s += 2
    s += 1
    scores[m]["general"].append(min(s, 10))

# general-003: Roman Empire
for m in models:
    full = get_full(all_responses[m]["general-003"])
    factors = 0
    if "military" in full.lower() or "barbarian" in full.lower(): factors += 1
    if "economic" in full.lower() or "inflation" in full.lower(): factors += 1
    if "political" in full.lower() or "civil war" in full.lower(): factors += 1
    if "corruption" in full.lower() or "admin" in full.lower(): factors += 1
    if "disease" in full.lower() or "plague" in full.lower(): factors += 1
    interaction = "led to" in full.lower() or "caused" in full.lower() or "result" in full.lower()
    s = min(factors, 4) + (3 if interaction else 1) + 2
    scores[m]["general"].append(min(s, 10))

# general-004: trolley problem
for m in models:
    full = get_full(all_responses[m]["general-004"])
    trolley = "trolley" in full.lower()
    util = "utilitarian" in full.lower()
    deont = "deontolog" in full.lower() or "kant" in full.lower()
    virtue = "virtue" in full.lower()
    variation = "fat man" in full.lower() or "surgeon" in full.lower() or "footbridge" in full.lower()
    s = 0
    if trolley: s += 2
    if variation: s += 2
    if util: s += 2
    if deont: s += 2
    if virtue: s += 2
    scores[m]["general"].append(min(s, 10))

# general-005: LLMs explained
for m in models:
    full = get_full(all_responses[m]["general-005"])
    concepts = sum(1 for t in ["tokeniz", "embed", "attention", "pretrain", "fine-tun", "inference"] if t in full.lower())
    s = min(concepts, 5) + 2
    if any(t in full.lower() for t in ["analogy", "like a", "imagine"]): s += 2
    scores[m]["general"].append(min(s, 10))

# general-006: markdown table
for m in models:
    ans = get_answer(all_responses[m]["general-006"])
    lines = [l.strip() for l in ans.strip().split('\n') if l.strip()]
    table_lines = [l for l in lines if '|' in l]
    non_table = [l for l in lines if '|' not in l and not l.startswith('#') and l != '```' and 'markdown' not in l.lower()]
    has_table = len(table_lines) >= 3
    only_table = len(non_table) <= 1
    has_5 = sum(1 for lang in ["python", "javascript", "rust", "go", "java"] if lang in ans.lower()) >= 5
    s = 0
    if has_table: s += 3
    if only_table: s += 2
    if has_5: s += 2
    s += 2
    scores[m]["general"].append(min(s, 10))

# general-007: hallucination test
for m in models:
    full = get_full(all_responses[m]["general-007"])
    refuse_words = ["cannot find", "can't find", "unable to find", "does not exist", "doesn't exist",
                    "not aware", "cannot verify", "can't verify", "no such paper", "not real",
                    "fabricated", "fictional", "don't have", "do not have", "not familiar",
                    "unable to locate", "no evidence", "i couldn't", "i could not",
                    "cannot confirm", "can't confirm", "no information"]
    refused = any(w in full.lower() for w in refuse_words)
    s = 8 if refused else 1
    if refused and any(w in full.lower() for w in ["scholar", "search", "database"]): s = 10
    scores[m]["general"].append(min(s, 10))

# general-008: Wakanda + Burkina Faso
for m in models:
    full = get_full(all_responses[m]["general-008"])
    birnin = "birnin" in full.lower()
    ouaga = "ouagadougou" in full.lower()
    fiction = "fiction" in full.lower() or "marvel" in full.lower()
    s = 0
    if birnin: s += 2
    if ouaga: s += 3
    if fiction: s += 3
    s += 2
    scores[m]["general"].append(min(s, 10))

# general-009: capitalism vs socialism
for m in models:
    full = get_full(all_responses[m]["general-009"])
    dims = sum(1 for t in ["ownership", "government", "wealth", "innovat", "example"] if t in full.lower())
    balanced = any(t in full.lower() for t in ["both", "advantage", "disadvantage", "pros"])
    s = min(dims, 5) + (3 if balanced else 1) + 1
    scores[m]["general"].append(min(s, 10))

# general-010: sentiment analysis
for m in models:
    full = get_full(all_responses[m]["general-010"])
    rule = any(t in full.lower() for t in ["rule", "lexicon", "vader"])
    ml = any(t in full.lower() for t in ["machine learning", "classifier", "supervised"])
    llm = any(t in full.lower() for t in ["llm", "large language", "zero-shot"])
    matrix = "|" in full and "---" in full  # markdown table
    recommend = "recommend" in full.lower()
    s = 0
    if rule: s += 1
    if ml: s += 1
    if llm: s += 1
    if matrix: s += 3
    if recommend: s += 2
    s += 2
    scores[m]["general"].append(min(s, 10))

# general-011: Great Wall myth
for m in models:
    full = get_full(all_responses[m]["general-011"])
    false_claim = any(t in full.lower() for t in ["false", "incorrect", "not true", "myth", "not visible"])
    narrow = any(t in full.lower() for t in ["narrow", "wide", "width", "meter"])
    astronaut = "astronaut" in full.lower() or "yang liwei" in full.lower()
    s = 0
    if false_claim: s += 3
    if narrow: s += 3
    if astronaut: s += 2
    s += 2
    scores[m]["general"].append(min(s, 10))

# general-012: JSON format
for m in models:
    ans = get_answer(all_responses[m]["general-012"])
    full = get_full(all_responses[m]["general-012"])
    # Check valid JSON in full response
    valid = False
    for pattern in [r'```json?\s*\n(.*?)\n```', r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', r'(\{.*\})']:
        match = re.search(pattern, full, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1) if '```' in pattern else match.group(0))
                valid = True
                break
            except:
                pass
    has_fields = all(t in full.lower() for t in ["ada", "lovelace", "1815", "british"])
    s = 0
    if valid: s += 4
    if has_fields: s += 3
    s += 2
    scores[m]["general"].append(min(s, 10))

# ============================================================
# CODING (10 prompts) - Need to read actual code quality
# ============================================================
