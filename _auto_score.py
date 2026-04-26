#!/usr/bin/env python3
"""
Auto-score responses where possible, output detailed analysis for manual scoring.
Focus on objective criteria that can be checked programmatically.
"""
import json, re

with open("_all_responses.json", encoding="utf-8") as f:
    all_responses = json.load(f)

models = sorted(all_responses.keys())

def clean_text(text):
    """Clean encoding artifacts"""
    t = text.replace("Ġ", " ").replace("Ċ", "\n")
    return t

def get_answer(text):
    """Extract answer after </think> tag"""
    clean = clean_text(text)
    idx = clean.find("</think>")
    if idx >= 0:
        return clean[idx+8:].strip()
    return clean.strip()

def word_count(text):
    """Count words in text"""
    return len(text.split())

def has_code_block(text):
    return "```" in text

def check_json_valid(text):
    """Check if response contains valid JSON"""
    # Find JSON block
    match = re.search(r'```json?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1))
            return True
        except:
            return False
    # Try finding raw JSON
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return True
        except:
            pass
    return False

def check_markdown_table(text):
    """Check if response contains a markdown table"""
    lines = text.strip().split('\n')
    table_lines = [l for l in lines if '|' in l]
    return len(table_lines) >= 3

def check_hallucination_007(text):
    """Check if model hallucinates for the fake paper (general-007)"""
    answer = get_answer(text).lower()
    refuse_indicators = [
        "cannot find", "can't find", "unable to find", "no record",
        "does not exist", "doesn't exist", "not aware", "cannot verify",
        "can't verify", "unable to verify", "no such paper", "not real",
        "fabricated", "fictional", "don't have", "do not have",
        "couldn't find", "could not find", "not able to find",
        "i'm not sure", "i am not sure", "cannot confirm", "can't confirm",
        "no information", "not available", "i don't", "i do not",
        "unable to locate", "cannot locate", "no evidence",
        "i couldn't", "i could not", "not familiar",
    ]
    for indicator in refuse_indicators:
        if indicator in answer:
            return "REFUSED (good)"
    return "HALLUCINATED (bad)"

# Score tracking
scores = {m: {
    "coding": [], "creative": [], "general": [], 
    "reasoning": [], "translation": []
} for m in models}

print("=" * 100)
print("DETAILED EVALUATION OF ALL 432 RESPONSES")
print("=" * 100)

# ============================================================
# REASONING EVALUATION
# ============================================================
print("\n\n" + "=" * 80)
print("REASONING CATEGORY")
print("=" * 80)

# reason-001: $129.60
for m in models:
    ans = get_answer(all_responses[m].get("reason-001", ""))
    has_129_60 = "129.60" in ans or "129.6" in ans
    has_steps = "120" in ans and ("9.60" in ans or "9.6" in ans)
    score = 0
    if has_129_60: score += 4
    if has_steps: score += 4
    if "step" in ans.lower() or "1." in ans: score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-001 | {m:35s} | correct={has_129_60} | score={score}")

# reason-002: 40 meters
for m in models:
    ans = get_answer(all_responses[m].get("reason-002", ""))
    has_40 = bool(re.search(r'\b40\b', ans))
    has_geometric = "geometric" in ans.lower() or "series" in ans.lower() or "infinite" in ans.lower()
    score = 0
    if has_40: score += 3
    if has_geometric: score += 3
    if "up and down" in ans.lower() or "both" in ans.lower() or "twice" in ans.lower(): score += 2
    if "formula" in ans.lower() or "1/(1" in ans or "sum" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-002 | {m:35s} | has_40={has_40} | score={score}")

# reason-003: ~70.6%
for m in models:
    ans = get_answer(all_responses[m].get("reason-003", ""))
    has_70 = bool(re.search(r'70\.6|70\.63|0\.706|70\.632', ans)) or "approximately 70" in ans.lower()
    has_complement = "1 -" in ans or "complement" in ans.lower() or "1-" in ans
    score = 0
    if has_70: score += 2
    if has_complement: score += 3
    if "365" in ans: score += 3
    if "step" in ans.lower() or "derivation" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-003 | {m:35s} | has_70%={has_70} | score={score}")

# reason-004: 8 arrangements
for m in models:
    ans = get_answer(all_responses[m].get("reason-004", ""))
    has_8 = bool(re.search(r'\b8\b', ans)) and ("arrangement" in ans.lower() or "possible" in ans.lower() or "valid" in ans.lower())
    score = 0
    if has_8: score += 2
    # Check if lists arrangements
    arrangement_count = len(re.findall(r'(?:Eve|Alice|Bob|Carol|Dave).*(?:Eve|Alice|Bob|Carol|Dave).*(?:Eve|Alice|Bob|Carol|Dave)', ans))
    if arrangement_count >= 6: score += 4
    elif arrangement_count >= 3: score += 2
    if "systematic" in ans.lower() or "case" in ans.lower(): score += 2
    if "constraint" in ans.lower() or "condition" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-004 | {m:35s} | has_8={has_8} | score={score}")

# reason-005: 1 draw
for m in models:
    ans = get_answer(all_responses[m].get("reason-005", ""))
    has_1 = bool(re.search(r'\b1\b.*draw|one.*draw|single.*draw|minimum.*1|minimum.*one', ans.lower()))
    has_mixed = "mixed" in ans.lower() and "draw" in ans.lower()
    score = 0
    if has_1: score += 3
    if has_mixed: score += 3
    if "wrong" in ans.lower() or "incorrect" in ans.lower(): score += 2
    if "deduc" in ans.lower() or "therefore" in ans.lower() or "remaining" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-005 | {m:35s} | has_1_draw={has_1} | score={score}")

# reason-006: pirate puzzle (98, 0, 1, 0, 1)
for m in models:
    ans = get_answer(all_responses[m].get("reason-006", ""))
    has_98 = "98" in ans
    has_backward = "backward" in ans.lower() or "work back" in ans.lower() or "reverse" in ans.lower()
    score = 0
    if has_98: score += 3
    if has_backward: score += 3
    if "2 pirate" in ans.lower() or "base case" in ans.lower(): score += 2
    if "vote" in ans.lower() or "majority" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-006 | {m:35s} | has_98={has_98} | score={score}")

# reason-007: micrometers / dust grain
for m in models:
    ans = get_answer(all_responses[m].get("reason-007", ""))
    has_micro = "micrometer" in ans.lower() or "micron" in ans.lower() or "μm" in ans
    has_tiny = "dust" in ans.lower() or "grain" in ans.lower() or "pinhead" in ans.lower() or "blood cell" in ans.lower()
    has_nucleus = "nucleus" in ans.lower() or "nuclei" in ans.lower()
    score = 0
    if has_micro or has_tiny: score += 3
    if has_nucleus: score += 3
    if "99.9" in ans or "empty" in ans.lower(): score += 2
    if "10^" in ans or "1e-" in ans or "×10" in ans: score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-007 | {m:35s} | micro={has_micro} tiny={has_tiny} | score={score}")

# reason-008: Rayleigh scattering
for m in models:
    ans = get_answer(all_responses[m].get("reason-008", ""))
    has_rayleigh = "rayleigh" in ans.lower()
    has_lambda4 = "λ" in ans or "lambda" in ans.lower() or "1/λ" in ans or "fourth power" in ans.lower() or "λ⁴" in ans or "λ^4" in ans
    has_path = "path" in ans.lower() and ("long" in ans.lower() or "short" in ans.lower())
    score = 0
    if has_rayleigh: score += 3
    if has_lambda4: score += 2
    if has_path: score += 3
    if "dust" in ans.lower() or "particle" in ans.lower() or "pollution" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-008 | {m:35s} | rayleigh={has_rayleigh} | score={score}")

# reason-009: $113.85
for m in models:
    ans = get_answer(all_responses[m].get("reason-009", ""))
    has_113_85 = "113.85" in ans
    has_round_trip = "round trip" in ans.lower() or "× 2" in ans or "×2" in ans or "times 2" in ans.lower() or "* 2" in ans
    score = 0
    if has_113_85: score += 2
    if "150" in ans and "350" in ans: score += 2  # city/highway split
    if "12" in ans and "21" in ans: score += 2  # fuel per segment
    if "15%" in ans or "1.15" in ans or "37.95" in ans: score += 2  # luggage
    if has_round_trip or "75.9" in ans: score += 2  # round trip
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-009 | {m:35s} | has_113.85={has_113_85} | score={score}")

# reason-010: 12-coin problem
for m in models:
    ans = get_answer(all_responses[m].get("reason-010", ""))
    has_4v4 = "4 vs 4" in ans or "4 against 4" in ans.lower() or re.search(r'weigh.*4.*against.*4', ans.lower()) is not None
    has_3_weighings = "3 weighing" in ans.lower() or "three weighing" in ans.lower()
    has_heavier_lighter = "heavier" in ans.lower() and "lighter" in ans.lower()
    score = 0
    if has_4v4: score += 2
    if has_3_weighings or "weighing 1" in ans.lower(): score += 2
    if has_heavier_lighter: score += 1
    # Check for case analysis
    if "case" in ans.lower() or "balanced" in ans.lower(): score += 3
    if "decision tree" in ans.lower() or "branch" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-010 | {m:35s} | 4v4={has_4v4} h/l={has_heavier_lighter} | score={score}")

# reason-011: 99 survivors, parity
for m in models:
    ans = get_answer(all_responses[m].get("reason-011", ""))
    has_99 = bool(re.search(r'\b99\b', ans))
    has_parity = "parity" in ans.lower() or "odd" in ans.lower() or "even" in ans.lower()
    score = 0
    if has_99: score += 2
    if has_parity: score += 4
    if "50" in ans or "50%" in ans or "50/50" in ans: score += 2  # last person's chance
    if "listen" in ans.lower() or "count" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-011 | {m:35s} | has_99={has_99} parity={has_parity} | score={score}")

# reason-012: snail - day 28 and day 14
for m in models:
    ans = get_answer(all_responses[m].get("reason-012", ""))
    has_28 = bool(re.search(r'\b28\b', ans))
    has_14 = bool(re.search(r'\b14\b', ans))
    has_insight = "slide" in ans.lower() or "slip" in ans.lower() or "doesn't slide" in ans.lower() or "no slide" in ans.lower() or "escape day" in ans.lower()
    score = 0
    if has_28: score += 3
    if has_14: score += 3
    if has_insight: score += 2
    if "net" in ans.lower() or "progress" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["reasoning"].append(score)
    print(f"reason-012 | {m:35s} | day28={has_28} day14={has_14} | score={score}")

# ============================================================
# GENERAL EVALUATION
# ============================================================
print("\n\n" + "=" * 80)
print("GENERAL CATEGORY")
print("=" * 80)

# general-001: fission vs fusion
for m in models:
    ans = get_answer(all_responses[m].get("general-001", ""))
    has_fission = "fission" in ans.lower() and ("split" in ans.lower() or "heavy" in ans.lower())
    has_fusion = "fusion" in ans.lower() and ("combin" in ans.lower() or "light" in ans.lower() or "hydrogen" in ans.lower())
    has_sun = "sun" in ans.lower() and "fusion" in ans.lower()
    has_plant = "power plant" in ans.lower() and "fission" in ans.lower()
    has_challenge = "temperature" in ans.lower() or "million" in ans.lower() or "confine" in ans.lower()
    score = 0
    if has_fission: score += 2
    if has_fusion: score += 2
    if has_sun and has_plant: score += 2
    if has_challenge: score += 3
    score += 1  # baseline for attempting
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-001 | {m:35s} | score={score}")

# general-002: transformers
for m in models:
    ans = get_answer(all_responses[m].get("general-002", ""))
    has_self_attn = "self-attention" in ans.lower() or "self attention" in ans.lower()
    has_multi_head = "multi-head" in ans.lower() or "multi head" in ans.lower() or "multiple head" in ans.lower()
    has_pos_enc = "positional" in ans.lower() and ("encoding" in ans.lower() or "embed" in ans.lower())
    has_rnn = "rnn" in ans.lower() or "recurrent" in ans.lower()
    has_qkv = "query" in ans.lower() or "key" in ans.lower() or "value" in ans.lower()
    score = 0
    if has_self_attn: score += 3
    if has_multi_head: score += 2
    if has_pos_enc: score += 2
    if has_rnn: score += 2
    if has_qkv: score += 1
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-002 | {m:35s} | score={score}")

# general-003: Roman Empire
for m in models:
    ans = get_answer(all_responses[m].get("general-003", ""))
    factors = 0
    if "military" in ans.lower() or "barbarian" in ans.lower() or "invasion" in ans.lower(): factors += 1
    if "economic" in ans.lower() or "inflation" in ans.lower() or "tax" in ans.lower(): factors += 1
    if "political" in ans.lower() or "civil war" in ans.lower() or "emperor" in ans.lower(): factors += 1
    if "admin" in ans.lower() or "corruption" in ans.lower() or "bureaucra" in ans.lower(): factors += 1
    if "social" in ans.lower() or "mercenari" in ans.lower() or "civic" in ans.lower(): factors += 1
    if "disease" in ans.lower() or "plague" in ans.lower(): factors += 1
    has_interaction = "interact" in ans.lower() or "led to" in ans.lower() or "caused" in ans.lower() or "result" in ans.lower()
    score = min(factors, 4) + (3 if has_interaction else 1) + 2  # accuracy baseline
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-003 | {m:35s} | factors={factors} | score={score}")

# general-004: trolley problem
for m in models:
    ans = get_answer(all_responses[m].get("general-004", ""))
    has_trolley = "trolley" in ans.lower()
    has_util = "utilitarian" in ans.lower()
    has_deont = "deontolog" in ans.lower() or "kantian" in ans.lower() or "kant" in ans.lower()
    has_virtue = "virtue" in ans.lower()
    has_variation = "fat man" in ans.lower() or "surgeon" in ans.lower() or "footbridge" in ans.lower()
    score = 0
    if has_trolley: score += 2
    if has_variation: score += 2
    if has_util: score += 2
    if has_deont: score += 2
    if has_virtue: score += 2
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-004 | {m:35s} | score={score}")

# general-005: LLMs explained
for m in models:
    ans = get_answer(all_responses[m].get("general-005", ""))
    concepts = 0
    if "tokeniz" in ans.lower(): concepts += 1
    if "embed" in ans.lower(): concepts += 1
    if "attention" in ans.lower(): concepts += 1
    if "pretrain" in ans.lower() or "pre-train" in ans.lower(): concepts += 1
    if "fine-tun" in ans.lower() or "finetun" in ans.lower(): concepts += 1
    if "inference" in ans.lower(): concepts += 1
    score = min(concepts, 5) + 2  # accuracy + accessibility
    if "analogy" in ans.lower() or "like" in ans.lower() or "imagine" in ans.lower(): score += 2
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-005 | {m:35s} | concepts={concepts} | score={score}")

# general-006: markdown table only
for m in models:
    ans = get_answer(all_responses[m].get("general-006", ""))
    has_table = check_markdown_table(ans)
    # Check if ONLY table (no extra text)
    lines = [l.strip() for l in ans.strip().split('\n') if l.strip()]
    non_table = [l for l in lines if '|' not in l and not l.startswith('#')]
    only_table = len(non_table) <= 1  # allow maybe one blank line
    has_5_langs = sum(1 for lang in ["python", "javascript", "rust", "go", "java"] if lang in ans.lower()) >= 5
    score = 0
    if has_table: score += 3
    if only_table: score += 2
    if has_5_langs: score += 2
    score += 2  # accuracy baseline
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-006 | {m:35s} | table={has_table} only={only_table} | score={score}")

# general-007: hallucination test (fake paper)
for m in models:
    ans = get_answer(all_responses[m].get("general-007", ""))
    result = check_hallucination_007(all_responses[m].get("general-007", ""))
    refused = "REFUSED" in result
    score = 0
    if refused:
        score = 8  # good - refused to hallucinate
        if "suggest" in ans.lower() or "search" in ans.lower() or "scholar" in ans.lower(): score = 10
    else:
        score = 1  # bad - hallucinated
    scores[m]["general"].append(score)
    print(f"general-007 | {m:35s} | {result} | score={score}")

# general-008: Wakanda + Burkina Faso
for m in models:
    ans = get_answer(all_responses[m].get("general-008", ""))
    has_birnin = "birnin" in ans.lower()
    has_ouaga = "ouagadougou" in ans.lower()
    has_fiction = "fiction" in ans.lower() or "fictional" in ans.lower() or "marvel" in ans.lower()
    has_real = "real" in ans.lower() or "actual" in ans.lower()
    score = 0
    if has_birnin: score += 2
    if has_ouaga: score += 3
    if has_fiction: score += 3
    if has_real: score += 2
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-008 | {m:35s} | birnin={has_birnin} ouaga={has_ouaga} | score={score}")

# general-009: capitalism vs socialism
for m in models:
    ans = get_answer(all_responses[m].get("general-009", ""))
    dims = 0
    if "ownership" in ans.lower() or "means of production" in ans.lower(): dims += 1
    if "government" in ans.lower() or "state" in ans.lower(): dims += 1
    if "wealth" in ans.lower() or "distribut" in ans.lower() or "inequality" in ans.lower(): dims += 1
    if "innovat" in ans.lower() or "incentive" in ans.lower(): dims += 1
    if "example" in ans.lower() or "ussr" in ans.lower() or "nordic" in ans.lower(): dims += 1
    balanced = "both" in ans.lower() or "advantage" in ans.lower() or "disadvantage" in ans.lower() or "pros" in ans.lower()
    score = min(dims, 5) + (3 if balanced else 1) + 1
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-009 | {m:35s} | dims={dims} | score={score}")

# general-010: sentiment analysis
for m in models:
    ans = get_answer(all_responses[m].get("general-010", ""))
    has_rule = "rule" in ans.lower() or "lexicon" in ans.lower() or "vader" in ans.lower()
    has_ml = "machine learning" in ans.lower() or "classifier" in ans.lower() or "supervised" in ans.lower()
    has_llm = "llm" in ans.lower() or "large language" in ans.lower() or "zero-shot" in ans.lower()
    has_matrix = check_markdown_table(ans) or "matrix" in ans.lower()
    has_recommend = "recommend" in ans.lower() or "suggest" in ans.lower()
    score = 0
    if has_rule: score += 1
    if has_ml: score += 1
    if has_llm: score += 1
    if has_matrix: score += 3
    if has_recommend: score += 2
    if "step" in ans.lower() or "1." in ans: score += 1
    score += 1  # baseline
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-010 | {m:35s} | rule={has_rule} ml={has_ml} llm={has_llm} | score={score}")

# general-011: Great Wall myth
for m in models:
    ans = get_answer(all_responses[m].get("general-011", ""))
    has_false = "false" in ans.lower() or "incorrect" in ans.lower() or "not true" in ans.lower() or "myth" in ans.lower() or "wrong" in ans.lower() or "not visible" in ans.lower()
    has_narrow = "narrow" in ans.lower() or "wide" in ans.lower() or "width" in ans.lower() or "meter" in ans.lower()
    has_astronaut = "astronaut" in ans.lower() or "yang liwei" in ans.lower() or "space" in ans.lower()
    score = 0
    if has_false: score += 3
    if has_narrow: score += 3
    if has_astronaut: score += 2
    score += 2  # baseline
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-011 | {m:35s} | false={has_false} | score={score}")

# general-012: JSON format
for m in models:
    ans = get_answer(all_responses[m].get("general-012", ""))
    valid_json = check_json_valid(ans)
    # Check no extra text
    lines = [l.strip() for l in ans.strip().split('\n') if l.strip()]
    non_json = [l for l in lines if not any(c in l for c in ['{', '}', '"', ':', '[', ']', '```'])]
    only_json = len(non_json) <= 2
    has_all_fields = all(f in ans.lower() for f in ["ada", "lovelace", "1815", "1852", "british", "mathematics"])
    score = 0
    if valid_json: score += 4
    if only_json: score += 2
    if has_all_fields: score += 2
    if "fields" in ans and "[" in ans: score += 1  # array for fields
    score += 1  # baseline
    score = min(score, 10)
    scores[m]["general"].append(score)
    print(f"general-012 | {m:35s} | valid_json={valid_json} only={only_json} | score={score}")

print("\n\n" + "=" * 100)
print("SUMMARY SCORES")
print("=" * 100)

# Now I need to handle coding, creative, and translation manually
# For now, output what we have and placeholder for manual scores

print(f"\n{'Model':35s} | {'Coding':>7s} | {'Transl':>7s} | {'Reason':>7s} | {'Creative':>8s} | {'General':>7s} | {'AVG':>5s}")
print("-" * 100)

for m in models:
    r_scores = scores[m]["reasoning"]
    g_scores = scores[m]["general"]
    r_avg = sum(r_scores) / len(r_scores) if r_scores else 0
    g_avg = sum(g_scores) / len(g_scores) if g_scores else 0
    
    short = m.replace("deepseek-r1-distill-qwen-32b", "ds-r1-32b")
    short = short.replace("qwen3.5-35b-a3b-fp8", "q3.5-35b-fp8")
    short = short.replace("qwen3.5-35b-a3b", "q3.5-35b")
    short = short.replace("qwen3.6-27b-fp8", "q3.6-27b-fp8")
    short = short.replace("qwen3.6-27b", "q3.6-27b")
    short = short.replace("qwen3-32b-fp8", "q3-32b-fp8")
    short = short.replace("qwen3-32b", "q3-32b")
    short = short.replace("qwen3-0.6b", "q3-0.6b")
    
    print(f"{short:35s} | {'TBD':>7s} | {'TBD':>7s} | {r_avg:>7.1f} | {'TBD':>8s} | {g_avg:>7.1f} | {'TBD':>5s}")
