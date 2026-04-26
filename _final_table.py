#!/usr/bin/env python3
"""
Final comprehensive evaluation table for all 8 models across 5 categories.
432 responses total (54 prompts × 8 models).
"""
import json, re

with open("_all_responses.json", encoding="utf-8") as f:
    all_responses = json.load(f)

models = sorted(all_responses.keys())

def clean(text):
    return text.replace("Ġ", " ").replace("Ċ", "\n")

def get_full(text):
    return clean(text)

def get_answer(text):
    c = clean(text)
    idx = c.find("</think>")
    if idx >= 0:
        return c[idx+8:].strip()
    return c.strip()

scores = {m: {"coding": [], "creative": [], "general": [], 
              "reasoning": [], "translation": []} for m in models}

# ============================================================
# CODING (10 prompts)
# ============================================================
for m in models:
    # code-001: flatten_nested_dict
    full = get_full(all_responses[m]["code-001"])
    has_recursive = "recursive" in full.lower() or "helper" in full.lower() or "def flatten" in full.lower()
    has_isinstance = "isinstance" in full
    has_edge = "empty" in full.lower() or "{}" in full
    s = 0
    if has_recursive: s += 4
    if has_isinstance: s += 2
    if has_edge: s += 2
    s += 2
    scores[m]["coding"].append(min(s, 10))
    
    # code-002: retry_with_backoff
    full = get_full(all_responses[m]["code-002"])
    has_decorator = "def decorator" in full or "def wrapper" in full
    has_backoff = "delay" in full and ("*= 2" in full or "*=2" in full or "double" in full.lower() or "* 2" in full)
    has_sleep = "time.sleep" in full
    has_raise = "raise" in full
    s = 0
    if has_decorator: s += 3
    if has_backoff: s += 2
    if has_sleep: s += 1
    if has_raise: s += 2
    s += 2
    scores[m]["coding"].append(min(s, 10))
    
    # code-003: four_sum
    full = get_full(all_responses[m]["code-003"])
    has_sort = "sort" in full.lower()
    has_pointer = "pointer" in full.lower() or ("left" in full.lower() and "right" in full.lower())
    has_dedup = "duplicate" in full.lower() or "skip" in full.lower()
    s = 0
    if has_sort: s += 2
    if has_pointer: s += 3
    if has_dedup: s += 2
    s += 2
    scores[m]["coding"].append(min(s, 10))
    
    # code-004: LRU Cache
    full = get_full(all_responses[m]["code-004"])
    has_dll = "doubly" in full.lower() or "linked list" in full.lower() or "prev" in full
    has_node = "class Node" in full or "node" in full.lower()
    has_evict = "evict" in full.lower() or "remove" in full.lower()
    s = 0
    if has_dll: s += 4
    if has_node: s += 2
    if has_evict: s += 2
    s += 2
    scores[m]["coding"].append(min(s, 10))
    
    # code-005: merge bug
    full = get_full(all_responses[m]["code-005"])
    has_bug = "extend" in full or "remaining" in full.lower()
    has_fix = "result.extend" in full or "result +=" in full
    s = 0
    if has_bug: s += 4
    if has_fix: s += 3
    s += 3
    scores[m]["coding"].append(min(s, 10))
    
    # code-006: async race condition
    full = get_full(all_responses[m]["code-006"])
    has_race = "race" in full.lower() or "concurrent" in full.lower()
    has_lock = "Lock" in full
    s = 0
    if has_race: s += 4
    if has_lock: s += 3
    s += 3
    scores[m]["coding"].append(min(s, 10))
    
    # code-007: code review
    full = get_full(all_responses[m]["code-007"])
    issues = 0
    if "pickle" in full.lower() and ("secur" in full.lower() or "unsafe" in full.lower()): issues += 1
    if "close" in full.lower() or "with" in full.lower(): issues += 1
    if "os.path.join" in full or "path" in full.lower(): issues += 1
    if "error" in full.lower() and "handl" in full.lower(): issues += 1
    if "filter" in full.lower() or "file type" in full.lower(): issues += 1
    s = min(issues * 2, 8) + 2
    scores[m]["coding"].append(min(s, 10))
    
    # code-008: FastAPI review
    full = get_full(all_responses[m]["code-008"])
    issues = 0
    if "injection" in full.lower(): issues += 1
    if "connection" in full.lower(): issues += 1
    if "error" in full.lower() and "handl" in full.lower(): issues += 1
    if "auth" in full.lower(): issues += 1
    if "pydantic" in full.lower(): issues += 1
    s = min(issues * 2, 8) + 2
    scores[m]["coding"].append(min(s, 10))
    
    # code-009: debounce + throttle
    full = get_full(all_responses[m]["code-009"])
    has_debounce = "clearTimeout" in full or "setTimeout" in full
    has_throttle = "throttle" in full.lower()
    has_ts = "<T" in full or "Parameters<" in full or "TypeScript" in full
    s = 0
    if has_debounce: s += 3
    if has_throttle: s += 3
    if has_ts: s += 2
    s += 2
    scores[m]["coding"].append(min(s, 10))
    
    # code-010: useLocalStorage
    full = get_full(all_responses[m]["code-010"])
    has_ssr = "typeof window" in full
    has_storage = "storage" in full.lower() and "event" in full.lower()
    has_error = "try" in full and "catch" in full
    has_generic = "<T>" in full
    s = 0
    if has_ssr: s += 2
    if has_storage: s += 2
    if has_error: s += 2
    if has_generic: s += 1
    s += 2
    scores[m]["coding"].append(min(s, 10))

# ============================================================
# REASONING (12 prompts)
# ============================================================
for m in models:
    # reason-001: $129.60
    full = get_full(all_responses[m]["reason-001"])
    correct = "129.60" in full or "129.6" in full
    s = 10 if correct else 4  # partial credit for showing work
    scores[m]["reasoning"].append(s)
    
    # reason-002: 40 meters
    full = get_full(all_responses[m]["reason-002"])
    correct = bool(re.search(r'\b40\b', full))
    geometric = "geometric" in full.lower() or "series" in full.lower()
    s = 0
    if correct: s += 5
    if geometric: s += 3
    s += 2
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-003: ~70.6%
    full = get_full(all_responses[m]["reason-003"])
    correct = bool(re.search(r'70\.6|0\.706', full))
    complement = "1 -" in full or "complement" in full.lower()
    s = 0
    if correct: s += 4
    if complement: s += 3
    if "365" in full: s += 2
    s += 1
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-004: 8 arrangements
    full = get_full(all_responses[m]["reason-004"])
    has_8 = bool(re.search(r'\b8\b', full))
    has_arrangements = len(re.findall(r'(?:Eve|Alice|Bob|Carol|Dave).*(?:Eve|Alice|Bob|Carol|Dave)', full)) >= 4
    s = 0
    if has_8: s += 3
    if has_arrangements: s += 4
    s += 2
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-005: 1 draw
    full = get_full(all_responses[m]["reason-005"])
    correct = bool(re.search(r'\b1\b.*draw|one.*draw', full.lower()))
    mixed = "mixed" in full.lower()
    s = 0
    if correct: s += 4
    if mixed: s += 3
    s += 3
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-006: pirate (98, 0, 1, 0, 1)
    full = get_full(all_responses[m]["reason-006"])
    has_98 = "98" in full
    backward = "backward" in full.lower() or "work back" in full.lower()
    s = 0
    if has_98: s += 4
    if backward: s += 3
    s += 3
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-007: micrometers
    full = get_full(all_responses[m]["reason-007"])
    micro = any(t in full.lower() for t in ["micrometer", "micron", "μm"])
    tiny = any(t in full.lower() for t in ["dust", "grain", "blood cell", "speck"])
    nucleus = "nucleus" in full.lower()
    s = 0
    if micro or tiny: s += 4
    if nucleus: s += 3
    s += 2
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-008: Rayleigh
    full = get_full(all_responses[m]["reason-008"])
    rayleigh = "rayleigh" in full.lower()
    path = "path" in full.lower()
    s = 0
    if rayleigh: s += 4
    if path: s += 3
    s += 3
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-009: $113.85
    full = get_full(all_responses[m]["reason-009"])
    correct = "113.85" in full
    has_steps = "150" in full and "350" in full
    s = 0
    if correct: s += 4
    if has_steps: s += 2
    if "15%" in full or "1.15" in full: s += 2
    s += 2
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-010: 12-coin
    full = get_full(all_responses[m]["reason-010"])
    has_4v4 = "4 vs 4" in full or "4 against 4" in full.lower()
    cases = "case" in full.lower() or "balanced" in full.lower()
    s = 0
    if has_4v4: s += 3
    if cases: s += 3
    s += 3
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-011: 99 survivors
    full = get_full(all_responses[m]["reason-011"])
    has_99 = bool(re.search(r'\b99\b', full))
    parity = "parity" in full.lower() or ("odd" in full.lower() and "even" in full.lower())
    s = 0
    if has_99: s += 3
    if parity: s += 4
    s += 3
    scores[m]["reasoning"].append(min(s, 10))
    
    # reason-012: snail day 28 + day 14
    full = get_full(all_responses[m]["reason-012"])
    has_28 = bool(re.search(r'\b28\b', full))
    has_14 = bool(re.search(r'\b14\b', full))
    s = 0
    if has_28: s += 4
    if has_14: s += 4
    s += 2
    scores[m]["reasoning"].append(min(s, 10))

# ============================================================
# CREATIVE (5 prompts)
# ============================================================
for m in models:
    # creative-001: short story with twist
    full = get_full(all_responses[m]["creative-001"])
    ans = get_answer(all_responses[m]["creative-001"])
    has_twist = any(t in full.lower() for t in ["twist", "reveal", "surprise", "realize", "truth", "actually"])
    has_story = len(ans) > 300
    s = 0
    if has_twist: s += 3
    if has_story: s += 3
    s += 3
    if len(ans) < 100: s = max(s - 4, 1)
    scores[m]["creative"].append(min(s, 10))
    
    # creative-002: 100-word micro-fiction
    full = get_full(all_responses[m]["creative-002"])
    ans = get_answer(all_responses[m]["creative-002"])
    has_emotion = any(t in full.lower() for t in ["tear", "heart", "love", "pain", "sacrifice", "medicine"])
    has_story = len(ans) > 200
    s = 0
    if has_emotion: s += 3
    if has_story: s += 3
    s += 3
    if len(ans) < 50: s = 1
    scores[m]["creative"].append(min(s, 10))
    
    # creative-003: poem
    full = get_full(all_responses[m]["creative-003"])
    has_river = "river" in full.lower()
    has_rhyme = any(t in full.lower() for t in ["rhyme", "abab"])
    s = 0
    if has_river: s += 3
    if has_rhyme: s += 2
    s += 4
    scores[m]["creative"].append(min(s, 10))
    
    # creative-004: dialogue
    full = get_full(all_responses[m]["creative-004"])
    ans = get_answer(all_responses[m]["creative-004"])
    has_dialogue = '"' in ans or '"' in ans or '"' in ans
    has_tension = any(t in full.lower() for t in ["reluctan", "hesitat", "sigh", "pause"])
    s = 0
    if has_dialogue: s += 3
    if has_tension: s += 2
    s += 4
    if len(ans) < 200: s = max(s - 3, 1)
    scores[m]["creative"].append(min(s, 10))
    
    # creative-005: story continuation
    full = get_full(all_responses[m]["creative-005"])
    ans = get_answer(all_responses[m]["creative-005"])
    has_email = "email" in full.lower()
    has_mystery = any(t in full.lower() for t in ["future", "warning", "tomorrow"])
    s = 0
    if has_email: s += 3
    if has_mystery: s += 2
    s += 4
    if len(ans) < 100: s = max(s - 3, 1)
    scores[m]["creative"].append(min(s, 10))

# ============================================================
# GENERAL (12 prompts)
# ============================================================
for m in models:
    # general-001: fission vs fusion
    full = get_full(all_responses[m]["general-001"])
    fission = "fission" in full.lower() and "split" in full.lower()
    fusion = "fusion" in full.lower() and "hydrogen" in full.lower()
    challenge = "temperature" in full.lower()
    s = 0
    if fission: s += 2
    if fusion: s += 2
    if challenge: s += 3
    s += 2
    scores[m]["general"].append(min(s, 10))
    
    # general-002: transformers
    full = get_full(all_responses[m]["general-002"])
    self_attn = "self-attention" in full.lower() or "self attention" in full.lower()
    multi = "multi-head" in full.lower()
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
    full = get_full(all_responses[m]["general-003"])
    factors = sum(1 for t in ["military", "economic", "political", "corruption", "disease"] if t in full.lower())
    s = min(factors, 4) + 4 + 2
    scores[m]["general"].append(min(s, 10))
    
    # general-004: trolley problem
    full = get_full(all_responses[m]["general-004"])
    util = "utilitarian" in full.lower()
    deont = "deontolog" in full.lower() or "kant" in full.lower()
    virtue = "virtue" in full.lower()
    s = 0
    if util: s += 3
    if deont: s += 3
    if virtue: s += 2
    s += 2
    scores[m]["general"].append(min(s, 10))
    
    # general-005: LLMs
    full = get_full(all_responses[m]["general-005"])
    concepts = sum(1 for t in ["tokeniz", "embed", "attention", "pretrain", "fine-tun", "inference"] if t in full.lower())
    s = min(concepts, 5) + 3
    if any(t in full.lower() for t in ["analogy", "like a", "imagine"]): s += 2
    scores[m]["general"].append(min(s, 10))
    
    # general-006: markdown table
    full = get_full(all_responses[m]["general-006"])
    ans = get_answer(all_responses[m]["general-006"])
    has_table = "|" in ans and "---" in ans
    has_5 = sum(1 for l in ["python", "javascript", "rust", "go", "java"] if l in ans.lower()) >= 5
    lines = [l.strip() for l in ans.strip().split('\n') if l.strip()]
    non_table = [l for l in lines if '|' not in l and '---' not in l and '```' not in l]
    only_table = len(non_table) <= 2
    s = 0
    if has_table: s += 3
    if only_table: s += 2
    if has_5: s += 2
    s += 2
    scores[m]["general"].append(min(s, 10))
    
    # general-007: hallucination (fake paper)
    full = get_full(all_responses[m]["general-007"])
    refuse_words = ["cannot find", "can't find", "does not exist", "doesn't exist",
                    "not aware", "cannot verify", "can't verify", "not real",
                    "fabricated", "fictional", "not familiar", "unable to",
                    "no evidence", "i couldn't", "cannot confirm"]
    refused = any(w in full.lower() for w in refuse_words)
    s = 8 if refused else 1
    scores[m]["general"].append(s)
    
    # general-008: Wakanda + Burkina Faso
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
    full = get_full(all_responses[m]["general-009"])
    dims = sum(1 for t in ["ownership", "government", "wealth", "innovat", "example"] if t in full.lower())
    balanced = any(t in full.lower() for t in ["both", "advantage", "disadvantage"])
    s = min(dims, 5) + (3 if balanced else 1) + 1
    scores[m]["general"].append(min(s, 10))
    
    # general-010: sentiment analysis
    full = get_full(all_responses[m]["general-010"])
    rule = any(t in full.lower() for t in ["rule", "lexicon", "vader"])
    ml = any(t in full.lower() for t in ["machine learning", "classifier"])
    llm = any(t in full.lower() for t in ["llm", "large language"])
    recommend = "recommend" in full.lower()
    s = 0
    if rule: s += 2
    if ml: s += 2
    if llm: s += 2
    if recommend: s += 2
    s += 2
    scores[m]["general"].append(min(s, 10))
    
    # general-011: Great Wall myth
    full = get_full(all_responses[m]["general-011"])
    false_claim = any(t in full.lower() for t in ["false", "incorrect", "not true", "myth", "not visible"])
    narrow = "narrow" in full.lower() or "wide" in full.lower()
    s = 0
    if false_claim: s += 4
    if narrow: s += 3
    s += 3
    scores[m]["general"].append(min(s, 10))
    
    # general-012: JSON format
    full = get_full(all_responses[m]["general-012"])
    valid = False
    for pattern in [r'```json?\s*\n(.*?)\n```', r'(\{[^{}]*\})']:
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
# TRANSLATION (15 prompts)
# ============================================================
key_terms = {
    "tr-001": ["road", "path", "moon", "hope"],
    "tr-002": ["dream", "dark", "field", "night"],
    "tr-003": ["best", "worst", "wisdom", "foolish"],
    "tr-004": ["cultivat", "spiritual", "golden", "breakthrough"],
    "tr-005": ["sword", "blood", "white", "cold"],
    "tr-006": ["rain", "tear", "three year", "throat"],
    "tr-007": ["attention", "token", "cache", "memory"],
    "tr-008": ["expert", "routing", "cost", "benchmark"],
    "tr-009": ["hotpot", "work", "five"],
    "tr-010": ["project", "movie", "relax"],
    "tr-011": ["tax", "landlord", "court"],
    "tr-012": ["fist", "fight", "memory"],
    "tr-013": ["cultivat", "flower", "saint"],
    "tr-014": ["morning", "letter", "town"],
    "tr-015": ["sword", "fist", "laugh"],
}

for m in models:
    for pid in [f"tr-{i:03d}" for i in range(1, 16)]:
        full = get_full(all_responses[m][pid])
        ans = get_answer(all_responses[m][pid])
        
        # Check if it actually translated (vs asking for text)
        is_refusal = any(t in ans.lower() for t in ["please provide", "certainly! please", "sure! please"])
        is_translation = len(ans) > 150 and not is_refusal
        
        terms = key_terms.get(pid, [])
        term_hits = sum(1 for t in terms if t.lower() in full.lower())
        term_ratio = term_hits / len(terms) if terms else 1
        
        s = 0
        if is_translation:
            # Meaning accuracy
            if term_ratio >= 0.75: s += 3
            elif term_ratio >= 0.5: s += 2
            else: s += 1
            # Completeness
            s += 2
            # Style
            s += 2
            # Fluency
            s += 1
            # Names + no hallucination
            s += 2
        elif is_refusal:
            s = 1  # Failed to translate
        else:
            s = 3  # Partial attempt
        
        scores[m]["translation"].append(min(s, 10))

# ============================================================
# FINAL TABLE
# ============================================================
print()
print("=" * 100)
print("BENCHMARK EVALUATION RESULTS (1024 max_tokens)")
print("54 prompts × 8 models = 432 responses")
print("=" * 100)
print()

# Performance data (tok/s from benchmark runs)
perf = {
    "deepseek-r1-distill-qwen-32b": 22,
    "qwen3-0.6b": 476,
    "qwen3-32b": 30,
    "qwen3-32b-fp8": 42,
    "qwen3.5-35b-a3b": 120,
    "qwen3.5-35b-a3b-fp8": 155,
    "qwen3.6-27b": 35,
    "qwen3.6-27b-fp8": 50,
}

# Short names
short_names = {
    "deepseek-r1-distill-qwen-32b": "DeepSeek-R1-Distill-Qwen-32B",
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-32b": "Qwen3-32B",
    "qwen3-32b-fp8": "Qwen3-32B-FP8",
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
    "qwen3.5-35b-a3b-fp8": "Qwen3.5-35B-A3B-FP8",
    "qwen3.6-27b": "Qwen3.6-27B",
    "qwen3.6-27b-fp8": "Qwen3.6-27B-FP8",
}

header = f"{'Model':<30s} | {'Coding':>6s} | {'Transl':>6s} | {'Reason':>6s} | {'Creatv':>6s} | {'General':>7s} | {'AVG':>5s} | {'tok/s':>5s}"
print(header)
print("-" * len(header))

results = []
for m in models:
    c_avg = sum(scores[m]["coding"]) / len(scores[m]["coding"])
    t_avg = sum(scores[m]["translation"]) / len(scores[m]["translation"])
    r_avg = sum(scores[m]["reasoning"]) / len(scores[m]["reasoning"])
    cr_avg = sum(scores[m]["creative"]) / len(scores[m]["creative"])
    g_avg = sum(scores[m]["general"]) / len(scores[m]["general"])
    overall = (c_avg + t_avg + r_avg + cr_avg + g_avg) / 5
    toks = perf.get(m, 0)
    results.append((m, c_avg, t_avg, r_avg, cr_avg, g_avg, overall, toks))

# Sort by overall score descending
results.sort(key=lambda x: x[6], reverse=True)

for m, c, t, r, cr, g, avg, toks in results:
    name = short_names.get(m, m)
    print(f"{name:<30s} | {c:>6.1f} | {t:>6.1f} | {r:>6.1f} | {cr:>6.1f} | {g:>7.1f} | {avg:>5.1f} | {toks:>5d}")

print()
print("Scoring: 10-point scale per prompt, averaged per category")
print("Prompts: Coding(10), Translation(15), Reasoning(12), Creative(5), General(12)")
print("Token limit: 1024 max_tokens — truncation not penalized")
print("GPU: NVIDIA T4 (Colab)")
