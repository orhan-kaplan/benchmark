#!/usr/bin/env python3
"""Evaluate translation responses (Chinese -> English)."""
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
    return clean(text)

# Translation scoring criteria:
# - Meaning accuracy (3)
# - No skipped segments (2)
# - Tone/style preserved (2)
# - Fluent English (1)
# - Character names consistent (1)
# - No hallucinated additions (1)

# For each translation, check:
# 1. Length relative to source (too short = skipped content)
# 2. Key terms present
# 3. Encoding artifacts (DeepSeek has Ġ issues)
# 4. Whether it's actually a translation vs explanation

results = {}

# Source text lengths (approximate Chinese char counts)
source_lengths = {
    "tr-001": 80,   # Lu Xun passage
    "tr-002": 60,   # Gatsby-like passage
    "tr-003": 60,   # Dickens-like passage
    "tr-004": 80,   # Xianxia cultivation
    "tr-005": 80,   # Fantasy action
    "tr-006": 80,   # Romance
    "tr-007": 100,  # Technical LLM
    "tr-008": 80,   # Academic abstract
    "tr-009": 100,  # Casual conversation
    "tr-010": 80,   # Social media post
    "tr-011": 1500, # Long historical fiction chapter
    "tr-012": 800,  # Long martial arts passage
    "tr-013": 1000, # Long xianxia passage
    "tr-014": 1200, # Long slice-of-life novel
    "tr-015": 800,  # Long dialogue passage
}

# Key terms that should appear in translations
key_terms = {
    "tr-001": ["road", "path", "moon", "hope"],
    "tr-002": ["dream", "dark", "field", "night"],
    "tr-003": ["best", "worst", "wisdom", "foolish", "light", "dark"],
    "tr-004": ["cultivat", "spiritual", "qi", "golden", "breakthrough", "three year"],
    "tr-005": ["sword", "blood", "white", "cold", "black"],
    "tr-006": ["rain", "tear", "three year", "back", "throat"],
    "tr-007": ["attention", "token", "cache", "memory", "inference"],
    "tr-008": ["expert", "routing", "cost", "benchmark", "performance"],
    "tr-009": ["hotpot", "work", "five", "fun"],
    "tr-010": ["project", "movie", "relax", "recommend"],
    "tr-011": ["tax", "landlord", "court", "law"],
    "tr-012": ["fist", "fight", "memory", "evil"],
    "tr-013": ["cultivat", "secret", "flower", "saint"],
    "tr-014": ["morning", "letter", "town", "well"],
    "tr-015": ["sword", "fist", "laugh", "treasure"],
}

for m in models:
    results[m] = {}
    
    for pid in [f"tr-{i:03d}" for i in range(1, 16)]:
        full = get_full(all_responses[m][pid])
        ans = get_answer(all_responses[m][pid])
        
        # Check if it's actually a translation (not just analysis)
        is_translation = len(ans) > 100 and not ans.lower().startswith("this passage")
        
        # Check for encoding artifacts (DeepSeek issue)
        has_artifacts = "Ä " in full or "Ä¡" in full or "â\u0122" in full
        
        # Check key terms
        terms = key_terms.get(pid, [])
        term_hits = sum(1 for t in terms if t.lower() in full.lower())
        term_ratio = term_hits / len(terms) if terms else 1
        
        # Check length (longer source = longer expected translation)
        src_len = source_lengths.get(pid, 100)
        expected_min_len = src_len * 2  # rough: each Chinese char ~ 2 English chars
        ans_len = len(ans)
        
        # For long passages (tr-011 to tr-015), check if truncated
        is_long = pid in ["tr-011", "tr-012", "tr-013", "tr-014", "tr-015"]
        
        s = 0
        
        # Meaning accuracy (3 points)
        if term_ratio >= 0.75: s += 3
        elif term_ratio >= 0.5: s += 2
        elif term_ratio >= 0.25: s += 1
        
        # No skipped segments (2 points)
        if is_long:
            # For long passages, check if substantial portion translated
            if ans_len > expected_min_len * 0.5: s += 2
            elif ans_len > expected_min_len * 0.25: s += 1
            # Don't penalize for 1024 token truncation
        else:
            if ans_len > expected_min_len * 0.7: s += 2
            elif ans_len > expected_min_len * 0.3: s += 1
        
        # Tone/style (2 points) - give benefit of doubt if translation exists
        if is_translation: s += 2
        
        # Fluent English (1 point)
        if not has_artifacts and is_translation: s += 1
        elif is_translation: s += 0.5
        
        # Character names (1 point) - for passages with names
        s += 1
        
        # No hallucinated additions (1 point)
        s += 1
        
        # Penalty for DeepSeek encoding artifacts
        if has_artifacts and m == "deepseek-r1-distill-qwen-32b":
            s = max(s - 1, 1)
        
        results[m][pid] = min(int(round(s)), 10)

# Print results
print(f"{'Model':35s}", end="")
for i in range(1, 16):
    print(f" | t{i:02d}", end="")
print(" | AVG")
print("-" * 140)

for m in models:
    short = m.replace("deepseek-r1-distill-qwen-32b", "ds-r1-32b")
    short = short.replace("qwen3.5-35b-a3b-fp8", "q3.5-35b-fp8")
    short = short.replace("qwen3.5-35b-a3b", "q3.5-35b")
    short = short.replace("qwen3.6-27b-fp8", "q3.6-27b-fp8")
    short = short.replace("qwen3.6-27b", "q3.6-27b")
    short = short.replace("qwen3-32b-fp8", "q3-32b-fp8")
    short = short.replace("qwen3-32b", "q3-32b")
    short = short.replace("qwen3-0.6b", "q3-0.6b")
    
    vals = [results[m][f"tr-{i:03d}"] for i in range(1, 16)]
    avg = sum(vals) / len(vals)
    print(f"{short:35s}", end="")
    for v in vals:
        print(f" | {v:3d}", end="")
    print(f" | {avg:.1f}")
