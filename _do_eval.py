#!/usr/bin/env python3
"""
Evaluate all 432 responses against rubrics.
Reads full responses and outputs per-prompt scores for each model.
Outputs responses grouped by category for manual review.
"""
import json

with open("_all_responses.json", encoding="utf-8") as f:
    all_responses = json.load(f)

models = sorted(all_responses.keys())
categories = ["coding", "creative", "general", "reasoning", "translation"]

# Map prompt_id to category
prompt_cats = {}
cat_prompts = {c: [] for c in categories}

# coding: code-001 to code-010
for i in range(1, 11):
    pid = f"code-{i:03d}"
    prompt_cats[pid] = "coding"
    cat_prompts["coding"].append(pid)

# creative: creative-001 to creative-005
for i in range(1, 6):
    pid = f"creative-{i:03d}"
    prompt_cats[pid] = "creative"
    cat_prompts["creative"].append(pid)

# general: general-001 to general-012
for i in range(1, 13):
    pid = f"general-{i:03d}"
    prompt_cats[pid] = "general"
    cat_prompts["general"].append(pid)

# reasoning: reason-001 to reason-012
for i in range(1, 13):
    pid = f"reason-{i:03d}"
    prompt_cats[pid] = "reasoning"
    cat_prompts["reasoning"].append(pid)

# translation: tr-001 to tr-015
for i in range(1, 16):
    pid = f"tr-{i:03d}"
    prompt_cats[pid] = "translation"
    cat_prompts["translation"].append(pid)

# For each model, for each category, print responses with lengths
for cat in categories:
    print(f"\n{'='*80}")
    print(f"CATEGORY: {cat.upper()} ({len(cat_prompts[cat])} prompts)")
    print(f"{'='*80}")
    
    for pid in cat_prompts[cat]:
        print(f"\n--- {pid} ---")
        for m in models:
            resp = all_responses.get(m, {}).get(pid, "")
            # Clean encoding artifacts
            clean = resp.replace("Ġ", " ").replace("Ċ", "\n").replace("â", "—").replace("ĠĠ", "  ")
            length = len(resp)
            # Show first 300 chars
            preview = clean[:300].strip()
            if len(clean) > 300:
                preview += "..."
            
            # Short model name
            short = m.replace("deepseek-r1-distill-qwen-32b", "ds-r1-32b")
            short = short.replace("qwen3.5-35b-a3b-fp8", "q3.5-35b-fp8")
            short = short.replace("qwen3.5-35b-a3b", "q3.5-35b")
            short = short.replace("qwen3.6-27b-fp8", "q3.6-27b-fp8")
            short = short.replace("qwen3.6-27b", "q3.6-27b")
            short = short.replace("qwen3-32b-fp8", "q3-32b-fp8")
            short = short.replace("qwen3-32b", "q3-32b")
            short = short.replace("qwen3-0.6b", "q3-0.6b")
            
            print(f"  [{short:15s}] ({length:5d} chars) {preview[:150]}")
