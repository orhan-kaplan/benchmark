#!/usr/bin/env python3
"""
Output full responses for a specific model and category.
Usage: python3 _eval_by_model.py <model> <category>
"""
import json, sys

with open("_all_responses.json", encoding="utf-8") as f:
    all_responses = json.load(f)

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-32b"
category = sys.argv[2] if len(sys.argv) > 2 else "coding"

# Map category to prompt IDs
cat_map = {
    "coding": [f"code-{i:03d}" for i in range(1, 11)],
    "creative": [f"creative-{i:03d}" for i in range(1, 6)],
    "general": [f"general-{i:03d}" for i in range(1, 13)],
    "reasoning": [f"reason-{i:03d}" for i in range(1, 13)],
    "translation": [f"tr-{i:03d}" for i in range(1, 16)],
}

prompts = cat_map.get(category, [])
responses = all_responses.get(model, {})

for pid in prompts:
    resp = responses.get(pid, "[NO RESPONSE]")
    # Clean encoding artifacts from DeepSeek tokenizer
    clean = resp.replace("Ġ", " ").replace("Ċ", "\n").replace("â\u0122Ķ", "…").replace("â\u0122ĵ", "—")
    
    print(f"\n{'='*60}")
    print(f"PROMPT: {pid} | MODEL: {model} | LENGTH: {len(resp)} chars")
    print(f"{'='*60}")
    
    # Find </think> tag to separate thinking from answer
    think_end = clean.find("</think>")
    if think_end >= 0:
        thinking = clean[:think_end]
        answer = clean[think_end+8:].strip()
        print(f"[THINKING: {len(thinking)} chars]")
        print(f"[ANSWER:]")
        print(answer)
    else:
        print(clean)
    print()
