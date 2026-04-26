#!/usr/bin/env python3
"""
Compare all models' answers for a specific prompt.
Usage: python3 _eval_compare.py <prompt_id>
"""
import json, sys

with open("_all_responses.json", encoding="utf-8") as f:
    all_responses = json.load(f)

pid = sys.argv[1] if len(sys.argv) > 1 else "code-001"
models = sorted(all_responses.keys())

for m in models:
    resp = all_responses.get(m, {}).get(pid, "[NO RESPONSE]")
    clean = resp.replace("Ġ", " ").replace("Ċ", "\n")
    
    # Find answer after </think>
    think_end = clean.find("</think>")
    if think_end >= 0:
        answer = clean[think_end+8:].strip()
        thinking_len = think_end
    else:
        answer = clean.strip()
        thinking_len = 0
    
    short = m.replace("deepseek-r1-distill-qwen-32b", "ds-r1-32b")
    short = short.replace("qwen3.5-35b-a3b-fp8", "q3.5-35b-fp8")
    short = short.replace("qwen3.5-35b-a3b", "q3.5-35b")
    short = short.replace("qwen3.6-27b-fp8", "q3.6-27b-fp8")
    short = short.replace("qwen3.6-27b", "q3.6-27b")
    short = short.replace("qwen3-32b-fp8", "q3-32b-fp8")
    short = short.replace("qwen3-32b", "q3-32b")
    short = short.replace("qwen3-0.6b", "q3-0.6b")
    
    print(f"\n{'='*60}")
    print(f"MODEL: {short} | PROMPT: {pid}")
    if thinking_len > 0:
        print(f"[Thinking: {thinking_len} chars]")
    print(f"[Answer: {len(answer)} chars]")
    print(f"{'='*60}")
    print(answer[:2000])
    if len(answer) > 2000:
        print(f"\n... [truncated, {len(answer)} total chars]")
