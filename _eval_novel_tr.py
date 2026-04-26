#!/usr/bin/env python3
"""
Detailed comparison of novel/web_novel translation quality.
Novel: tr-001, tr-002, tr-003, tr-011, tr-014, tr-015
Web Novel: tr-004, tr-005, tr-006, tr-012, tr-013
"""
import json

with open("_all_responses.json", encoding="utf-8") as f:
    data = json.load(f)

models = sorted(data.keys())

def clean(text):
    return text.replace("Ġ", " ").replace("Ċ", "\n")

def get_answer(text):
    c = clean(text)
    idx = c.find("</think>")
    if idx >= 0:
        return c[idx+8:].strip()
    return c.strip()

novel_ids = ["tr-001", "tr-002", "tr-003", "tr-011", "tr-014", "tr-015"]
web_novel_ids = ["tr-004", "tr-005", "tr-006", "tr-012", "tr-013"]

novel_labels = {
    "tr-001": "Lu Xun (road/hope)",
    "tr-002": "Gatsby-style (dream/night)",
    "tr-003": "Dickens-style (best/worst times)",
    "tr-011": "Historical fiction LONG (tax/politics)",
    "tr-014": "Slice-of-life LONG (small town)",
    "tr-015": "Dialogue/banter LONG (cultivation)",
}

web_novel_labels = {
    "tr-004": "Xianxia cultivation (breakthrough)",
    "tr-005": "Fantasy action (sword/blood)",
    "tr-006": "Romance (rain/tears)",
    "tr-012": "Martial arts LONG (fist/evil)",
    "tr-013": "Xianxia LONG (secret realm/flowers)",
}

all_tr = novel_ids + web_novel_ids
all_labels = {**novel_labels, **web_novel_labels}

for pid in all_tr:
    print(f"\n{'='*80}")
    print(f"{pid}: {all_labels[pid]}")
    print(f"{'='*80}")
    
    for m in models:
        ans = get_answer(data[m][pid])
        full = clean(data[m][pid])
        
        short = m.replace("deepseek-r1-distill-qwen-32b", "ds-r1-32b")
        short = short.replace("qwen3.5-35b-a3b-fp8", "q3.5-35b-fp8")
        short = short.replace("qwen3.5-35b-a3b", "q3.5-35b")
        short = short.replace("qwen3.6-27b-fp8", "q3.6-27b-fp8")
        short = short.replace("qwen3.6-27b", "q3.6-27b")
        short = short.replace("qwen3-32b-fp8", "q3-32b-fp8")
        short = short.replace("qwen3-32b", "q3-32b")
        short = short.replace("qwen3-0.6b", "q3-0.6b")
        
        is_refusal = "please provide" in ans.lower()[:100]
        
        print(f"\n  [{short:15s}] ({len(ans):5d} chars) {'❌ REFUSED' if is_refusal else ''}")
        if not is_refusal:
            # Show first 400 chars of actual translation
            print(f"  {ans[:400]}")
            if len(ans) > 400:
                print(f"  ... [{len(ans)} total chars]")
