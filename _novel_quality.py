#!/usr/bin/env python3
"""
Extract ONLY the final translation output (not thinking/analysis) for novel prompts.
"""
import json, re

with open("_all_responses.json", encoding="utf-8") as f:
    data = json.load(f)

models = sorted(data.keys())

def clean(text):
    return text.replace("Ġ", " ").replace("Ċ", "\n")

def extract_translation(text):
    """Try to extract just the translation, not the analysis."""
    c = clean(text)
    idx = c.find("</think>")
    if idx >= 0:
        c = c[idx+8:].strip()
    
    # For models that show thinking process, find the actual translation
    # Look for patterns like "---\n" or "**Translation:**" or quoted text
    
    # Try to find translation after "---"
    parts = c.split("---")
    if len(parts) >= 2:
        # Take the part after first ---
        candidate = parts[1].strip()
        if len(candidate) > 100:
            return candidate.split("---")[0].strip()
    
    # Try to find after "Translation:" or "Here is"
    for marker in ["**Translation:**", "**Translation**", "Translation:", "Here is the translation", "Here's the translation"]:
        idx = c.find(marker)
        if idx >= 0:
            return c[idx+len(marker):].strip()
    
    # Try to find after "Here's" or "Certainly!"
    for marker in ["Here's the translated", "Certainly! Here", "Here is the translated"]:
        idx = c.find(marker)
        if idx >= 0:
            # Find the actual text after the intro
            rest = c[idx:]
            # Skip the intro line
            lines = rest.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('*') and not line.strip().startswith('#') and i > 0:
                    return '\n'.join(lines[i:]).strip()
    
    # For qwen3-0.6b style (direct translation without analysis)
    if not c.startswith("Okay") and not c.startswith("Here's a thinking") and not c.startswith("Alright") and not c.startswith("We need"):
        return c
    
    # For models with long analysis, try to find the actual translation block
    # Look for a paragraph that doesn't start with analysis words
    paragraphs = re.split(r'\n\n+', c)
    for p in paragraphs:
        p = p.strip()
        if (len(p) > 100 and 
            not p.startswith("Okay") and 
            not p.startswith("First") and
            not p.startswith("The ") and
            not p.startswith("I ") and
            not p.startswith("Let me") and
            not p.startswith("Now") and
            not p.startswith("Here's a thinking") and
            not p.startswith("*") and
            not p.startswith("#") and
            not p.startswith("So,") and
            not p.startswith("Alright")):
            return p
    
    return c[:500] + "..."

# Focus on key novel translations
focus = {
    "tr-001": "Lu Xun - road/hope (short)",
    "tr-006": "Romance - rain/tears (short)",
    "tr-011": "Historical fiction (long)",
    "tr-014": "Slice-of-life (long)",
    "tr-004": "Xianxia cultivation (short)",
}

for pid, label in focus.items():
    print(f"\n{'='*80}")
    print(f"{pid}: {label}")
    print(f"{'='*80}")
    
    for m in models:
        ans = extract_translation(data[m][pid])
        
        short = m.replace("deepseek-r1-distill-qwen-32b", "ds-r1-32b")
        short = short.replace("qwen3.5-35b-a3b-fp8", "q3.5-35b-fp8")
        short = short.replace("qwen3.5-35b-a3b", "q3.5-35b")
        short = short.replace("qwen3.6-27b-fp8", "q3.6-27b-fp8")
        short = short.replace("qwen3.6-27b", "q3.6-27b")
        short = short.replace("qwen3-32b-fp8", "q3-32b-fp8")
        short = short.replace("qwen3-32b", "q3-32b")
        short = short.replace("qwen3-0.6b", "q3-0.6b")
        
        is_refusal = "please provide" in ans.lower()[:100]
        
        print(f"\n  >>> {short} {'❌ REFUSED' if is_refusal else ''}")
        if not is_refusal:
            print(f"  {ans[:500]}")
        print()
