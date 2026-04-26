#!/usr/bin/env python3
"""Evaluate creative responses."""
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

def word_count_story(text):
    """Count words in the actual story part (skip thinking/analysis)"""
    # Remove markdown headers and thinking
    lines = text.split('\n')
    story_lines = []
    in_story = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('**') and stripped.endswith('**'):
            in_story = True
            continue
        if in_story or (not stripped.startswith('#') and not stripped.startswith('*') 
                       and 'word count' not in stripped.lower() and 'rubric' not in stripped.lower()):
            story_lines.append(stripped)
    story = ' '.join(story_lines)
    return len(story.split())

results = {}

for m in models:
    results[m] = {}
    
    # creative-001: short story 300-500 words, twist ending
    full = get_full(all_responses[m]["creative-001"])
    ans = get_answer(all_responses[m]["creative-001"])
    wc = word_count_story(ans)
    has_twist = any(t in full.lower() for t in ["twist", "reveal", "surprise", "realize", "truth", "actually", "all along"])
    has_narrative = len(ans) > 500  # substantial story
    s = 0
    if 200 <= wc <= 600: s += 1  # word count in range (generous)
    if has_twist: s += 3
    if has_narrative: s += 2
    s += 3  # quality baseline for attempting
    if len(ans) < 100: s = max(s - 3, 1)  # truncated
    results[m]["creative-001"] = min(s, 10)
    
    # creative-002: exactly 100 words micro-fiction
    full = get_full(all_responses[m]["creative-002"])
    ans = get_answer(all_responses[m]["creative-002"])
    # Find the actual story (not the thinking/analysis)
    # Look for a titled story or a paragraph that looks like fiction
    story_text = ""
    lines = ans.split('\n')
    in_story = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('**') and not stripped.startswith('**Word') and not stripped.startswith('**Result'):
            in_story = True
            continue
        if in_story and stripped and not stripped.startswith('*') and not stripped.startswith('#'):
            story_text += stripped + " "
    if not story_text:
        # Just use the whole answer
        story_text = ans
    
    wc = len(story_text.split())
    close_to_100 = abs(wc - 100) <= 10
    exact_100 = abs(wc - 100) <= 3
    has_emotion = any(t in full.lower() for t in ["tear", "cry", "heart", "love", "pain", "sacrifice"])
    s = 0
    if exact_100: s += 3
    elif close_to_100: s += 1
    if has_emotion: s += 2
    if len(story_text) > 200: s += 2  # world-building
    s += 2  # quality baseline
    if len(ans) < 50: s = 1  # too short
    results[m]["creative-002"] = min(s, 10)
    
    # creative-003: poem, 4 stanzas, ABAB
    full = get_full(all_responses[m]["creative-003"])
    ans = get_answer(all_responses[m]["creative-003"])
    # Count stanzas (separated by blank lines)
    stanzas = [s.strip() for s in re.split(r'\n\s*\n', ans) if s.strip() and not s.strip().startswith('#') and len(s.strip()) > 20]
    has_river = "river" in full.lower()
    has_personification = any(t in full.lower() for t in ["whisper", "dance", "sing", "speak", "call", "embrace", "carry", "weep"])
    has_simile = "like" in full.lower() or "as a" in full.lower()
    s = 0
    if len(stanzas) >= 4: s += 2
    if has_river: s += 2
    if has_personification: s += 1
    if has_simile: s += 1
    s += 3  # quality baseline
    if len(ans) < 100: s = max(s - 3, 1)
    results[m]["creative-003"] = min(s, 10)
    
    # creative-004: dialogue scene, 15-20 lines
    full = get_full(all_responses[m]["creative-004"])
    ans = get_answer(all_responses[m]["creative-004"])
    # Count dialogue lines
    dialogue_lines = len(re.findall(r'["""].*?["""]|".*?"', ans))
    has_stage = any(t in full.lower() for t in ["stage direction", "pause", "lean", "sip", "look", "glance", "nod", "shake"])
    has_tension = any(t in full.lower() for t in ["reluctan", "hesitat", "sigh", "silence", "pause"])
    has_coffee = "coffee" in full.lower() or "cup" in full.lower() or "café" in full.lower()
    s = 0
    if dialogue_lines >= 10: s += 2
    elif dialogue_lines >= 5: s += 1
    if has_stage: s += 1
    if has_tension: s += 2
    if has_coffee: s += 1
    s += 3  # quality baseline
    if len(ans) < 200: s = max(s - 2, 1)
    results[m]["creative-004"] = min(s, 10)
    
    # creative-005: story continuation, 200-300 words
    full = get_full(all_responses[m]["creative-005"])
    ans = get_answer(all_responses[m]["creative-005"])
    wc = len(ans.split())
    has_email = "email" in full.lower()
    has_mystery = any(t in full.lower() for t in ["future", "warning", "tomorrow", "work", "message"])
    has_hook = len(ans) > 300  # substantial continuation
    s = 0
    if 150 <= wc <= 400: s += 1
    if has_email: s += 2
    if has_mystery: s += 2
    if has_hook: s += 2
    s += 2  # quality baseline
    if len(ans) < 100: s = max(s - 3, 1)
    results[m]["creative-005"] = min(s, 10)

# Print results
print(f"{'Model':35s} | cr001 | cr002 | cr003 | cr004 | cr005 | AVG")
print("-" * 80)

for m in models:
    short = m.replace("deepseek-r1-distill-qwen-32b", "ds-r1-32b")
    short = short.replace("qwen3.5-35b-a3b-fp8", "q3.5-35b-fp8")
    short = short.replace("qwen3.5-35b-a3b", "q3.5-35b")
    short = short.replace("qwen3.6-27b-fp8", "q3.6-27b-fp8")
    short = short.replace("qwen3.6-27b", "q3.6-27b")
    short = short.replace("qwen3-32b-fp8", "q3-32b-fp8")
    short = short.replace("qwen3-32b", "q3-32b")
    short = short.replace("qwen3-0.6b", "q3-0.6b")
    
    vals = [results[m][f"creative-{i:03d}"] for i in range(1, 6)]
    avg = sum(vals) / len(vals)
    print(f"{short:35s} | {vals[0]:5d} | {vals[1]:5d} | {vals[2]:5d} | {vals[3]:5d} | {vals[4]:5d} | {avg:.1f}")
