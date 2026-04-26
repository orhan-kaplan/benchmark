#!/usr/bin/env python3
"""
Evaluate coding responses - check for key implementation details.
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

def has_code(text):
    return "```" in text or "def " in text or "class " in text or "function " in text

results = {}

for m in models:
    results[m] = {}
    
    # code-001: flatten_nested_dict
    full = get_full(all_responses[m]["code-001"])
    has_recursive = "recursive" in full.lower() or "helper" in full.lower() or "def flatten" in full.lower()
    has_isinstance = "isinstance" in full
    has_dot = "." in full and ("join" in full or "f\"" in full or "+" in full)
    has_edge = "empty" in full.lower() or "{}" in full
    s = 0
    if has_recursive: s += 4
    if has_isinstance: s += 2
    if has_edge: s += 2
    if has_code(full): s += 2
    results[m]["code-001"] = min(s, 10)
    
    # code-002: retry_with_backoff decorator
    full = get_full(all_responses[m]["code-002"])
    has_decorator = "def decorator" in full or "def wrapper" in full or "@" in full
    has_backoff = "delay" in full and ("*= 2" in full or "*=2" in full or "double" in full.lower() or "* 2" in full)
    has_sleep = "time.sleep" in full
    has_wraps = "wraps" in full or "functools" in full
    has_raise = "raise" in full
    s = 0
    if has_decorator: s += 2
    if has_backoff: s += 2
    if has_sleep: s += 1
    if has_wraps: s += 1
    if has_raise: s += 2
    if has_code(full): s += 2
    results[m]["code-002"] = min(s, 10)
    
    # code-003: four_sum
    full = get_full(all_responses[m]["code-003"])
    has_sort = "sort" in full.lower()
    has_two_pointer = "pointer" in full.lower() or "left" in full.lower() and "right" in full.lower()
    has_dedup = "duplicate" in full.lower() or "skip" in full.lower()
    has_code_impl = "def four_sum" in full or "def fourSum" in full
    s = 0
    if has_sort: s += 2
    if has_two_pointer: s += 3
    if has_dedup: s += 2
    if has_code_impl: s += 2
    s += 1
    results[m]["code-003"] = min(s, 10)
    
    # code-004: LRU Cache
    full = get_full(all_responses[m]["code-004"])
    has_dll = "doubly" in full.lower() or "linked list" in full.lower() or "prev" in full
    has_dict = "dict" in full.lower() or "cache" in full.lower() or "hash" in full.lower()
    has_node = "class Node" in full or "class ListNode" in full or "node" in full.lower()
    has_evict = "evict" in full.lower() or "remove" in full.lower()
    s = 0
    if has_dll: s += 3
    if has_dict: s += 2
    if has_node: s += 2
    if has_evict: s += 2
    s += 1
    results[m]["code-004"] = min(s, 10)
    
    # code-005: merge_sorted_lists bug
    full = get_full(all_responses[m]["code-005"])
    has_bug_id = "extend" in full or "remaining" in full.lower() or "left over" in full.lower() or "append" in full.lower()
    has_fix = "result.extend" in full or "result +=" in full or "result +" in full
    has_explain = "while loop" in full.lower() or "exhausted" in full.lower() or "one list" in full.lower()
    s = 0
    if has_bug_id: s += 4
    if has_fix: s += 3
    if has_explain: s += 2
    s += 1
    results[m]["code-005"] = min(s, 10)
    
    # code-006: async race condition
    full = get_full(all_responses[m]["code-006"])
    has_race = "race" in full.lower() or "concurrent" in full.lower()
    has_lock = "asyncio.Lock" in full or "Lock()" in full
    has_explain = "read" in full.lower() and "write" in full.lower() or "same value" in full.lower()
    s = 0
    if has_race: s += 4
    if has_lock: s += 3
    if has_explain: s += 2
    s += 1
    results[m]["code-006"] = min(s, 10)
    
    # code-007: code review (pickle, file handles, etc.)
    full = get_full(all_responses[m]["code-007"])
    issues = 0
    if "pickle" in full.lower() and ("security" in full.lower() or "unsafe" in full.lower() or "arbitrary" in full.lower()): issues += 1
    if "close" in full.lower() or "with" in full.lower(): issues += 1
    if "os.path.join" in full or "path" in full.lower() and "join" in full.lower(): issues += 1
    if "error" in full.lower() and "handling" in full.lower(): issues += 1
    if "filter" in full.lower() or ".DS_Store" in full or "file type" in full.lower(): issues += 1
    if "heapq" in full or "performance" in full.lower(): issues += 1
    s = min(issues * 2, 8) + 2
    results[m]["code-007"] = min(s, 10)
    
    # code-008: FastAPI review
    full = get_full(all_responses[m]["code-008"])
    issues = 0
    if "sql injection" in full.lower() or "injection" in full.lower(): issues += 1
    if "connection" in full.lower() and ("pool" in full.lower() or "thread" in full.lower() or "global" in full.lower()): issues += 1
    if "error" in full.lower() and ("handling" in full.lower() or "404" in full): issues += 1
    if "auth" in full.lower(): issues += 1
    if "pydantic" in full.lower() or "model" in full.lower(): issues += 1
    if "parameterized" in full.lower() or "?" in full: issues += 1
    s = min(issues * 2, 8) + 2
    results[m]["code-008"] = min(s, 10)
    
    # code-009: debounce + throttle (JavaScript)
    full = get_full(all_responses[m]["code-009"])
    has_debounce = "clearTimeout" in full or "setTimeout" in full
    has_throttle = "throttle" in full.lower()
    has_ts = "TypeScript" in full or ": (" in full or "<T" in full or "Parameters<" in full
    has_both = has_debounce and has_throttle
    s = 0
    if has_debounce: s += 3
    if has_throttle: s += 3
    if has_ts: s += 2
    if has_code(full): s += 2
    results[m]["code-009"] = min(s, 10)
    
    # code-010: useLocalStorage React hook
    full = get_full(all_responses[m]["code-010"])
    has_ssr = "typeof window" in full or "server" in full.lower()
    has_storage_event = "storage" in full.lower() and "event" in full.lower()
    has_error = "try" in full and "catch" in full
    has_generic = "<T>" in full or "<T," in full
    has_usestate = "useState" in full
    s = 0
    if has_ssr: s += 2
    if has_storage_event: s += 2
    if has_error: s += 2
    if has_generic: s += 1
    if has_usestate: s += 2
    s += 1
    results[m]["code-010"] = min(s, 10)

# Print results
print(f"{'Model':35s}", end="")
for i in range(1, 11):
    print(f" | c{i:03d}", end="")
print(" | AVG")
print("-" * 120)

for m in models:
    short = m.replace("deepseek-r1-distill-qwen-32b", "ds-r1-32b")
    short = short.replace("qwen3.5-35b-a3b-fp8", "q3.5-35b-fp8")
    short = short.replace("qwen3.5-35b-a3b", "q3.5-35b")
    short = short.replace("qwen3.6-27b-fp8", "q3.6-27b-fp8")
    short = short.replace("qwen3.6-27b", "q3.6-27b")
    short = short.replace("qwen3-32b-fp8", "q3-32b-fp8")
    short = short.replace("qwen3-32b", "q3-32b")
    short = short.replace("qwen3-0.6b", "q3-0.6b")
    
    vals = [results[m][f"code-{i:03d}"] for i in range(1, 11)]
    avg = sum(vals) / len(vals)
    print(f"{short:35s}", end="")
    for v in vals:
        print(f" | {v:4d}", end="")
    print(f" | {avg:.1f}")
