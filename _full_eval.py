import json, os, zipfile, tempfile

runs_dir = "benchmark_data/runs"
test_sets_dir = "benchmark_data/test_sets"

# Load rubrics
rubrics = {}
for ts_file in os.listdir(test_sets_dir):
    if not ts_file.endswith(".json"):
        continue
    with open(os.path.join(test_sets_dir, ts_file), encoding="utf-8") as f:
        ts = json.load(f)
    for p in ts.get("prompts", []):
        rubrics[p["id"]] = {
            "text": p["text"][:200],
            "reference": p.get("reference_answer", ""),
            "category": p.get("category", ""),
        }

# Extract all responses
all_responses = {}  # {model: {prompt_id: response_text}}

for model_dir in sorted(os.listdir(runs_dir)):
    model_path = os.path.join(runs_dir, model_dir)
    if not os.path.isdir(model_path):
        continue
    
    all_responses[model_dir] = {}
    
    for zf_name in sorted(os.listdir(model_path)):
        if not zf_name.endswith(".zip"):
            continue
        
        zf_path = os.path.join(model_path, zf_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zf_path, 'r') as zf:
                zf.extractall(tmpdir)
            
            results_path = os.path.join(tmpdir, "results.jsonl")
            if not os.path.exists(results_path):
                continue
            
            with open(results_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    pid = r["prompt_id"]
                    txt = r.get("response_text") or ""
                    all_responses[model_dir][pid] = txt

# Write combined output for evaluation
output = []
for model in sorted(all_responses.keys()):
    for pid in sorted(all_responses[model].keys()):
        txt = all_responses[model][pid]
        ref = rubrics.get(pid, {})
        output.append({
            "model": model,
            "prompt_id": pid,
            "category": ref.get("category", ""),
            "response_length": len(txt),
            "response_preview": txt[:500],
            "has_content": bool(txt and txt.strip()),
            "reference_preview": (ref.get("reference", "") or "")[:200],
        })

# Summary
print(f"Total responses: {len(output)}")
print(f"Models: {sorted(all_responses.keys())}")
print()

# Check for empty responses
empty = [o for o in output if not o["has_content"]]
if empty:
    print(f"WARNING: {len(empty)} empty responses:")
    for e in empty:
        print(f"  {e['model']} / {e['prompt_id']}")
    print()

# Per model stats
for model in sorted(all_responses.keys()):
    responses = all_responses[model]
    total = len(responses)
    has_text = sum(1 for t in responses.values() if t and t.strip())
    avg_len = sum(len(t) for t in responses.values()) / total if total else 0
    print(f"{model:35s}: {has_text}/{total} with text, avg {avg_len:.0f} chars")

# Save full data for detailed review
with open("_eval_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nFull data saved to _eval_data.json")
