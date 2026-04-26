#!/usr/bin/env python3
"""
Read ALL responses from benchmark zips and output full text for scoring.
Outputs one big JSON: {model: {prompt_id: full_response_text}}
"""
import json, os, zipfile, tempfile

runs_dir = "benchmark_data/runs"
all_responses = {}

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

# Save full responses
with open("_all_responses.json", "w", encoding="utf-8") as f:
    json.dump(all_responses, f, indent=2, ensure_ascii=False)

# Print summary
models = sorted(all_responses.keys())
print(f"Models: {len(models)}")
for m in models:
    print(f"  {m}: {len(all_responses[m])} responses")
print(f"\nSaved to _all_responses.json")
