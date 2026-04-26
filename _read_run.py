import json, os, zipfile, tempfile, sys

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-32b"
test_set = sys.argv[2] if len(sys.argv) > 2 else "coding"

runs_dir = f"benchmark_data/runs/{model}"
for zf_name in sorted(os.listdir(runs_dir)):
    if test_set in zf_name and zf_name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(os.path.join(runs_dir, zf_name), 'r') as zf:
                zf.extractall(tmpdir)
            with open(os.path.join(tmpdir, "results.jsonl")) as f:
                for line in f:
                    r = json.loads(line)
                    print(f"\n{'='*60}")
                    print(f"Prompt: {r['prompt_id']}")
                    print(f"{'='*60}")
                    txt = r.get("response_text") or "NO TEXT"
                    print(txt[:2000])
                    if len(txt) > 2000:
                        print(f"\n... ({len(txt)} chars total)")
        break
