#!/usr/bin/env python3
"""Record bench/baselines/<tag>.json from a results file (run by release.yml on tag push)."""
import json, pathlib, shutil, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
results = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results.json")
tag = sys.argv[2] if len(sys.argv) > 2 else subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], cwd=ROOT, text=True).strip()
dst = ROOT / "bench" / "baselines" / f"{tag}.json"
data = json.loads(results.read_text())
data["_tag"] = tag
data["_machine"] = subprocess.check_output(["uname", "-m"], text=True).strip()
try:
    cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].split(":", 1)[1].strip()
except Exception:
    cpu = "unknown"
data["_cpu"] = cpu
dst.write_text(json.dumps(data, indent=2) + "\n")
print(f"wrote {dst}")
