#!/usr/bin/env python3
"""CI performance gate.

Compares a fresh `fastframe --bench --json` result against the baseline recorded
for the last git tag (bench/baselines/<tag>.json). Fails (exit 1) when median FPS
drops by more than --threshold (default 5 %). Never auto-updates the baseline;
`make_baseline.py` does that on a tag push.
"""
import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

def last_tag() -> str | None:
    try:
        return subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="JSON produced by `fastframe --bench --json`")
    ap.add_argument("--threshold", type=float, default=0.05, help="allowed FPS drop fraction")
    ap.add_argument("--baseline", help="explicit baseline file (default: bench/baselines/<last tag>.json)")
    a = ap.parse_args()

    new = json.loads(pathlib.Path(a.results).read_text())
    base_path = pathlib.Path(a.baseline) if a.baseline else None
    if base_path is None:
        tag = last_tag()
        if tag is None:
            print("no git tag found -> no baseline to compare against; passing (first release)")
            return 0
        base_path = ROOT / "bench" / "baselines" / f"{tag}.json"
    if not base_path.exists():
        print(f"baseline {base_path} missing -> passing; create it with scripts/make_baseline.py")
        return 0
    base = json.loads(base_path.read_text())

    nf, bf = float(new["fps"]), float(base["fps"])
    delta = (nf - bf) / bf
    print(f"baseline {base_path.name}: {bf:.2f} fps   new: {nf:.2f} fps   delta: {delta*100:+.1f}%")
    print(f"p99 latency: baseline {base.get('p99_ms', '?')} ms  new {new.get('p99_ms', '?')} ms")
    if new.get("dropped", 0) != 0:
        print("FAIL: frames were dropped")
        return 1
    if nf < bf * (1.0 - a.threshold):
        print(f"FAIL: throughput regressed more than {a.threshold*100:.0f}%")
        return 1
    if nf > bf * (1.0 + a.threshold):
        print("improvement beyond threshold: consider tagging a new baseline")
    print("PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
