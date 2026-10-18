"""Episode-weighted success over task_XX.json files: python -m efvla.aggregate <eval_dir>"""
import glob
import json
import sys

files = sorted(glob.glob(f"{sys.argv[1]}/task_*.json"))
s = e = 0
for f in files:
    r = json.load(open(f))
    s += r["successes"]
    e += r["episodes"]
    print(f"task {r['task_id']:2d} {r['successes']:3d}/{r['episodes']:3d}  {r['language']}")
print(f"TOTAL {s}/{e} = {100 * s / max(e, 1):.1f}%  ({len(files)} tasks)")
