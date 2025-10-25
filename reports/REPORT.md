@'
import pathlib, re

rows=[]
for p in pathlib.Path("runs").glob("*/summary.md"):
    run=p.parent.name
    m=open(p,encoding="utf-8").read()
    kv=dict(re.findall(r"-\s*([A-Za-z0-9_]+):\s*([0-9.]+)", m))
    rows.append((run, kv.get("accuracy",""), kv.get("f1_macro","")))

pathlib.Path("reports").mkdir(parents=True, exist_ok=True)

out=["| run | accuracy | f1_macro |","|---|---:|---:|"]
for r,a,f in sorted(rows):
    out.append(f"| {r} | {a} | {f} |")

report = "\n".join(out if rows else ["No runs found."])
pathlib.Path("reports/REPORT.md").write_text(report, encoding="utf-8")
print("Wrote reports/REPORT.md")
'@ | Set-Content tools\compare_runs.py -Encoding utf8
