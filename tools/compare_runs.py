# tools/compare_runs.py
import pathlib, re
rows=[]
for p in pathlib.Path("runs").glob("*/summary.md"):
    run=p.parent.name
    m=open(p,encoding="utf-8").read()
    kv=dict(re.findall(r"-\s*([A-Za-z0-9_]+):\s*([0-9.]+)", m))
    rows.append((run, kv.get("accuracy",""), kv.get("f1_macro","")))
rows.sort()
out=["| run | accuracy | f1_macro |","|---|---:|---:|"]
out+= [f"| {r} | {a} | {f} |" for r,a,f in rows]
pathlib.Path("runs/REPORT.md").write_text("\n".join(out),encoding="utf-8")
print("Wrote runs/REPORT.md")
