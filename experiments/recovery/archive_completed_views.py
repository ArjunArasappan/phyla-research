"""Archive completed Markdown/figures/videos; omit cells still being written."""
from pathlib import Path
import hashlib,json,zipfile
root=Path('/workspace/phyla-research-runs');report=root/'results/experiment-03';data=root/'exp03-recovery-root/runs/exp03/pilot'
rows=json.loads((report/'metrics.json').read_text())['results'];maps={};entries=[]
for r in rows:
 run=f'N{r["N"]:03d}_{r["arm"]}_seed{r["optimizer_seed"]}';c=r['condition'];p=data/run/('eval_'+c)
 for v in p.glob('*.mp4'):
  name=f'videos/{run}/{c}/{v.name}';maps[str(v)]=name;entries.append((v,name))
text=(report/'README.md').read_text()
for old,new in maps.items():text=text.replace(old,new)
dest=root/'results/experiment-03-completed-preview.zip';tmp=dest.with_suffix('.zip.partial')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 z.writestr('README.md',text)
 for p in report.iterdir():
  if p.is_file() and p.name!='README.md':z.write(p,p.name)
 z.write(data/'contract_audit.json','contract_audit.json')
 for p,name in entries:z.write(p,name)
tmp.replace(dest)
v={'archive':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'groups':len(rows),'videos':len(entries)}
dest.with_suffix('.zip.sha256.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
