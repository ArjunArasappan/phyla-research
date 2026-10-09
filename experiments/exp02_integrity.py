"""Export checksum ledger for data survival/backups; no GPU allocation."""
import pathlib,json,hashlib,numpy as np,time
base=pathlib.Path('/mnt/nvme/scratch/phyla-ubuntu');repo=base/'worktrees/exp02';roots=[base/'runs/exp02/simulator-pilot-v2',base/'runs/exp02/simulator-context33-v1'];records=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
for root in roots:
 for complete in sorted(root.glob('*/*/*/complete.json')):
  job=complete.parent;m=json.loads((job/'metrics.json').read_text());assert m['identity_max_error_m']<=1e-6
  with np.load(job/'reconstruction.npz') as d:
   assert np.isfinite(d['decoded_rgb_raw']).all(),str(job)
   assert np.isfinite(d['decoded_flow'][d['valid']]).all(),str(job)
  files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in job.iterdir() if p.is_file()}
  records.append(dict(root=str(root),job=str(job.relative_to(root)),input_signature=m['input_signature'],codec_signature=m['codec_signature'],files=files))
counts={str(root):sum(r['root']==str(root) for r in records) for root in roots};assert list(counts.values())==[168,24],counts
out=repo/'experiments/02-motion-vae-reconstruction/results/ai/artifact-manifest.json';out.write_text(json.dumps(dict(created_unix=time.time(),counts=counts,total_file_bytes=sum(f['bytes'] for r in records for f in r['files'].values()),records=records),indent=2));print(counts,out)
