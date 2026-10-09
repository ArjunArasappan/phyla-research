from pathlib import Path
import concurrent.futures, hashlib, json, tarfile, shutil, urllib.request, os
ROOT=Path('/workspace/phyla-research-runs/exp03-recovery-root');ROOT.mkdir(parents=True,exist_ok=True)
legacy=Path('/mnt/nvme/scratch/phyla-ubuntu');legacy.parent.mkdir(parents=True,exist_ok=True)
if not legacy.exists():legacy.symlink_to(ROOT,target_is_directory=True)
archive=Path('/workspace/phyla-research-preserved/archives/exp03-core.tar.gz')
with tarfile.open(archive,'r|gz') as tf:
 for member in tf:
  if not member.isfile():continue
  rel=Path(member.name)
  assert not rel.is_absolute() and '..' not in rel.parts
  dest=ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():
   with tf.extractfile(member) as src, dest.open('wb') as out:shutil.copyfileobj(src,out,256*1024)
print('RESTORED CORE',flush=True)
base=ROOT/'cache/checkpoints';rev='8984fa25007f376c1a299016d0957a37a2f797bb';ar='e58e28c39631af4d1468ee57a853764e11c1d37e';assets=base/'ltx095-config'/ar
listing=json.load(urllib.request.urlopen('https://huggingface.co/api/models/Lightricks/LTX-Video-0.9.5/revision/'+ar))
files=[x['rfilename'] for x in listing['siblings'] if x['rfilename'].startswith(('text_encoder/','tokenizer/')) or x['rfilename'] in ('transformer/config.json','vae/config.json')]
jobs=[('Lightricks/LTX-Video-0.9.5',ar,f,assets/f) for f in files]
jobs.append(('Lightricks/LTX-Video',rev,'ltxv-2b-0.9.6-dev-04-25.safetensors',base/'ltx096'/rev/'ltxv-2b-0.9.6-dev-04-25.safetensors'))
def fetch(job):
 repo,r,name,dest=job;dest.parent.mkdir(parents=True,exist_ok=True)
 marker=Path(str(dest)+'.complete.json')
 if dest.exists() and marker.exists():return
 tmp=Path(str(dest)+'.partial')
 with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{r}/{name}',timeout=120) as src:
  expected=int(src.headers.get('Content-Length',0))
  if not (expected and tmp.exists() and tmp.stat().st_size==expected):
   with tmp.open('wb') as out:shutil.copyfileobj(src,out,256*1024)
 h=hashlib.sha256()
 with tmp.open('rb') as f:
  for b in iter(lambda:f.read(256*1024),b''):h.update(b)
 tmp.replace(dest);marker.write_text(json.dumps({'repo':repo,'revision':r,'file':name,'bytes':dest.stat().st_size,'sha256':h.hexdigest()},indent=2))
 print('DOWNLOADED',name,dest.stat().st_size,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
 for x in pool.map(fetch,jobs):pass
print('DOWNLOADS COMPLETE',flush=True)
