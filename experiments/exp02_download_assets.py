import urllib.request,json,pathlib,hashlib,time,fcntl,os
BASE=pathlib.Path('/mnt/nvme/scratch/phyla-ubuntu/cache/checkpoints')
BASE.mkdir(parents=True,exist_ok=True)
def fetch(repo,rev,file,dest):
 dest=pathlib.Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 with open(str(dest)+'.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if dest.exists() and pathlib.Path(str(dest)+'.complete.json').exists():return
  url=f'https://huggingface.co/{repo}/resolve/{rev}/{file}'
  tmp=pathlib.Path(str(dest)+'.partial')
  with urllib.request.urlopen(url) as src,open(tmp,'wb') as out:
   while True:
    b=src.read(8*1024*1024)
    if not b:break
    out.write(b)
  digest=hashlib.sha256()
  with tmp.open('rb') as f:
   for b in iter(lambda:f.read(16*1024*1024),b''):digest.update(b)
  tmp.rename(dest)
  pathlib.Path(str(dest)+'.complete.json').write_text(json.dumps(dict(repo=repo,revision=rev,file=file,sha256=digest.hexdigest(),bytes=dest.stat().st_size),indent=2))
  print('COMPLETE',dest,flush=True)
from concurrent.futures import ThreadPoolExecutor
wan='921dbaf3f1674a56f47e83fb80a34bac8a8f203e';ltx='8984fa25007f376c1a299016d0957a37a2f797bb';config='e58e28c39631af4d1468ee57a853764e11c1d37e'
jobs=[('Wan-AI/Wan2.2-TI2V-5B',wan,'Wan2.2_VAE.pth',BASE/'wan22-ti2v5b'/wan/'Wan2.2_VAE.pth'),('Lightricks/LTX-Video',ltx,'ltxv-2b-0.9.6-dev-04-25.safetensors',BASE/'ltx096'/ltx/'ltxv-2b-0.9.6-dev-04-25.safetensors')]
for name in ['vae/config.json','transformer/config.json','scheduler/scheduler_config.json','text_encoder/config.json','tokenizer/tokenizer_config.json','tokenizer/special_tokens_map.json','tokenizer/spiece.model','tokenizer/added_tokens.json']:
 jobs.append(('Lightricks/LTX-Video-0.9.5',config,name,BASE/'ltx095-config'/config/name))
with ThreadPoolExecutor(max_workers=4) as pool:
 fs=[pool.submit(fetch,*job) for job in jobs]
 for f in fs:
  try:f.result()
  except Exception as e:print('FAILED',repr(e),flush=True)
