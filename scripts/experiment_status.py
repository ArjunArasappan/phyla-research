#!/usr/bin/env python3
"""Read-only monitoring for the three experiment workers and GPU jobs."""
import json, subprocess, argparse
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path('/mnt/nvme/scratch/phyla-ubuntu')
p=argparse.ArgumentParser();p.add_argument('--json',action='store_true');args=p.parse_args()
report={'observed_at':datetime.now(timezone.utc).isoformat(),'experiments':{}}
for name in ('exp01','exp02','exp03'):
 f=ROOT/'control'/name/'status.json'
 try: state=json.loads(f.read_text())
 except (OSError,ValueError) as e: state={'state':'status_unavailable','error':str(e)}
 jobs=[]
 for job_file in sorted((ROOT/'control'/name/'jobs').glob('*.json')):
  try:
   job=json.loads(job_file.read_text())
   jobs.append({'job':job.get('job',job_file.stem),'state':job.get('state','unknown'),'error':job.get('error'),'updated':job.get('updated',job.get('finished'))})
  except (OSError,ValueError):
   jobs.append({'job':job_file.stem,'state':'unreadable'})
 if jobs:
  state['job_counts']=dict(Counter(job['state'] for job in jobs))
  state['failed_jobs']=[job for job in jobs if job['state'] in ('failed','unreadable')]
 report['experiments'][name]=state
try:
 report['gpus']=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout.strip().splitlines()
except (OSError,subprocess.CalledProcessError) as e:report['gpu_error']=str(e)
ready=ROOT/'control/core-ready.json'
report['core_ready']=json.loads(ready.read_text()) if ready.exists() else False
if args.json:print(json.dumps(report,indent=2))
else:
 print('Observed:',report['observed_at'])
 print('Core GPU runtime:', 'READY' if report['core_ready'] else 'INSTALLING / NOT VERIFIED')
 for name,state in report['experiments'].items():
  print('\n'+name)
  for k,v in state.items():
   if isinstance(v, dict) or (isinstance(v, list) and len(str(v))>240):
    text=json.dumps(v)
    print(f'  {k}: {text[:220]}'+(' ... [use --json for full details]' if len(text)>220 else ''))
   else:print(f'  {k}: {v}')
 print('\nGPUs: index, used MiB, utilization %')
 print('\n'.join(report.get('gpus',[])))
