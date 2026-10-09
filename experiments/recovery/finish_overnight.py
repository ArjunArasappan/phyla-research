"""Unattended postprocessing of the existing evaluation/cache queues.

Starts no evaluation or training. Saves failures and completion on /workspace.
"""
from pathlib import Path
import csv, fcntl, hashlib, html, json, os, re, shutil, subprocess, sys, time, zipfile
import mistune

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/workspace/phyla-research-runs')
RESULTS = ROOT/'results'
HELPERS = REPO/'experiments/recovery'
lock = open(ROOT/'overnight-finisher.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def status(phase, **fields):
    path=ROOT/'overnight-status.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**fields),indent=2))
    temp.replace(path)
    print(path.read_text(),flush=True)

def run(*args):
    subprocess.run(args,cwd=REPO,check=True)

def wait_for(name):
    status('waiting',queue=name)
    while True:
        value=json.loads((ROOT/f'{name}-status.json').read_text())
        if value['phase']=='completed':return value
        if value['phase']=='failed':raise RuntimeError(f'{name} failed: {value}')
        try:os.kill(value['pid'],0)
        except ProcessLookupError:raise RuntimeError(f'{name} process stopped before completion: {value}')
        time.sleep(60)

render=mistune.create_markdown(plugins=['table'])
def copy_tree(source,target):
    # Persistent volume supports file bytes, not directory chmod/copystat.
    for path in source.rglob('*'):
        dest=target/path.relative_to(source)
        if path.is_dir():dest.mkdir(parents=True,exist_ok=True)
        elif path.is_file():
            dest.parent.mkdir(parents=True,exist_ok=True)
            with path.open('rb') as inf,dest.open('wb') as outf:shutil.copyfileobj(inf,outf,256*1024)

def webpage(text):
    body=render(text)
    body=re.sub(r'<a href="([^"]+\.mp4)">([^<]+)</a>',r'<a href="\1">\2</a><video controls preload="none" src="\1"></video>',body)
    return '<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:30px auto;font:16px/1.5 system-ui}img,video{max-width:100%}th,td{border:1px solid #ccc;padding:6px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'+body

try:
    evaluated=wait_for('evaluation')
    assert evaluated['completed_cells']==36
    status('auditing_and_reporting')
    for script in ('check_eval_contracts.py','report_evaluations.py','archive_completed_views.py'):
        run(sys.executable,str(HELPERS/script))
    archive=RESULTS/'experiment-03-completed-preview.zip'
    expected=json.loads(archive.with_suffix('.zip.sha256.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected['sha256']
    assert expected['groups']==36 and expected['videos']==180
    cohort=RESULTS/'experiment-03-portable'
    cohort.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:z.extractall(cohort)
    (cohort/'index.html').write_text(webpage((cohort/'README.md').read_text()))
    cached=wait_for('training-cache')
    assert cached['validated'] and cached['requested_windows']==398
    status('saving_cache_manifests')
    cache_report=cohort/'training-cache'
    cache_report.mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'training-cache-status.json',cache_report/'status.json')
    for n in (16,64):
        source=ROOT/f'exp03-recovery-root/data/exp03/pushcube/debug-cache/ltx/N{n:03d}'
        target=cache_report/f'N{n:03d}'
        target.mkdir(exist_ok=True)
        for name in ('dataset.json','windows.jsonl','stats.json','sample-hashes.json','rebuilding.json'):
            shutil.copyfile(source/name,target/name)
    text='# Reconstructed training caches\n\nValidated N016: 80 windows; N064: 318 windows. Shared conditioning, text, actions, RGB and flow tensors. New hashes and rebuild provenance distinguish them from historical samples. Full tensors remain on `/workspace/phyla-research-runs/exp03-recovery-root/data/exp03/pushcube/debug-cache/ltx`.\n'
    (cache_report/'README.md').write_text(text)
    (cache_report/'index.html').write_text(webpage(text))
    status('packaging_downloads')
    bundle=RESULTS/'download'
    dest=bundle/'experiment-03-final'
    copy_tree(cohort,dest)
    # Replace only Experiment 3 entries in the existing Experiment 1/2 packet.
    for suffix in ('md','html'):
        path=bundle/f'index.{suffix}'
        text=path.read_text().replace('experiment-03.html','experiment-03-final/index.html').replace('experiment-03.md','experiment-03-final/README.md')
        text=text.replace('Experiment3 remains in progress; this snapshot contains completed groups only.','Experiment 3 completed: 36 groups, 180 rollouts; both debugging caches validated.')
        path.write_text(text)
    records=[]
    for path in bundle.rglob('*'):
        if path.is_file() and path.name!='manifest.json':
            records.append(dict(path=str(path.relative_to(bundle)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (bundle/'manifest.json').write_text(json.dumps(records,indent=2))
    packet=RESULTS/'experiment-results-download.zip'
    temp=packet.with_suffix('.zip.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for path in bundle.rglob('*'):
            if path.is_file():z.write(path,str(Path('experiment-results-download')/path.relative_to(bundle)))
    temp.replace(packet)
    packet_hash=hashlib.sha256(packet.read_bytes()).hexdigest()
    packet.with_suffix('.zip.sha256').write_text(packet_hash+'\n')
    status('publishing_compact_results')
    run('git','fetch','origin','main')
    run('git','merge','--ff-only','origin/main')
    target=REPO/'experiments/03-flow-vs-rgb-data-efficiency-ood/results/ai/recovery-cohort'
    copy_tree(cohort,target)
    run('git','add',str(target.relative_to(REPO)))
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode
    if changed:
        run('git','commit','-m','Save completed recovery evaluations, videos and debugging cache manifests')
        run('git','push','origin','main')
    run('rsync','-r','--no-perms','--no-owner','--no-group','--exclude','/runs',str(REPO)+'/', '/workspace/phyla-research-preserved/repository/')
    status('completed',groups=36,rollouts=180,cache_windows=398,download=str(packet),sha256=packet_hash,git_pushed=True)
except Exception as error:
    status('failed',error=repr(error))
    raise
