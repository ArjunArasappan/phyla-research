"""Non-destructive persistent experiment index; links retain live job paths."""
from pathlib import Path
import argparse, json, os, time
BASE=Path('/workspace/phyla-research')
SAVED=Path('/workspace/phyla-research-preserved')
RUNS=Path('/workspace/phyla-research-runs')
PACKET=RUNS/'results/download'
REPO=SAVED/'repository'

def link(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():
        if os.readlink(path)==str(target):return
        path.unlink()
    elif path.exists():raise FileExistsError(f'Refusing to replace existing file: {path}')
    path.symlink_to(target,target_is_directory=target.is_dir())

def document(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)

def organize():
    BASE.mkdir(exist_ok=True)
    specs=[('01-tracking','01-tracker-benchmark','experiment-01','3D tracker benchmark'),
           ('02-vae','02-motion-vae-reconstruction','experiment-02','Motion-RGB VAE reconstruction'),
           ('03-policy','03-flow-vs-rgb-data-efficiency-ood','experiment-03','Flow versus RGB policy supervision')]
    for folder,old,tag,title in specs:
        exp=BASE/'experiments'/folder
        for part in ('reports','media','data'): (exp/part).mkdir(parents=True,exist_ok=True)
        ai=REPO/'experiments'/old/'results/ai'
        link(exp/'reports/archive-and-analysis',ai)
        link(exp/'reports/plan.md',SAVED/'outputs'/('experiment-01-tracker-benchmark.md' if tag=='experiment-01' else 'experiment-02-motion-vae-reconstruction.md' if tag=='experiment-02' else 'experiment-03-flow-vs-rgb-data-efficiency-ood.md'))
        link(exp/'reports/assets',PACKET/'assets')
        if tag!='experiment-03':
            link(exp/'reports/results.md',PACKET/(tag+'.md'))
            link(exp/'reports/results.html',PACKET/(tag+'.html'))
            link(exp/'media/interactive.html',PACKET/('tracker-3d-interactive.html' if tag=='experiment-01' else 'vae-interactive.html'))
            link(exp/'media/report-assets',PACKET/'assets'/tag)
            if tag=='experiment-01':link(exp/'media/scene-videos',ai/'textured64-pilot/videos')
            else:link(exp/'media/animations',ai/'animations')
            prefix='exp01' if tag=='experiment-01' else 'exp02'
            for archive in (SAVED/'archives').glob(prefix+'*.tar.gz'):link(exp/'data'/archive.name,archive)
            document(exp/'data/README.md','# Scientific data\n\nThe linked checksummed archives contain original scientific arrays, GT inputs and model outputs. They are compressed archives; no live datasets were moved.\n')
        else:
            portable=RUNS/'results/experiment-03-portable'
            report=portable if (portable/'README.md').exists() else RUNS/'results/experiment-03'
            link(exp/'reports/results.md',report/'README.md')
            if (portable/'index.html').exists():
                link(exp/'reports/results.html',portable/'index.html')
                for item in portable.iterdir():
                    if item.name not in ('README.md','index.html'):link(exp/'reports'/item.name,item)
            link(exp/'reports/live-results',RUNS/'results/experiment-03')
            link(exp/'reports/recorded-evaluations',RUNS/'exp03-recovery-root/runs/exp03/pilot')
            link(exp/'media/rollouts-and-forecasts',RUNS/'exp03-recovery-root/runs/exp03/pilot')
            link(exp/'data/demonstrations-and-training-cache',RUNS/'exp03-recovery-root/data/exp03/pushcube')
            link(exp/'data/original-core.tar.gz',SAVED/'archives/exp03-core.tar.gz')
            checkpoints=exp/'checkpoints';checkpoints.mkdir(exist_ok=True)
            for policy in (RUNS/'exp03-recovery-root/runs/exp03/pilot').glob('N*/policy'):link(checkpoints/policy.parent.name,policy)
            document(exp/'data/README.md','# Training data\n\n`demonstrations-and-training-cache/raw` contains the 64 demonstrations. Original `pilot_N*_prepared_v2` directories preserve metadata, windows, statistics and tracks. Regenerated tensors appear under `demonstrations-and-training-cache/debug-cache/ltx` after evaluation.\n')
        document(exp/'README.md',f'# Experiment {folder[:2]}: {title}\n\n- [Results Markdown](reports/results.md)\n- [Experimental plan](reports/plan.md)\n- [All preserved analysis](reports/archive-and-analysis/)\n- [Media](media/)\n- [Scientific data](data/)\n\n'+('- [Results HTML](reports/results.html)\n- [Interactive viewer](media/interactive.html)\n' if tag!='experiment-03' else '- [Learned adapters](checkpoints/)\n- [Live evaluation results](reports/recorded-evaluations/)\n')+'\nThese are links to the preserved originals. Existing compute paths remain valid.\n')
    downloads=BASE/'downloads';downloads.mkdir(exist_ok=True)
    link(downloads/'reports-and-viewers',PACKET)
    link(downloads/'experiment-results.zip',RUNS/'results/experiment-results-download.zip')
    infra=BASE/'infrastructure';infra.mkdir(exist_ok=True)
    link(infra/'source-code',REPO)
    link(infra/'model-assets',RUNS/'exp03-recovery-root/cache/checkpoints')
    link(infra/'scientific-archives',SAVED/'archives')
    logs=infra/'logs-and-status';logs.mkdir(exist_ok=True)
    for name in ('evaluation','training-cache','overnight'):
        link(logs/(name+'-status.json'),RUNS/(name+'-status.json'))
    for name in ('evaluation.log','training-cache.log','overnight-finisher.log'):link(logs/name,RUNS/name)
    document(BASE/'README.md','# Phyla experiments\n\nStart here on the GPU.\n\n- [Experiment 1: tracking](experiments/01-tracking/README.md)\n- [Experiment 2: VAE reconstruction](experiments/02-vae/README.md)\n- [Experiment 3: policy supervision](experiments/03-policy/README.md)\n- [Downloadable reports, viewers and ZIP](downloads/)\n- [Code, model assets, logs and status](infrastructure/)\n\nAll links stay on persistent `/workspace`. Original runtime directories are retained because active queues and checksummed policies reference them. This directory is the organized entry point.\n')
    broken=[str(p) for p in BASE.rglob('*') if p.is_symlink() and not p.exists()]
    if broken:raise RuntimeError(f'Broken links: {broken}')
    print(json.dumps({'organized_root':str(BASE),'broken_links':broken,'updated_unix':time.time()}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    organize()
    if args.watch:
        while True:
            status=json.loads((RUNS/'overnight-status.json').read_text())
            if status['phase'] in ('completed','failed'):
                organize();break
            time.sleep(60)
