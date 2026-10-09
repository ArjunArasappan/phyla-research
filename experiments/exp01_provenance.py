"""Export checkpoint/environment pins and verified immutable artifact ledger."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    repos={name:subprocess.check_output(['git','-C',str(root/'third_party'/folder),'rev-parse','HEAD'],text=True).strip() for name,folder in [('spatrackerv2','SpaTrackerV2'),('cotracker3','co-tracker'),('delta','DELTA')]}
    checkpoints={}
    for name,path in [('cotracker3',root/'checkpoints/cotracker3/scaled_offline.pth'),('delta',root/'checkpoints/delta/densetrack3d.pth')]:checkpoints[name]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    for name,folder in [('spatrackerv2','models--Yuxihenry--SpatialTrackerV2-Offline'),('frontend','models--Yuxihenry--SpatialTrackerV2_Front')]:
        hub=root/'checkpoints/hub'/folder;revision=(hub/'refs/main').read_text().strip();blob=max((hub/'blobs').glob('*'),key=lambda p:p.stat().st_size)
        checkpoints[name]={'revision':revision,'path':str(blob),'sha256':sha(blob),'bytes':blob.stat().st_size}
    files=[]
    for path in sorted((root/'data/exp01').rglob('*')):
        if not path.is_file() or 'QUARANTINED' in str(path) or path.suffix not in ['.npz','.json','.png'] or not (path.parent/'READY').exists():continue
        files.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':sha(path)})
    dataset=[]
    for file in sorted((root/'data/exp01').rglob('metadata.json')):
        if not (file.parent/'gt.npz').exists():continue
        meta=json.loads(file.read_text());bundle=file.parent/'gt.npz';assert sha(bundle)==meta['sha256']==(file.parent/'READY').read_text().strip();dataset.append({'clip':file.parent.name,'split':str(file.parent.parent.relative_to(root/'data/exp01')),'checks':meta['checks'],'sha256':meta['sha256']})
    patch=root/'worktrees/exp01/experiments/01-tracker-benchmark/ai_notes/spatrackerv2-sdpa-fail-closed.patch'
    provenance={'upstream_revisions':repos,'checkpoint_pins':checkpoints,'attention_patch_sha256':sha(patch),'python':sys.version,'hardware':'NVIDIA B300; Torch2.12cu130; isolated trackers overlay on verified sharedcore',
                'environment_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),'solver_pins':{'pycolmap':'3.11.1','pyceres':'2.4'},
                'immutable_dataset_validation':dataset,'files':files,'total_file_bytes':sum(f['bytes'] for f in files),'quarantine':'QUARANTINED_spatrackerv2_fp32_silent_attention_failure excluded; old invalid READY removed'}
    (out/'artifact-ledger.json').write_text(json.dumps(provenance,indent=2));print(json.dumps({'validated_dataset_clips':len(dataset),'files':len(files),'bytes':provenance['total_file_bytes']}))

if __name__=='__main__':main()
