"""Exact 16×16 material-point subset of textured64 clips for query-density control."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    axis=np.rint(np.linspace(0,63,16)).astype(int);rows,cols=np.meshgrid(axis,axis,indexing='ij');ids=(rows*64+cols).ravel()
    for file in sorted(Path(a.source).glob('*/gt.npz')):
        out=Path(a.output)/file.parent.name
        if (out/'READY').exists():continue
        d=np.load(file);arrays={}
        for key in d.files:
            value=d[key]
            if len(value.shape)>0 and value.shape[0]==4096:value=value[ids]
            elif len(value.shape)>1 and value.shape[1]==4096:value=value[:,ids]
            arrays[key]=value
        arrays['grid_shape']=np.array([16,16]);out.mkdir(parents=True,exist_ok=True);target=out/'gt.npz';np.savez_compressed(target,**arrays)
        meta=json.loads((file.parent/'metadata.json').read_text());meta.update(parent_gt_path=str(file),parent_gt_sha256=meta['sha256'],query_layout='16x16_exact_subset_of_dense64_grid',query_ids_preserve_dense_parent=True,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        meta['checks']['valid_queries']=int(arrays['initial_valid'].sum());(out/'metadata.json').write_text(json.dumps(meta,indent=2));(out/'READY').write_text(meta['sha256']+'\n')
        print(out,flush=True)

if __name__=='__main__':main()
