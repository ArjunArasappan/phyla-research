"""Audit actual saved evaluation inputs/physical resets and counts; no fabricated scores."""
from pathlib import Path
import json,numpy as np,hashlib
ROOT=Path('/mnt/nvme/scratch/phyla-ubuntu');runs=json.loads((ROOT/'runs/exp03/pilot/manifest.json').read_text())['runs'];rows=[];errors=[];reference_states=None;reference_ledger=None;reference_cell=None;cross_policy_pairs=0
for r in runs:
 run=ROOT/'runs/exp03/pilot'/r['run'];states={};summaries={}
 audit=run/'lora_audit.json'
 if not audit.exists() or not json.loads(audit.read_text()).get('verified_after_training'):errors.append(r['run']+': missing frozen-base integrity audit')
 for condition in ('id','cube_color','camera30'):
  cell=run/('eval_'+condition)
  if not (cell/'summary.json').exists():continue
  cfg=json.loads((cell/'config.json').read_text());assert cfg['eval']['sampling_steps']==8 and cfg['eval']['horizon']==150 and cfg['eval']['episodes']==5 and cfg['eval']['execute_steps']==4
  assert cfg['eval']['motion_diagnostics'] is False and cfg['eval']['ood_condition']==condition
  ledger=json.loads((cell/'summary.json').read_text());summaries[condition]=ledger
  eps=[json.loads(x) for x in (cell/'episodes.jsonl').read_text().splitlines()];assert [x['seed'] for x in eps]==list(range(20000,20005))
  positions=[]
  for i in range(5):
   arr=np.load(cell/f'cohort_{i:06d}.npz');assert arr['rgb'].shape[0]==151 and arr['actions_native'].shape==(150,1,8)
   assert np.isfinite(arr['actions_native']).all() and np.isfinite(arr['object_positions_m']).all()
   positions.append(np.concatenate([arr['object_positions_m'][0,0],arr['tcp_positions_m'][0,0],arr['goal_positions_m'][0]]))
   forecast=np.load(cell/f'predicted_aux_{i:06d}.npz');assert forecast['actions_normalized'].dtype==np.float32
   for key in forecast.files:
    if key.endswith('_decoded_rgb'):assert forecast[key].dtype==np.float32 and np.isfinite(forecast[key]).all()
  states[condition]=np.stack(positions)
  if reference_states is None:
   reference_states=states[condition].copy();reference_ledger=ledger['ledger_id'];reference_cell=r['run']+'/'+condition
  else:
   cross_policy_pairs+=1
   if not np.array_equal(reference_states,states[condition]):errors.append(r['run']+'/'+condition+': initialphysicalstates differ from '+reference_cell)
   if ledger['ledger_id']!=reference_ledger:errors.append(r['run']+'/'+condition+': ledger differs from '+reference_cell)
  rows.append({'run':r['run'],'condition':condition,'N':r['N'],'arm':r['arm'],'optimizer_seed':r['optimizer_seed'],'successes':sum(x['success_at_end'] for x in eps),'trials':len(eps),'ledger_id':ledger['ledger_id']})
 if len(states)==3:
  for condition in ('cube_color','camera30'):
   if not np.array_equal(states['id'],states[condition]):errors.append(r['run']+': physical reset changed under '+condition)
  if len({x['ledger_id'] for x in summaries.values()})!=1:errors.append(r['run']+': paired ledger IDs differ')
result={'trained_policies':len(runs),'completed_condition_cells':len(rows),'requested_cells':36,'evaluated_rollouts':sum(x['trials'] for x in rows),'errors':errors,'reference_cell':reference_cell,'cross_policy_physical_reset_comparisons':cross_policy_pairs,'results':rows}
out=ROOT/'runs/exp03/pilot/contract_audit.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if errors:raise RuntimeError('Actual saved evaluation contract violations')
