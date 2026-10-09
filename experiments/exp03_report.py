"""Report only measured runs; missing cells remain missing."""
from __future__ import annotations
import csv,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path("/mnt/nvme/scratch/phyla-ubuntu");REPO=pathlib.Path(__file__).resolve().parents[1]
OUT=REPO/"experiments/03-flow-vs-rgb-data-efficiency-ood/results/ai";OUT.mkdir(parents=True,exist_ok=True)
def main():
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 import imageio.v2 as imageio
 manifest=json.loads((ROOT/"runs/exp03/pilot/manifest.json").read_text());runs=manifest["runs"];results=[];training=[]
 for r in runs:
  run=ROOT/"runs/exp03/pilot"/r["run"]
  if (run/"metrics.jsonl").exists():
   metrics=[json.loads(x) for x in (run/"metrics.jsonl").read_text().splitlines()]
   training.append({**r,"completed_updates":metrics[-1]["step"],"total_update_seconds":sum(x["seconds"] for x in metrics),"last_action_loss":metrics[-1]["action"],"frozen_base_verified":json.loads((run/"lora_audit.json").read_text()).get("verified_after_training",False) if (run/"lora_audit.json").exists() else False})
  for condition in ("id","cube_color","camera30"):
   ep=run/("eval_"+condition)
   if not (ep/"summary.json").exists():continue
   summary=json.loads((ep/"summary.json").read_text());task=summary["tasks"]["PushCube-v1"]
   result={"run":r["run"],"N":r["N"],"arm":r["arm"],"optimizer_seed":r["optimizer_seed"],"condition":condition,"successes":task["successes"],"trials":task["trials"],"success":task["success_at_end"],"planning_p50_seconds":summary["planning_p50_seconds"],"ledger_id":summary["ledger_id"],"artifact":str(ep)};results.append(result)
   if r["optimizer_seed"]==0 and r["N"]==16:
    arrays=np.load(ep/"cohort_000000.npz")
    frames=arrays["rgb"][:,0,0];imageio.imwrite(OUT/(r["arm"]+"_"+condition+"_rollout.png"),np.concatenate(frames[np.linspace(0,len(frames)-1,5).astype(int)],axis=1))
    pred=np.load(ep/"predicted_aux_000000.npz")
    key=("video" if r["arm"]=="R" else "flow")+"_decoded_rgb"
    if r["arm"]!="A" and key in pred:
     frames=np.clip(pred[key],0,1);imageio.imwrite(OUT/(r["arm"]+"_"+condition+"_forecast.png"),np.concatenate((frames[[0,3,7,11,15]]*255).astype(np.uint8),axis=1))
 (OUT/"pilot_metrics.json").write_text(json.dumps({"results":results,"training":training,"protocol":manifest},indent=2))
 if results:
  with open(OUT/"pilot_metrics.csv","w") as f:
   writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
  fig,axes=plt.subplots(1,3,figsize=(11,3.3),sharey=True)
  colors={"A":"#444444","R":"#c7671c","F-GT":"#1473b5"}
  for ax,condition in zip(axes,("id","cube_color","camera30")):
   for arm in ("A","R","F-GT"):
    cells=[r for r in results if r["arm"]==arm and r["condition"]==condition]
    ns=sorted(set(r["N"] for r in cells));means=[np.mean([r["success"] for r in cells if r["N"]==n]) for n in ns]
    ax.plot(ns,means,"o-",color=colors[arm],label=arm)
    ax.scatter([r["N"] for r in cells],[r["success"] for r in cells],marker="x",color=colors[arm],alpha=.5)
   ax.set(title=condition,xlabel="Unique demonstrations",ylim=(-.03,1.03));ax.set_xscale("log",base=2);ax.set_xticks([16,64],labels=[16,64]);ax.grid(alpha=.2)
  axes[0].set_ylabel("Closed-loop success");axes[-1].legend();fig.suptitle("Exploratory pilot: 5 paired resets / condition / optimizer seed; missing cells omitted");fig.tight_layout();fig.savefig(OUT/"pilot_success.png",dpi=180);plt.close(fig)
  # Same world-space limits for all displayed object/TCP trajectories.
  selected=[r for r in results if r["N"]==16 and r["optimizer_seed"]==0]
  fig,axes=plt.subplots(3,3,figsize=(9,8),sharex=True,sharey=True)
  for r in selected:
   arr=np.load(pathlib.Path(r["artifact"])/"cohort_000000.npz")
   if "object_positions_m" not in arr:continue
   ax=axes[("A","R","F-GT").index(r["arm"]),("id","cube_color","camera30").index(r["condition"])]
   obj=arr["object_positions_m"][:,0];tcp=arr["tcp_positions_m"][:,0];goal=arr["goal_positions_m"][0]
   ax.plot(obj[:,0],obj[:,1],color="#1473b5",label="cube");ax.plot(tcp[:,0],tcp[:,1],color="#70a98e",label="TCP");ax.scatter(goal[0],goal[1],marker="*",color="#c7671c",s=70);ax.set_title(r["arm"]+" / "+r["condition"]);ax.grid(alpha=.2);ax.set(xlabel="world x (m)",ylabel="world y (m)")
  fig.suptitle("Matched initial physical state; trajectories may diverge under each policy");fig.tight_layout();fig.savefig(OUT/"pilot_trajectories.png",dpi=180);plt.close(fig)
 lines=["# Experiment 3 measured pilot results","","2026-10-08 (America/Los_Angeles). This file contains actual completed cells only.","",f"Started training records: {len(training)}/12; completed ID/OOD condition cells: {len(results)}/36.","","1000 updates, effective batch4, frozen LTX2B with rank16 LoRA + action interfaces. N16/N64 nested episodes; two optimizer seeds; five paired resets per condition. Fixed final checkpoint. No future RGB/flow enters runtime observations.","","| N | Arm | Optimizer seed | Condition | Success |","|---|---|---|---|---|"]
 for r in results:lines.append("| {N} | {arm} | {optimizer_seed} | {condition} | {successes}/{trials} |".format(**r))
 lines.extend(["","Small pilot counts and a single task do not establish data efficiency or broad OOD generalization. A small OOD gap can reflect uniformly unsuccessful policies. Prediction-domain losses are not compared across RGB and flow.","","Protocol deviations: native8D joint-position control, RGB+instruction without proprioception, one color shift and one camera30 shift, shortened budget, one subset seed. GT labels are privileged training supervision. VAE/text/base transformer remain frozen.","","[Measured cells CSV](pilot_metrics.csv) · [Full metrics and protocol JSON](pilot_metrics.json)"])
 if results:lines.extend(["","![Measured success](pilot_success.png)","","![World-space trajectories](pilot_trajectories.png)"])
 (OUT/"2026-10-08-pilot-results.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"training_records":len(training),"evaluated_conditions":len(results),"output":str(OUT)}))
if __name__=="__main__":main()
