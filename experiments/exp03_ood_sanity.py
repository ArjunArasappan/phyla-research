import sys,pathlib,json,numpy as np,imageio.v2 as imageio
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from flomo.config import load_config
from flomo.sim import ManiSkillAdapter
from mani_skill.examples.motionplanning.panda.solutions.push_cube import solve
import gymnasium as gym
out=pathlib.Path("/mnt/nvme/scratch/phyla-ubuntu/runs/exp03/ood_sanity");out.mkdir(exist_ok=True)
rows=[];images=[];states=[]
for condition in ["id","cube_color","camera30"]:
 cfg=load_config("experiments/03-flow-vs-rgb-data-efficiency-ood/configs/N016_F-GT_seed0.yaml");cfg.eval.ood_condition=condition
 adapter=ManiSkillAdapter(cfg);rgb=adapter.reset([20000]);env=adapter.env.unwrapped
 state=np.concatenate([adapter.numpy(env.obj.pose.p).ravel(),adapter.numpy(env.goal_region.pose.p).ravel(),adapter.numpy(env.agent.robot.get_qpos()).ravel()]);states.append(state);images.append(rgb[0,0]);imageio.imwrite(out/(condition+".png"),rgb[0,0])
 class Reset(gym.Wrapper):
  def reset(self,seed=None,**kw):adapter.reset([seed]);return adapter.obs,{}
 result=solve(Reset(adapter.env),seed=20000,vis=False,debug=False)
 rows.append({"condition":condition,"seed":20000,"expert_success":result!=-1 and bool(result[-1]["success"].item()),"mean_abs_pixel_change_from_id":float(np.abs(rgb[0,0].astype(float)-images[0].astype(float)).mean()),"physical_reset_equal_id":bool(np.array_equal(state,states[0]))});adapter.close()
 print(json.dumps(rows[-1]),flush=True)
assert all(x["physical_reset_equal_id"] for x in rows)
assert all(x["expert_success"] for x in rows)
assert all(x["mean_abs_pixel_change_from_id"]>0 for x in rows[1:])
imageio.imwrite(out/"paired_resets.png",np.concatenate(images,axis=1));(out/"results.json").write_text(json.dumps(rows,indent=2))
np.savez_compressed(out/"paired_resets.npz",rgb=np.stack(images),states=np.stack(states))
