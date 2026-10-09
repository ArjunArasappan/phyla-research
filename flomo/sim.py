"""CPU toy integration environment and lazy ManiSkill collection/evaluation.

Toy results are pipeline checks, never robot benchmarks. Native simulator state
is available to collection/diagnostics only; policy requests contain RGB/text.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import time

import numpy as np
import torch

from .config import Config, atomic_json, digest
from .data import save_episode, write_jsonl, read_jsonl
from .geometry import oracle_tracks, canonical_flow, PercentileStats
from .policy import BundlePolicy, NativeActionCodec, ChunkExecutor, wilson


class ToyAdapter:
    def __init__(self,config):
        self.config=config; self.n=config.eval.num_envs; self.size=config.model.image_size
        self.action_dim=2; self.low=np.full(2,-1.); self.high=np.ones(2)
        if config.model.views!=1 or config.model.action_dim!=2: raise ValueError("Toy requires one camera and 2D native actions")

    def reset(self,seeds):
        self.position=np.stack([np.random.default_rng(seed).uniform(-.7,.7,2) for seed in seeds])
        self.target=np.stack([np.random.default_rng(seed+100000).uniform(-.6,.6,2) for seed in seeds])
        self.steps=0; return self.observation()

    def observation(self):
        images=[]
        for p,g in zip(self.position,self.target):
            image=np.full((self.size,self.size,3),30,np.uint8)
            def draw(pos,color,r):
                x,y=np.rint((pos+1)*.5*(self.size-1)).astype(int)
                image[max(0,y-r):min(self.size,y+r+1),max(0,x-r):min(self.size,x+r+1)]=color
            draw(g,[40,200,40],2); draw(p,[220,50,40],2); images.append(image)
        return np.stack(images)[:,None]

    def step(self,action):
        action=np.asarray(action)
        if action.shape!=(self.n,2) or not np.isfinite(action).all(): raise ValueError("Bad toy controller action")
        self.position=np.clip(self.position+np.clip(action,-1,1)*.05,-.95,.95); self.steps+=1
        success=np.linalg.norm(self.position-self.target,axis=-1)<.08
        return self.observation(),{"success":success,"reward":-np.linalg.norm(self.position-self.target,axis=-1)}

    def expert_action(self): return np.clip((self.target-self.position)/.05,-1,1)

    def geometry(self):
        snapshots=[]
        for p in self.position:
            h=self.size; yy,xx=np.mgrid[:h,:h]; x=(xx/(h-1)*2-1); y=(yy/(h-1)*2-1)
            mask=(abs(x-p[0])<.14)&(abs(y-p[1])<.14)
            depth=np.full((h,h),2.,np.float32); depth[mask]=1.
            seg=mask.astype(np.int32); body=np.eye(4,dtype=np.float32); body[:3,3]=[p[0],p[1],1.]
            intrinsic=np.array([[(h-1)/2,0,(h-1)/2],[0,(h-1)/2,(h-1)/2],[0,0,1]],np.float32)
            snapshots.append({"depth":depth,"segmentation":seg,"intrinsics":intrinsic,"camera_to_world":np.eye(4,dtype=np.float32),
                              "body_poses":body[None],"body_ids":np.array([1],np.int32)})
        return snapshots

    def close(self): pass


def quaternion_matrix(q):
    q=np.asarray(q); q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)


class ManiSkillAdapter:
    def __init__(self,config,collect_geometry=False):
        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install .[sim] on a supported Linux/NVIDIA machine") from exc
        e=config.eval; self.config=config; self.n=e.num_envs; self.geometry_enabled=collect_geometry
        if e.action_space!="native": raise ValueError("Only verified native ManiSkill action spaces are enabled")
        if collect_geometry and self.n!=1:
            raise ValueError("Oracle collection currently requires num_envs=1 to avoid unverified subscene frame conventions")
        sensor_configs={"width":config.model.image_size,"height":config.model.image_size}
        if e.ood_condition!="id" and e.env_id!="PushCube-v1": raise ValueError("Pilot OOD interventions are PushCube-only")
        if e.ood_condition=="camera30":
            from mani_skill.utils import sapien_utils
            sensor_configs["base_camera"]={"pose":sapien_utils.look_at(eye=[-.1+.4*np.cos(np.pi/6),.4*np.sin(np.pi/6),.6],target=[-.1,0,.1])}
        if e.ood_condition not in {"id","cube_color","camera30"}: raise ValueError("Unknown OOD intervention")
        self.env=gym.make(e.env_id,robot_uids=e.robot,num_envs=self.n,obs_mode="rgb+depth+segmentation" if collect_geometry else "rgb",
                          control_mode=e.control_mode,sim_backend=e.sim_backend,render_backend=e.render_backend,reconfiguration_freq=1,
                          sensor_configs=sensor_configs,max_episode_steps=e.horizon)
        space=self.env.unwrapped.single_action_space
        self.action_dim=space.shape[-1]; self.low=np.asarray(space.low); self.high=np.asarray(space.high)
        if self.action_dim!=config.model.action_dim: raise ValueError(f"Native action dimensions {self.action_dim} != model {config.model.action_dim}")
        if int(self.env.unwrapped.control_freq)!=e.control_hz: raise ValueError("Control frequency mismatch")
        self.obs=None

    @staticmethod
    def numpy(value): return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)

    def observation(self):
        sensors=self.obs["sensor_data"]
        names=self.config.eval.camera_names
        if set(names)-sensors.keys(): raise ValueError(f"Missing cameras {set(names)-sensors.keys()}; available {list(sensors)}")
        return np.stack([self.numpy(sensors[name]["rgb"]) for name in names],axis=1).astype(np.uint8)

    def reset(self,seeds):
        self.obs,_=self.env.reset(seed=list(seeds) if self.n>1 else int(seeds[0])); self.steps=0
        if self.config.eval.ood_condition=="cube_color":
            import sapien
            for entity in self.env.unwrapped.obj._objs:
                body=entity.find_component_by_type(sapien.render.RenderBodyComponent)
                if body is None: raise ValueError("Cube lacks render geometry")
                for shape in body.render_shapes:
                    material=shape.material
                    material.base_color=[1.,.75,0.,1.]
            self.obs=self.env.unwrapped.get_obs()
        return self.observation()

    def step(self,action):
        if not np.isfinite(action).all(): raise ValueError("Nonfinite controller action")
        command=torch.as_tensor(action,device=self.env.unwrapped.device,dtype=torch.float32)
        self.obs,reward,terminated,truncated,info=self.env.step(command); self.steps+=1
        # Native environment has no automatic reset; ignore success termination and use fixed horizons.
        return self.observation(),{"success":self.numpy(info.get("success",torch.zeros(self.n,dtype=torch.bool))),
                                   "reward":self.numpy(reward),"terminated":self.numpy(terminated),"truncated":self.numpy(truncated)}

    def geometry(self):
        if not self.geometry_enabled: raise RuntimeError("Geometry capture was not requested")
        name=self.config.eval.camera_names[0]; data=self.obs["sensor_data"][name]; params=self.obs["sensor_param"][name]
        env=self.env.unwrapped
        poses=[]; ids=[]
        for body_id,obj in sorted(env.segmentation_id_map.items()):
            if not hasattr(obj,"pose"): continue
            pose=obj.pose
            position=self.numpy(pose.p)[0]; q=self.numpy(pose.q)[0]
            transform=np.eye(4,dtype=np.float32); transform[:3,:3]=quaternion_matrix(q); transform[:3,3]=position
            poses.append(transform); ids.append(body_id)
        world_to_cv=self.numpy(params["extrinsic_cv"])[0]
        if world_to_cv.shape==(3,4):
            world_to_cv=np.concatenate([world_to_cv,np.array([[0,0,0,1]],dtype=world_to_cv.dtype)])
        if world_to_cv.shape!=(4,4): raise ValueError(f"Unsupported extrinsic shape {world_to_cv.shape}")
        cv_to_world=np.linalg.inv(world_to_cv)
        return [{"depth":self.numpy(data["depth"])[0,...,0].astype(np.float32)/1000.,
                 "segmentation":self.numpy(data["segmentation"])[0,...,0],"intrinsics":self.numpy(params["intrinsic_cv"])[0],
                 "camera_to_world":cv_to_world,"body_poses":np.stack(poses),"body_ids":np.array(ids,dtype=np.int32)}]

    def close(self): self.env.close()


def make_adapter(config,geometry=False):
    return ToyAdapter(config) if config.eval.backend=="toy" else ManiSkillAdapter(config,geometry)


def collect(config: Config,count=20,expert=None,source="sim",human_fraction=0.):
    if config.eval.backend=="maniskill" and expert is None:
        raise ValueError("ManiSkill collection requires an explicit teacher bundle; import planner demos via import-maniskill")
    env=make_adapter(config,geometry=config.data.provider=="oracle")
    if env.n!=1: raise ValueError("Collection currently uses one episode at a time")
    teacher=BundlePolicy(expert,config.eval.device) if expert else None
    if teacher: teacher.assert_compatible(config)
    try:
        for i in range(count):
            episode_id=f"{source}_{i:06d}"; obs=env.reset([config.train.seed+i]); observations=[obs[0]]; actions=[]; geometry=[]
            if config.data.provider=="oracle": geometry.append(env.geometry()[0])
            executor=ChunkExecutor(config.eval.execute_steps); executor.reset([episode_id])
            if teacher:
                teacher.reset([episode_id]); codec=NativeActionCodec(teacher.stats["action"],env.action_dim,env.low,env.high)
            for step in range(config.eval.horizon):
                if teacher:
                    if executor.needs_chunk():
                        executor.install(teacher.predict(obs,[config.eval.instruction],[episode_id],[step],config.train.seed+i*10000+step),[step])
                    action=codec.decode(executor.pop()).numpy()
                else: action=env.expert_action()
                obs,info=env.step(action); observations.append(obs[0]); actions.append(action[0])
                if config.data.provider=="oracle": geometry.append(env.geometry()[0])
            arrays={"rgb":np.stack(observations),"times":np.arange(len(observations),dtype=np.float64)/config.eval.control_hz}
            labeled=i>=round(count*human_fraction)
            if labeled: arrays.update(actions=np.stack(actions).astype(np.float32),action_times=arrays["times"][:-1])
            if geometry:
                arrays.update({k:np.stack([g[k] for g in geometry]) for k in geometry[0] if k!="body_ids"})
                arrays["body_ids"]=geometry[0]["body_ids"]
            meta={"source":source if labeled else f"{source}_action_free","instruction":config.eval.instruction,"camera_names":config.eval.camera_names,
                  "embodiment":config.eval.embodiment,"control_mode":config.eval.control_mode,"control_hz":config.eval.control_hz,
                  "split_group":episode_id,"collection":"teacher" if teacher else "toy_analytic_expert","success_at_end":bool(info["success"][0]),
                  "action_free_kind":None if labeled else "synthetic_robot_labels_removed"}
            save_episode(config.data.raw,episode_id,arrays,meta)
    finally: env.close()
    return {"episodes":count,"raw":config.data.raw}


def episode_ledger(config):
    if config.eval.ledger:
        ledger=read_jsonl(config.eval.ledger)
        if len({r["episode_id"] for r in ledger})!=len(ledger): raise ValueError("Duplicate evaluation episode IDs")
        return ledger
    return [{"episode_id":f"eval_{i:06d}","seed":config.eval.seed+i,"instruction":config.eval.instruction,
             "task":config.eval.env_id if config.eval.backend=="maniskill" else "toy_point"} for i in range(config.eval.episodes)]


def motion_consistency(predicted,encoders,stats,geometry,grid_size):
    """Compare predicted front-view flow to realized rigid flow, offline only."""
    h=min(encoders.config.horizon,len(geometry))
    poses=np.stack([g["camera_to_world"] for g in geometry[:h]])
    tracks,valid=oracle_tracks(geometry[0]["depth"],geometry[0]["segmentation"],geometry[0]["intrinsics"],poses,
                             np.stack([g["body_poses"] for g in geometry[:h]]),geometry[0]["body_ids"],grid_size)
    actual=canonical_flow(tracks,poses)
    decoded=encoders.decode_video(predicted,h).permute(0,2,3,1).float().cpu().numpy()
    # Decode onto the same initial query grid; label boundaries stay diagnostic limitations.
    from .geometry import query_grid
    uv=query_grid(decoded.shape[1],decoded.shape[2],grid_size)
    recovered=stats.denormalize(decoded[:,np.rint(uv[:,1]).astype(int),np.rint(uv[:,0]).astype(int)])
    return float(np.linalg.norm(recovered-actual,axis=-1)[valid].mean())


def evaluate(config: Config,bundle=None,baseline=None,policy_instance=None):
    config.validate(); policy=policy_instance or (BundlePolicy(bundle,config.eval.device,config.eval.sampler,config.eval.sampling_steps,config.eval.shift) if bundle else None)
    if not policy and baseline not in {"random","expert"}: raise ValueError("Specify bundle or random/expert baseline")
    if baseline=="expert" and config.eval.backend!="toy": raise ValueError("Analytic expert is toy-only; supply a robot teacher bundle")
    if policy: policy.assert_compatible(config)
    flow_source=config.eval.flow_source
    if policy and config.eval.record and config.eval.motion_diagnostics and config.data.provider=="oracle":
        if not flow_source and len(policy.stats["flow"])==1: flow_source=next(iter(policy.stats["flow"]))
        if flow_source not in policy.stats["flow"]:
            raise ValueError("Set eval.flow_source to the training source whose flow renderer applies to this simulator")
    output=Path(config.eval.output)
    if (output/"episodes.jsonl").exists(): raise FileExistsError("Evaluation run exists; choose another output")
    output.mkdir(parents=True,exist_ok=True)
    ledger=episode_ledger(config); write_jsonl(output/"ledger.jsonl",ledger)
    atomic_json(output/"config.json",asdict(config)); env=make_adapter(config,geometry=config.eval.record and config.eval.motion_diagnostics and config.data.provider=="oracle")
    codec=NativeActionCodec(policy.stats["action"],env.action_dim,env.low,env.high) if policy else None
    rows=[]; generator=np.random.default_rng(config.eval.seed); start_all=time.monotonic()
    try:
        for start in range(0,len(ledger),env.n):
            cohort=ledger[start:start+env.n]
            # Pad the last cohort, but never count padded episodes in aggregates.
            padded=cohort+[cohort[-1]]*(env.n-len(cohort)); ids=[r["episode_id"] for r in padded]
            obs=env.reset([r["seed"] for r in padded]); executor=ChunkExecutor(config.eval.execute_steps); executor.reset(ids)
            if policy: policy.reset(ids)
            ever=np.zeros(env.n,bool); last=np.zeros(env.n,bool); first=np.full(env.n,-1,int); returns=np.zeros(env.n); latencies=[]; recorded=[]
            active_geometry=[]; consistency=[]; active_flow=None; recorded_actions=[]; object_positions=[]; tcp_positions=[]
            for step in range(config.eval.horizon):
                if policy:
                    if executor.needs_chunk():
                        if active_flow is not None and active_geometry:
                            # Source selection is explicit: deployment dataset must supply its flow renderer identity.
                            source=flow_source
                            if source and source in policy.stats["flow"]:
                                consistency.append(motion_consistency(active_flow,policy.encoders,PercentileStats(**policy.stats["flow"][source]),active_geometry,config.data.grid_size))
                        active_geometry=[]
                        begin=time.monotonic(); chunk=policy.predict(obs,[r["instruction"] for r in padded],ids,[step]*env.n,config.eval.seed+start*10000+step)
                        latencies.append(time.monotonic()-begin); executor.install(chunk,[step]*env.n)
                        if config.eval.record and step==0:
                            from .data import atomic_npz
                            predicted={"actions_normalized":chunk.actions.float().numpy()}
                            for modality in ("flow","video"):
                                latent=getattr(chunk,modality,None)
                                if latent is not None:
                                    predicted[modality+"_latent"]=latent[0].float().cpu().numpy()
                                    predicted[modality+"_decoded_rgb"]=policy.encoders.decode_video(latent[0],config.model.horizon).permute(0,2,3,1).float().cpu().numpy()
                            atomic_npz(output/f"predicted_aux_{start:06d}.npz",predicted)
                        active_flow=chunk.flow[0] if chunk.flow is not None and env.n==1 and config.eval.record else None
                    action=codec.decode(executor.pop()).numpy()
                else: action=env.expert_action() if baseline=="expert" else generator.uniform(env.low,env.high,(env.n,env.action_dim))
                if config.eval.record:
                    recorded.append(obs.copy()); recorded_actions.append(np.asarray(action).copy())
                    if config.eval.backend=="maniskill" and hasattr(env.env.unwrapped,"obj"):
                        object_positions.append(env.numpy(env.env.unwrapped.obj.pose.p))
                        tcp_positions.append(env.numpy(env.env.unwrapped.agent.tcp.pose.p))
                    if config.eval.motion_diagnostics and config.data.provider=="oracle" and env.n==1: active_geometry.append(env.geometry()[0])
                obs,info=env.step(action)
                last=info["success"].astype(bool); first[(first<0)&last]=step+1; ever|=last; returns+=info["reward"]
            if active_flow is not None and active_geometry and flow_source:
                consistency.append(motion_consistency(active_flow,policy.encoders,PercentileStats(**policy.stats["flow"][flow_source]),active_geometry,config.data.grid_size))
            for j,row in enumerate(cohort):
                record={**row,"bundle_id":policy.bundle_id if policy else baseline,"success_once":bool(ever[j]),"success_at_end":bool(last[j]),
                        "first_success_step":int(first[j]),"return":float(returns[j]),"control_steps":config.eval.horizon,
                        "planning_seconds":latencies,"motion_consistency_error":float(np.mean(consistency)) if consistency else None}
                rows.append(record)
                with open(output/"episodes.jsonl","a") as f: f.write(json.dumps(record,allow_nan=False)+"\n")
            if recorded:
                from .data import atomic_npz
                recorded.append(obs.copy())
                arrays={"rgb":np.stack(recorded),"actions_native":np.stack(recorded_actions)}
                if object_positions:
                    object_positions.append(env.numpy(env.env.unwrapped.obj.pose.p)); tcp_positions.append(env.numpy(env.env.unwrapped.agent.tcp.pose.p))
                    arrays.update(object_positions_m=np.stack(object_positions),tcp_positions_m=np.stack(tcp_positions))
                    if hasattr(env.env.unwrapped,"goal_region"): arrays["goal_positions_m"]=env.numpy(env.env.unwrapped.goal_region.pose.p)
                atomic_npz(output/f"cohort_{start:06d}.npz",arrays)
                if config.eval.backend=="maniskill":
                    import imageio.v2 as imageio
                    imageio.mimsave(output/f"cohort_{start:06d}.mp4",np.stack(recorded)[:,0,0],fps=config.eval.control_hz)
    finally: env.close()
    summary=aggregate(rows); summary.update(wall_seconds=time.monotonic()-start_all,protocol="synchronous_fixed_horizon",ledger_id=digest(ledger))
    atomic_json(output/"summary.json",summary)
    return summary


def aggregate(rows):
    if not rows: raise ValueError("No episodes to aggregate")
    tasks={}
    for task in sorted({r["task"] for r in rows}):
        subset=[r for r in rows if r["task"]==task]; n=len(subset)
        successes=sum(r["success_at_end"] for r in subset)
        tasks[task]={"trials":n,"successes":successes,"success_at_end":successes/n,
                     "success_once":sum(r["success_once"] for r in subset)/n,"wilson95":wilson(successes,n)}
    timings=[t for r in rows for t in r.get("planning_seconds",[])]
    return {"episodes":len(rows),"tasks":tasks,"macro_success_at_end":float(np.mean([v["success_at_end"] for v in tasks.values()])),
            "planning_p50_seconds":float(np.percentile(timings,50)) if timings else None,
            "planning_p95_seconds":float(np.percentile(timings,95)) if timings else None}
