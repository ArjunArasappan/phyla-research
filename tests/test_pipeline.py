import copy
import json

import numpy as np
import pytest
import torch

from flomo.config import Config
from flomo.data import PreparedDataset,MixtureSampler,read_jsonl,validate_episode
from flomo.train import train,load_artifact,export_bundle
from flomo.policy import BundlePolicy,ActionChunk,ChunkExecutor
from flomo.sim import evaluate


def test_split_isolation_and_manifest_validation(prepared):
    config,metadata=prepared
    rows=read_jsonl(f"{config.data.prepared}/windows.jsonl")
    splits={}
    for row in rows: splits.setdefault(row["episode_id"],set()).add(row["split"])
    assert all(len(values)==1 for values in splits.values())
    assert metadata["windows"]>0
    assert PreparedDataset(config.data.prepared).validate()["split_windows"]>0


def test_stateless_distributed_sampler(prepared):
    dataset=PreparedDataset(prepared[0].data.prepared)
    all_=MixtureSampler(dataset,{},4,3)
    r0=MixtureSampler(dataset,{},2,3,0,2); r1=MixtureSampler(dataset,{},2,3,1,2)
    assert all_.indices(17)==r0.indices(17)+r1.indices(17)


def test_robot_episode_alignment_rejected():
    arrays={"rgb":np.zeros((4,1,8,8,3),np.uint8),"times":np.arange(4),"actions":np.zeros((4,2))}
    with pytest.raises(ValueError,match="N actions"):
        validate_episode(arrays,{"camera_names":["front"],"instruction":"move","source":"test"})


def test_training_resume_is_exact(config,tmp_path):
    full=copy.deepcopy(config); full.train.steps=4; full.train.output=str(tmp_path/"full")
    state_full=load_artifact(train(full))
    partial=copy.deepcopy(config); partial.train.steps=2; partial.train.output=str(tmp_path/"partial")
    first=train(partial); partial.train.steps=4; partial.train.resume=first
    state_resumed=load_artifact(train(partial))
    assert all(torch.equal(value,state_resumed["weights"][key]) for key,value in state_full["weights"].items())


def test_sft_bundle_and_probe_handoff(config,tmp_path):
    initial=train(config); bundle=tmp_path/"initial_bundle"; export_bundle(initial,bundle)
    sft=copy.deepcopy(config); sft.train.stage="sft"; sft.train.init=str(bundle); sft.train.output=str(tmp_path/"sft")
    sft.train.losses["action"]=1.; second=train(sft)
    deployed=tmp_path/"policy"; export_bundle(second,deployed)
    policy=BundlePolicy(deployed); policy.assert_compatible(config)
    c=copy.deepcopy(config); c.eval.control_mode="different_controller"
    with pytest.raises(ValueError,match="control_mode"): policy.assert_compatible(c)
    c=copy.deepcopy(config); c.train.stage="probe"; c.train.probe="video"; c.model.targets=["action"]
    c.train.init=str(bundle); c.train.output=str(tmp_path/"probe"); c.train.losses={"action":1.}
    artifact=tmp_path/"probe_bundle"; export_bundle(train(c),artifact)
    with pytest.raises(ValueError,match="cannot be deployed"): BundlePolicy(artifact)


def test_chunk_reset_and_stale_response():
    executor=ChunkExecutor(4); executor.reset(["episode-a"])
    chunk=ActionChunk(torch.zeros(1,16,2),["episode-a"],[0],"bundle")
    executor.install(chunk,[0]); executor.pop(); executor.reset(["episode-b"])
    assert executor.needs_chunk()
    with pytest.raises(ValueError,match="Stale"): executor.install(chunk,[0])
    with pytest.raises(RuntimeError): executor.pop()


def test_expert_fixed_horizon_and_partial_cohort(config):
    config.eval.episodes=3; config.eval.num_envs=2; config.eval.horizon=32
    result=evaluate(config,baseline="expert")
    assert result["episodes"]==3 and result["macro_success_at_end"]==1.
    records=read_jsonl(f"{config.eval.output}/episodes.jsonl")
    assert len(records)==3 and all(r["control_steps"]==32 for r in records)


def test_direct_bc_training_and_bundle(config,tmp_path):
    config.model.architecture="bc"; config.model.targets=["action"]
    config.train.stage="sft"; config.train.losses={"action":1.}
    bundle=tmp_path/"bc"; export_bundle(train(config),bundle)
    policy=BundlePolicy(bundle)
    assert getattr(policy.model,"deterministic",False)


def test_import_replayed_maniskill_hdf5(tmp_path):
    import h5py
    from flomo.cli import import_maniskill
    config=Config(); config.data.raw=str(tmp_path/"raw")
    path=tmp_path/"trajectory.h5"
    with h5py.File(path,"w") as f:
        g=f.create_group("traj_0")
        g.create_dataset("actions",data=np.zeros((16,2),np.float32))
        g.create_dataset("obs/sensor_data/front/rgb",data=np.zeros((17,8,8,3),np.uint8))
    path.with_suffix(".json").write_text(json.dumps({"env_info":{"env_id":"toy_point","env_kwargs":{"control_mode":"xy_delta"}},"episodes":[{"episode_id":0}]}))
    result=import_maniskill(config,path,"demo")
    assert result["episodes"]==1
    assert read_jsonl(f"{config.data.raw}/episodes.jsonl")[0]["has_actions"]


def test_recorded_rollout_motion_diagnostics(config,tmp_path):
    bundle=tmp_path/"diag_policy"; export_bundle(train(config),bundle)
    config.eval.record=True; config.eval.flow_source="sim"; config.eval.num_envs=1; config.eval.episodes=2
    evaluate(config,bundle=str(bundle))
    rows=read_jsonl(f"{config.eval.output}/episodes.jsonl")
    assert all(np.isfinite(r["motion_consistency_error"]) for r in rows)


def test_paired_comparison_rejects_different_seeds():
    from flomo.policy import paired_comparison
    left=[{"episode_id":"a","seed":1,"task":"pick","instruction":"pick","success_at_end":False}]
    right=[{**left[0],"success_at_end":True}]
    assert paired_comparison(left,right,bootstrap=10)["right_minus_left_macro_success"]==1.
    right[0]["seed"]=2
    with pytest.raises(ValueError,match="initializations"): paired_comparison(left,right)
