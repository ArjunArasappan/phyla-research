import numpy as np
import pytest
import torch

from flomo.geometry import canonical_flow, oracle_tracks, PercentileStats, rotation_6d_to_matrix, matrix_to_axis_angle


def test_static_scene_under_camera_motion():
    pose=np.repeat(np.eye(4)[None],3,axis=0); pose[:,0,3]=[0,1,2]
    world=np.array([[1.,2.,4.],[3.,4.,5.]])
    local=np.repeat(world[None],3,axis=0)-pose[:,:3,3][:,None]
    assert np.allclose(canonical_flow(local,pose,"camera"),0)


def test_anchor_rotation_and_translation():
    camera=np.repeat(np.eye(4)[None],2,axis=0)
    camera[:,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    points=np.array([[[0.,0.,1.]],[[1.,0.,1.]]])
    flow=canonical_flow(points,camera)
    assert np.allclose(flow[0],0)
    assert np.allclose(flow[1,0],[0,-1,0])


def test_oracle_body_translation_and_background():
    depth=np.ones((2,2)); seg=np.array([[1,0],[1,0]])
    camera=np.repeat(np.eye(4)[None],2,axis=0)
    bodies=np.repeat(np.eye(4)[None,None],2,axis=0); bodies[1,0,0,3]=.4
    points,valid=oracle_tracks(depth,seg,np.eye(3),camera,bodies,[1],2)
    flow=canonical_flow(points,camera)
    assert valid.all()
    assert np.allclose(flow[1,[0,2],0],.4)
    assert np.allclose(flow[1,[1,3]],0)


def test_percentiles_constant_channel_and_inverse():
    values=np.array([[0.,2.],[1.,2.],[2.,2.]],np.float32)
    stats=PercentileStats.fit(values,(0,100)); encoded=stats.normalize(values,signed=True)
    assert np.allclose(encoded[:,1],0)
    assert np.allclose(stats.denormalize(encoded,signed=True),values)
    assert torch.allclose(stats.denormalize(torch.tensor(encoded),signed=True),torch.tensor(values))


def test_degenerate_rotation_and_pi_rotation():
    rotation=rotation_6d_to_matrix(torch.zeros(3,6))
    eye=torch.eye(3).expand(3,3,3)
    assert torch.allclose(rotation.transpose(-1,-2)@rotation,eye,atol=1e-6)
    assert torch.allclose(torch.linalg.det(rotation),torch.ones(3))
    pi=torch.diag(torch.tensor([1.,-1.,-1.]))
    vector=matrix_to_axis_angle(pi)
    assert torch.allclose(vector.abs(),torch.tensor([torch.pi,0,0]),atol=1e-5)
