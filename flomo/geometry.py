"""Explicit coordinate conventions, cumulative flow, and reversible scaling."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...qj->...qi",transform[...,:3,:3],points) + transform[...,:3,3][...,None,:]


def canonical_flow(points: np.ndarray, camera_to_world: np.ndarray, coordinates="world"):
    points = np.asarray(points,dtype=np.float64)
    if coordinates == "camera": points = transform_points(camera_to_world,points)
    elif coordinates != "world": raise ValueError("coordinates must be world or camera")
    local = transform_points(np.broadcast_to(np.linalg.inv(camera_to_world[0]),camera_to_world.shape),points)
    flow = local - local[:1]
    if not np.isfinite(flow).all(): raise ValueError("Nonfinite scene flow")
    return flow.astype(np.float32)


def gl_to_cv(camera_to_world_gl):
    return np.asarray(camera_to_world_gl) @ np.diag([1.,-1.,-1.,1.])


def query_grid(height, width, size):
    # Pixel centers with an explicit row-major y/x layout.
    y = np.linspace(0,height-1,size); x = np.linspace(0,width-1,size)
    xx,yy = np.meshgrid(x,y)
    return np.stack([xx,yy],axis=-1).reshape(-1,2)


def oracle_tracks(depth0, segmentation0, intrinsic, camera_to_world, body_poses, body_ids, grid_size, pixel_center_offset=0.0):
    """Track initial rigid surface points via actor/link-local coordinates.

    depth in meters; CV camera axes; body_poses [T,K,4,4] maps body -> world.
    Visibility after t=0 is not asserted: points survive occlusion geometrically.
    """
    h,w = depth0.shape
    uv = query_grid(h,w,grid_size)
    ix = np.rint(uv[:,0]).astype(int); iy = np.rint(uv[:,1]).astype(int)
    depth = depth0[iy,ix]
    valid = np.isfinite(depth) & (depth > 0)
    # Depth/segmentation came from integer pixel centers; unproject that ray.
    rays = np.stack([ix+pixel_center_offset,iy+pixel_center_offset,np.ones(len(uv))],axis=-1) @ np.linalg.inv(intrinsic).T
    points = transform_points(camera_to_world[:1],(rays*depth[:,None])[None])[0]
    ids = segmentation0[iy,ix]
    out = np.broadcast_to(points,(len(camera_to_world),*points.shape)).copy()
    for j,body_id in enumerate(body_ids):
        selected = ids == body_id
        if not selected.any(): continue
        local = transform_points(np.linalg.inv(body_poses[0,j])[None],points[selected][None])[0]
        out[:,selected] = transform_points(body_poses[:,j],np.broadcast_to(local,(len(out),*local.shape)))
    # Segmentation 0 is static background; unknown nonzero IDs are not safe labels.
    valid &= (ids == 0) | np.isin(ids,body_ids)
    out[:,~valid] = 0
    return out.astype(np.float32),np.broadcast_to(valid,(len(out),len(valid))).copy()


@dataclass
class PercentileStats:
    low: list[float]
    high: list[float]
    quantiles: list[float]

    @classmethod
    def fit(cls, values, quantiles=(1.,99.)):
        values = np.asarray(values,dtype=np.float32)
        if values.ndim != 2 or len(values)==0 or not np.isfinite(values).all():
            raise ValueError("Stats require finite nonempty [N,D] values")
        lo,hi = np.percentile(values,quantiles,axis=0)
        return cls(lo.tolist(),hi.tolist(),list(quantiles))

    def normalize(self, values, signed=False):
        lo = np.asarray(self.low); hi = np.asarray(self.high)
        constant = hi-lo < 1e-8
        unit = np.clip((np.asarray(values)-lo)/np.where(constant,1.,hi-lo),0.,1.)
        unit = np.where(constant,.5,unit)
        return (2*unit-1 if signed else unit).astype(np.float32)

    def denormalize(self, values, signed=False):
        if isinstance(values,torch.Tensor):
            lo = values.new_tensor(self.low); hi = values.new_tensor(self.high)
            u = (values+1)/2 if signed else values
            return lo + u.clamp(0,1)*(hi-lo)
        u = (np.asarray(values)+1)/2 if signed else np.asarray(values)
        return (np.asarray(self.low)+np.clip(u,0,1)*(np.asarray(self.high)-self.low)).astype(np.float32)


def render_flow(flow, valid, stats: PercentileStats, grid_size, image_size):
    if flow.shape[1:] != (grid_size*grid_size,3): raise ValueError("Unexpected query layout")
    rgb = stats.normalize(flow)
    # Explicit fill: invalid motion is filled with zero-displacement color;
    # quality rejection and validity remain separate from the rendered target.
    rgb[~valid] = stats.normalize(np.zeros((1,3),np.float32))[0]
    tensor = torch.from_numpy(rgb.reshape(len(flow),grid_size,grid_size,3)).permute(0,3,1,2)
    return F.interpolate(tensor,size=(image_size,image_size),mode="bilinear",align_corners=False)


def rotation_6d_to_matrix(x: torch.Tensor):
    """First two columns convention, with deterministic degeneracy handling."""
    a,b = x[...,:3],x[...,3:]
    fallback = torch.zeros_like(a); fallback[...,0]=1
    a = torch.where(a.norm(dim=-1,keepdim=True)>1e-7,a,fallback)
    e1 = F.normalize(a,dim=-1)
    b = b-(e1*b).sum(-1,keepdim=True)*e1
    axis = F.one_hot(e1.abs().argmin(-1),3).to(e1)
    alt = axis-(axis*e1).sum(-1,keepdim=True)*e1
    b = torch.where(b.norm(dim=-1,keepdim=True)>1e-7,b,alt)
    e2 = F.normalize(b,dim=-1)
    return torch.stack([e1,e2,torch.linalg.cross(e1,e2)],dim=-1)


def matrix_to_axis_angle(matrix):
    # Stable matrix -> quaternion -> shortest rotation vector, including pi.
    m = matrix
    qabs = torch.stack([1+m[...,0,0]+m[...,1,1]+m[...,2,2],
                       1+m[...,0,0]-m[...,1,1]-m[...,2,2],
                       1-m[...,0,0]+m[...,1,1]-m[...,2,2],
                       1-m[...,0,0]-m[...,1,1]+m[...,2,2]],-1).clamp_min(0).sqrt()
    candidates = torch.stack([
        torch.stack([qabs[...,0]**2,m[...,2,1]-m[...,1,2],m[...,0,2]-m[...,2,0],m[...,1,0]-m[...,0,1]],-1),
        torch.stack([m[...,2,1]-m[...,1,2],qabs[...,1]**2,m[...,1,0]+m[...,0,1],m[...,0,2]+m[...,2,0]],-1),
        torch.stack([m[...,0,2]-m[...,2,0],m[...,1,0]+m[...,0,1],qabs[...,2]**2,m[...,2,1]+m[...,1,2]],-1),
        torch.stack([m[...,1,0]-m[...,0,1],m[...,0,2]+m[...,2,0],m[...,2,1]+m[...,1,2],qabs[...,3]**2],-1)],-2)
    candidates = candidates/(2*qabs[...,None].clamp_min(.1))
    idx = qabs.argmax(-1)
    q = candidates.gather(-2,idx[...,None,None].expand(*idx.shape,1,4)).squeeze(-2)
    q = F.normalize(q,dim=-1); q = torch.where(q[...,:1]<0,-q,q)
    norm = q[...,1:].norm(dim=-1,keepdim=True)
    angle = 2*torch.atan2(norm,q[...,:1])
    return q[...,1:]*torch.where(norm>1e-7,angle/norm.clamp_min(1e-7),torch.full_like(norm,2.))
