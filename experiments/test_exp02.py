"""Acceptance of geometry/rendering, independent of pretrained model kernels."""
import importlib.util,pathlib,unittest
import numpy as np
p=pathlib.Path(__file__).parent/'exp02.py'
spec=importlib.util.spec_from_file_location('exp02',p);exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
class RendererTests(unittest.TestCase):
 def test_constant_is_invertible_and_rgb_float32(self):
  g=16;d=np.broadcast_to(np.array([.05,-.02,.0],np.float32),(17,g*g,3)).copy();lo=np.array([-.1,-.03,-.02]);hi=np.array([.2,.1,.08]);valid=np.ones(g*g,bool)
  image,clipped,finite=exp.render(d,g,valid,lo,hi)
  self.assertEqual(str(image.dtype),'torch.float32');self.assertTrue(finite.all())
  np.testing.assert_allclose(exp.recover(image,g,lo,hi),d,atol=3e-8)
 def test_zero_code_uses_asymmetric_bounds(self):
  g=8;d=np.zeros((17,g*g,3),np.float32);lo=np.array([-.01,-.02,-.1]);hi=np.array([.2,.03,.01]);valid=np.ones(g*g,bool)
  image,_,_=exp.render(d,g,valid,lo,hi)
  np.testing.assert_allclose(image[0,:,0,0].numpy(),-lo/(hi-lo),atol=1e-7)
  np.testing.assert_allclose(exp.recover(image,g,lo,hi),d,atol=1e-8)
 def test_nonfinite_render_copy_does_not_mutate_tracks(self):
  g=8;d=np.zeros((17,g*g,3),np.float32);d[4,2]=np.nan;valid=np.ones(g*g,bool)
  image,_,finite=exp.render(d,g,valid,np.array([-.1]*3),np.array([.1]*3))
  self.assertTrue(np.isnan(d[4,2]).all());self.assertFalse(finite[4,2]);self.assertTrue(np.isfinite(image.numpy()).all())
 def test_vector_error_identity_and_epe_not_additive(self):
  g=8;rng=np.random.default_rng(4);gt=rng.uniform(-.1,.1,(17,g*g,3));d=gt+rng.normal(0,.03,gt.shape);valid=np.ones(g*g,bool);lo=np.array([-.08]*3);hi=np.array([.08]*3)
  image,clip,_=exp.render(d,g,valid,lo,hi);bar=exp.recover(image,g,lo,hi);decoded=bar+.003
  np.testing.assert_allclose((d-gt)+(clip-d)+(bar-clip)+(decoded-bar),decoded-gt,atol=1e-12)
 def test_metrics_in_meters(self):
  pred=np.array([[[.03,.04,0]]]);gt=np.zeros_like(pred);mask=np.ones((1,1),bool)
  self.assertAlmostEqual(exp.metric(pred,gt,mask)['epe_m'],.05)
if __name__=='__main__':unittest.main()
