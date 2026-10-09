"""One monitored subprocess; never leaves a failed job marked running."""
import argparse,json,os,pathlib,subprocess,time,sys,tempfile
p=argparse.ArgumentParser();p.add_argument("--name",required=True);p.add_argument("--log",required=True);p.add_argument("command",nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==["--"] else a.command
root=pathlib.Path(os.environ.get("PHYLA_RUN_ROOT","/mnt/nvme/scratch/phyla-ubuntu"));path=root/"control/exp03/jobs"/(a.name+".json");path.parent.mkdir(parents=True,exist_ok=True)
def save(d):
 tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(d,indent=2));tmp.replace(path)
d={"job":a.name,"state":"starting","command":cmd,"wrapper_pid":os.getpid(),"log":a.log,"started":time.time(),"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES","")};save(d)
try:
 with open(a.log,"a") as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);d.update(state="running",pid=child.pid);save(d);code=child.wait()
 d.update(state="completed" if code==0 else "failed",returncode=code,finished=time.time());save(d);sys.exit(code)
except Exception as exc:
 d.update(state="failed",error=repr(exc),finished=time.time());save(d);raise
