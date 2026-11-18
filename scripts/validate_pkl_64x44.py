import os,pickle,glob
from pathlib import Path
import numpy as np
raw=Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B")
out=Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B-pkl")
groups=[]
for d in raw.glob("*/*/*"):
    if d.is_dir() and any(d.glob("*.png")):
        groups.append((d.parts[-3],d.parts[-2],d.parts[-1],d))
missing=[]
bad=[]
ok=0
train=test=0
for sid,seq,view,d in groups:
    f=out/sid/seq/view/(view+".pkl")
    if not f.exists():
        missing.append((sid,seq,view,len(list(d.glob("*.png")))))
        continue
    try:
        with open(f,"rb") as h: a=pickle.load(h)
        if not isinstance(a,np.ndarray) or a.ndim!=3 or a.shape[1:]!=(64,44) or len(a)<5:
            bad.append((sid,seq,view,getattr(a,"shape",None)))
        else:
            ok+=1
            if int(sid)<=74: train+=1
            else: test+=1
    except Exception as e:
        bad.append((sid,seq,view,str(e)))
print("RAW_GROUPS",len(groups))
print("PKL_OK",ok)
print("MISSING",len(missing))
for x in missing: print("MISS",*x)
print("BAD_PKL",len(bad))
for x in bad[:50]: print("BAD",*x)
print("TRAIN_PKL",train)
print("TEST_PKL",test)
print("VALIDATION_PASS", len(bad)==0 and ok+len(missing)==len(groups))
