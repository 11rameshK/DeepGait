from pathlib import Path
import pickle, numpy as np, random, os, json
root=Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B-pkl")
files=list(root.glob("*/*/*/*.pkl"))
train=[p for p in files if int(p.parts[-4])<=74]
test=[p for p in files if int(p.parts[-4])>=75]
sample=files[::max(1,len(files)//100)][:100]
bad=[]
for p in sample:
    try:
        with open(p,"rb") as f: a=pickle.load(f)
        if not isinstance(a,np.ndarray) or a.ndim!=3 or a.shape[1:]!=(64,44) or len(a)<5:
            bad.append((str(p),getattr(a,"shape",None)))
    except Exception as e:
        bad.append((str(p),repr(e)))
print("PKL_COUNT",len(files))
print("TRAIN_COUNT",len(train))
print("TEST_COUNT",len(test))
print("SAMPLE_CHECKED",len(sample))
print("SAMPLE_BAD",len(bad))
for x in bad: print("BAD",x)
print("QUICK_VERIFY_PASS",len(files)==13572 and len(train)==8092 and len(test)==5480 and not bad)
