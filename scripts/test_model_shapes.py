import os, sys, torch
ROOT=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\DeepGait_Project"
sys.path.insert(0,os.path.join(ROOT,"src"))
from deepgait_model import DeepGaitGGCN
m=DeepGaitGGCN().eval()
x=torch.rand(1,4,1,64,44)
with torch.no_grad():
    out=m(nm=x,bg=x,cl=x)
for k,v in out.items():
    print(k,"low",tuple(v["low"].shape),"high",tuple(v["high"].shape),"emb",tuple(v["embedding"].shape))
print("mapping_loss",float(m.mapping_loss(out)))
assert out["nm"]["low"].shape==(1,4,32,64,44)
assert out["nm"]["high"].shape==(1,4,128,16,11)
assert out["nm"]["embedding"].shape==(1,256,63)
print("MODEL_SHAPE_TEST_OK")
