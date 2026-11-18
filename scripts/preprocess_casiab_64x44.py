import argparse, os, pickle, multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from functools import partial
import cv2, numpy as np
from tqdm import tqdm

def norm(p):
    im=cv2.imread(str(p),0)
    if im is None or im.sum()<=10000: return None
    ys=np.where(im.sum(1)>0)[0]
    if not len(ys): return None
    im=im[ys[0]:ys[-1]+1]
    ratio=im.shape[1]/im.shape[0]
    nw=max(1,int(round(64*ratio)))
    im=cv2.resize(im,(nw,64),interpolation=cv2.INTER_CUBIC)
    cs=np.cumsum(im.sum(0)); total=cs[-1]
    if total<=0: return None
    xc=int(np.searchsorted(cs,total/2))
    im=np.pad(im,((0,0),(44,44)))
    xc+=44
    crop=im[:,xc-22:xc+22]
    return crop.astype(np.uint8) if crop.shape==(64,44) else None

def one(item,outroot):
    (sid,seq,view),paths=item
    arr=[x for x in (norm(p) for p in sorted(paths)) if x is not None]
    if len(arr)<5: return 0
    d=Path(outroot)/sid/seq/view; d.mkdir(parents=True,exist_ok=True)
    with open(d/(view+".pkl"),"wb") as f: pickle.dump(np.stack(arr),f,pickle.HIGHEST_PROTOCOL)
    return 1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("-i",required=True); ap.add_argument("-o",required=True); ap.add_argument("-n",type=int,default=6)
    a=ap.parse_args(); root=Path(a.i); groups=defaultdict(list)
    for p in root.rglob("*.png"):
        sid,seq,view=p.parts[-4],p.parts[-3],p.parts[-2]
        groups[(sid,seq,view)].append(p)
    print("GROUPS",len(groups),"FRAMES",sum(map(len,groups.values())),flush=True)
    fn=partial(one,outroot=a.o); ok=0
    with mp.Pool(a.n) as pool:
        for v in tqdm(pool.imap_unordered(fn,groups.items()),total=len(groups),unit="seq"):
            ok+=v
    print("PREPROCESS_DONE",ok,"/",len(groups),flush=True)

if __name__=="__main__":
    mp.freeze_support(); main()
