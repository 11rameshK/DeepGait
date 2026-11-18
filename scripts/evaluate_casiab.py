import os, sys, argparse, pickle, json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\DeepGait_Project")
sys.path.insert(0,str(ROOT/"src"))
from deepgait_model import DeepGaitGGCN, pairwise_euclidean
from dataset import load_pkl, sample_frames

GALLERY=["nm-01","nm-02","nm-03","nm-04"]
PROBES={"NM":["nm-05","nm-06"],"BG":["bg-01","bg-02"],"CL":["cl-01","cl-02"]}
VIEWS=["000","018","036","054","072","090","108","126","144","162","180"]

def embed(model,path,device,frames=30):
    arr=load_pkl(path)
    x=sample_frames(arr,frames,train=False).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(nm=x)["nm"]["embedding"].cpu()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B-pkl")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frames",type=int,default=30)
    ap.add_argument("--max-subjects",type=int,default=0)
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(args.checkpoint,map_location=device)
    if ck.get("smoke_only"):
        raise RuntimeError("Smoke checkpoint is not valid for reporting accuracy. Use a full training checkpoint.")
    model=DeepGaitGGCN().to(device).eval(); model.load_state_dict(ck["model"])
    root=Path(args.data)
    subjects=list(range(75,125))
    if args.max_subjects: subjects=subjects[:args.max_subjects]
    results={}
    for cond,seqs in PROBES.items():
        correct=total=0
        for view in VIEWS:
            gal_e=[]; gal_y=[]
            for sid in subjects:
                s=f"{sid:03d}"
                for seq in GALLERY:
                    p=root/s/seq/view/(view+".pkl")
                    if p.exists():
                        gal_e.append(embed(model,p,device,args.frames)); gal_y.append(sid)
            if not gal_e: continue
            G=torch.cat(gal_e,0)
            for sid in subjects:
                s=f"{sid:03d}"
                for seq in seqs:
                    p=root/s/seq/view/(view+".pkl")
                    if not p.exists(): continue
                    q=embed(model,p,device,args.frames)
                    d=pairwise_euclidean(q,G)
                    pred=gal_y[int(d.argmin())]
                    total+=1; correct+=int(pred==sid)
        results[cond]={"correct":correct,"total":total,"accuracy":(100*correct/total if total else None)}
    print(json.dumps(results,indent=2))
    (ROOT/"outputs"/"evaluation_results.json").write_text(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
