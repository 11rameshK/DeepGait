import os,sys,argparse
from pathlib import Path
import torch

ROOT=Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\DeepGait_Project")
sys.path.insert(0,str(ROOT/"src"))
from deepgait_model import DeepGaitGGCN,pairwise_euclidean
from dataset import load_pkl,sample_frames

def emb(model,p,device,frames):
    x=sample_frames(load_pkl(p),frames,False).unsqueeze(0).to(device)
    with torch.no_grad(): return model(nm=x)["nm"]["embedding"].cpu()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--query",required=True)
    ap.add_argument("--gallery-root",default=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B-pkl")
    ap.add_argument("--view",default="090")
    ap.add_argument("--frames",type=int,default=30)
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(args.checkpoint,map_location=device)
    if ck.get("smoke_only"): raise RuntimeError("Smoke checkpoint is for code verification only, not identification.")
    model=DeepGaitGGCN().to(device).eval(); model.load_state_dict(ck["model"])
    q=emb(model,Path(args.query),device,args.frames)
    G=[]; Y=[]
    root=Path(args.gallery_root)
    for sid in range(75,125):
        s=f"{sid:03d}"
        for seq in ["nm-01","nm-02","nm-03","nm-04"]:
            p=root/s/seq/args.view/(args.view+".pkl")
            if p.exists(): G.append(emb(model,p,device,args.frames)); Y.append(sid)
    if not G: raise RuntimeError("No gallery embeddings found")
    dist=pairwise_euclidean(q,torch.cat(G,0))[0]
    idx=int(dist.argmin())
    print("PREDICTED_SUBJECT",Y[idx],"DISTANCE",float(dist[idx]))

if __name__=="__main__":
    main()
