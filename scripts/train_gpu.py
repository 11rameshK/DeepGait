import os, sys, argparse, json, random
from pathlib import Path
import torch

ROOT=Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\DeepGait_Project")
sys.path.insert(0,str(ROOT/"src"))
from deepgait_model import DeepGaitGGCN
from dataset import PairedConditionDataset, collate_paired
from losses import batch_all_triplet_loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B-pkl")
    ap.add_argument("--iterations", type=int, default=160000)
    ap.add_argument("--subjects-per-batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--resume", default="")
    args=ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for full training. Use smoke_train.py on this laptop.")
    device=torch.device("cuda")
    ds=PairedConditionDataset(args.data, range(1,75), frames=args.frames, train=True)
    model=DeepGaitGGCN().to(device).train()
    opt=torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.999))
    start=0
    if args.resume:
        ck=torch.load(args.resume,map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start=ck.get("iteration",0)
    outdir=ROOT/"outputs"/"gpu_training"; outdir.mkdir(parents=True,exist_ok=True)
    for it in range(start,args.iterations):
        ids=random.sample(range(len(ds)), min(args.subjects_per_batch,len(ds)))
        b=collate_paired([ds[i] for i in ids])
        opt.zero_grad()
        out=model(nm=b["nm"].to(device), bg=b["bg"].to(device), cl=b["cl"].to(device))
        map_loss=model.mapping_loss(out,0.01,0.1)
        embs=torch.cat([out["nm"]["embedding"],out["bg"]["embedding"],out["cl"]["embedding"]],0)
        labels=torch.cat([b["subject"],b["subject"],b["subject"]]).to(device)
        tri=batch_all_triplet_loss(embs,labels,0.2)
        loss=map_loss+tri
        loss.backward(); opt.step()
        if (it+1)%100==0:
            print(it+1,float(loss.detach()),float(map_loss.detach()),float(tri.detach()),flush=True)
        if (it+1)%args.save_every==0 or it+1==args.iterations:
            torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"iteration":it+1},outdir/f"checkpoint_{it+1:06d}.pt")

if __name__=="__main__":
    main()
