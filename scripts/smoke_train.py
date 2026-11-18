import os, sys, argparse, json
from pathlib import Path
import torch

ROOT = Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\DeepGait_Project")
sys.path.insert(0, str(ROOT/"src"))
from deepgait_model import DeepGaitGGCN
from dataset import PairedConditionDataset, collate_paired
from losses import batch_all_triplet_loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B-pkl")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--subjects", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1)
    args=ap.parse_args()

    device=torch.device("cpu")
    ds=PairedConditionDataset(args.data, range(1, min(74,args.subjects)+1), frames=args.frames, train=True)
    if len(ds) < 2:
        ds=PairedConditionDataset(args.data, range(1,75), frames=args.frames, train=True)
    batch=[ds[i] for i in range(min(args.subjects,len(ds)))]
    b=collate_paired(batch)
    model=DeepGaitGGCN().to(device).train()
    opt=torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.999))
    history=[]
    for step in range(args.steps):
        opt.zero_grad()
        out=model(nm=b["nm"].to(device), bg=b["bg"].to(device), cl=b["cl"].to(device))
        map_loss=model.mapping_loss(out, lambda_bg=0.01, lambda_cl=0.1)
        embs=torch.cat([out["nm"]["embedding"],out["bg"]["embedding"],out["cl"]["embedding"]],dim=0)
        labels=torch.cat([b["subject"],b["subject"],b["subject"]]).to(device)
        tri=batch_all_triplet_loss(embs, labels, margin=0.2)
        total=map_loss+tri
        total.backward()
        grad=sum((p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None))
        opt.step()
        row={"step":step+1,"mapping_loss":float(map_loss.detach()),"triplet_loss":float(tri.detach()),"total_loss":float(total.detach()),"grad_abs_sum":float(grad)}
        history.append(row)
        print(json.dumps(row))
    outdir=ROOT/"outputs"; outdir.mkdir(exist_ok=True)
    ckpt=outdir/"smoke_checkpoint.pt"
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"history":history,"smoke_only":True},ckpt)
    (outdir/"smoke_history.json").write_text(json.dumps(history,indent=2))
    print("SMOKE_TRAIN_OK", ckpt)

if __name__=="__main__":
    main()
