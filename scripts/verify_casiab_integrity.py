import os, glob, sys
root=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B"
expected_seqs=[f"nm-0{i}" for i in range(1,7)]+["bg-01","bg-02","cl-01","cl-02"]
expected_views=[f"{i:03d}" for i in range(0,181,18)]
bad=[]
subjects=[f"{i:03d}" for i in range(1,125)]
for sid in subjects:
    sp=os.path.join(root,sid)
    reasons=[]
    if not os.path.isdir(sp):
        reasons.append("missing subject")
    else:
        seqs=set(d for d in os.listdir(sp) if os.path.isdir(os.path.join(sp,d)))
        miss_seq=[s for s in expected_seqs if s not in seqs]
        if miss_seq: reasons.append("missing seqs:"+",".join(miss_seq))
        for seq in expected_seqs:
            qp=os.path.join(sp,seq)
            if not os.path.isdir(qp): continue
            views=set(d for d in os.listdir(qp) if os.path.isdir(os.path.join(qp,d)))
            miss_views=[v for v in expected_views if v not in views]
            if miss_views:
                reasons.append(f"{seq} missing views:{','.join(miss_views)}")
                continue
            for view in expected_views:
                n=len(glob.glob(os.path.join(qp,view,"*.png")))
                if n<5:
                    reasons.append(f"{seq}/{view} frames={n}")
                    break
    if reasons:
        bad.append((sid,reasons))
print("TOTAL_SUBJECTS",len([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]) if os.path.isdir(root) else 0)
print("BAD_SUBJECTS",len(bad))
for sid,reasons in bad:
    print(sid," | ".join(reasons))
sys.exit(1 if bad else 0)
