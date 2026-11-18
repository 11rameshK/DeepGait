import os, glob, tarfile, shutil
root=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B"
src=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\GaitDatasetB-silh\GaitDatasetB-silh"
expected_seqs=[f"nm-0{i}" for i in range(1,7)]+["bg-01","bg-02","cl-01","cl-02"]
expected_views=[f"{i:03d}" for i in range(0,181,18)]

def valid(sid):
    sp=os.path.join(root,sid)
    if not os.path.isdir(sp): return False
    for seq in expected_seqs:
        qp=os.path.join(sp,seq)
        if not os.path.isdir(qp): return False
        for view in expected_views:
            vp=os.path.join(qp,view)
            if not os.path.isdir(vp): return False
            if len(glob.glob(os.path.join(vp,"*.png")))<5: return False
    return True

bad=[f"{i:03d}" for i in range(1,125) if not valid(f"{i:03d}")]
print("Repairing",len(bad),"subjects:",bad,flush=True)
for i,sid in enumerate(bad,1):
    dst=os.path.join(root,sid)
    if os.path.exists(dst): shutil.rmtree(dst)
    arc=os.path.join(src,sid+".tar.gz")
    with tarfile.open(arc,"r:gz") as t:
        t.extractall(root, filter="data")
    print(f"{i}/{len(bad)} repaired {sid}",flush=True)
print("REPAIR_COMPLETE",flush=True)
