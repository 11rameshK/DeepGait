import os, tarfile, glob, collections
src=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\GaitDatasetB-silh\GaitDatasetB-silh"
dst=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B"
bad=["005","026","037","048","068","079","088","096","109"]
for sid in bad:
    arc=os.path.join(src,sid+".tar.gz")
    counts=collections.Counter()
    with tarfile.open(arc,"r:gz") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.lower().endswith(".png"):
                parts=m.name.replace("\\","/").split("/")
                if len(parts)>=4:
                    counts[(parts[-3],parts[-2])]+=1
    mism=[]
    for key,n in sorted(counts.items()):
        seq,view=key
        got=len(glob.glob(os.path.join(dst,sid,seq,view,"*.png")))
        if got!=n: mism.append((seq,view,n,got))
    extracted_extra=[]
    print(sid,"archive_png",sum(counts.values()),"pairs",len(counts),"mismatches",len(mism))
    for x in mism[:20]: print(" ",x)
