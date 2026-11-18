import os, glob, json, collections
root=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B"
out=r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\DeepGait_Project\docs\casiab_stats.json"
subjects=sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d)))
condition_frames=collections.Counter()
condition_sequences=collections.Counter()
views=collections.Counter()
total_frames=0
total_sequences=0
for sid in subjects:
    for seq in sorted(os.listdir(os.path.join(root,sid))):
        seqp=os.path.join(root,sid,seq)
        if not os.path.isdir(seqp): continue
        for view in sorted(os.listdir(seqp)):
            vp=os.path.join(seqp,view)
            if not os.path.isdir(vp): continue
            frames=glob.glob(os.path.join(vp,"*.png"))
            if not frames: continue
            cond=seq.split("-")[0]
            n=len(frames)
            total_frames+=n
            total_sequences+=1
            condition_frames[cond]+=n
            condition_sequences[cond]+=1
            views[view]+=1
data={
 "subjects":len(subjects),
 "subject_ids":subjects,
 "total_sequences":total_sequences,
 "total_frames":total_frames,
 "condition_sequences":dict(condition_sequences),
 "condition_frames":dict(condition_frames),
 "view_sequence_counts":dict(sorted(views.items()))
}
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w") as f: json.dump(data,f,indent=2)
print(json.dumps(data,indent=2))
