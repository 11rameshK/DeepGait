from pathlib import Path
import zipfile, json, re, os
root=Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\DeepGait_Project")
required=[
"README.md","requirements.txt","configs/project.yaml",
"src/deepgait_model.py","src/dataset.py","src/losses.py",
"scripts/preprocess_casiab_64x44.py","scripts/validate_pkl_64x44.py","scripts/quick_verify_processed.py",
"scripts/test_model_shapes.py","scripts/smoke_train.py","scripts/train_gpu.py","scripts/evaluate_casiab.py","scripts/infer_query.py",
"outputs/smoke_checkpoint.pt","outputs/smoke_history.json",
"docs/System_Design_and_Methodology.md","docs/FINAL_VERIFICATION_REPORT.md",
"docs/diagrams/DeepGait_Project_Diagrams.drawio","docs/diagrams/DeepGait_Project_Diagrams_Package.zip",
"docs/reference/GGCN_Base_Paper_2026.pdf","docs/review/DeepGait_Review_Presentation.pptx",
"docs/review/DeepGait_Detailed_Progress_Report.pdf","docs/review/DeepGait_Project_Blueprint_Reviewed.xlsx"
]
missing=[x for x in required if not (root/x).exists()]
print("MISSING_REQUIRED",len(missing))
for x in missing: print("MISS",x)

# placeholder / incomplete marker scan
patterns=[r"TODO",r"FIXME",r"NotImplemented",r"pass\s*(#.*)?$",r"YOUR_PATH",r"REPLACE_ME"]
hits=[]
for p in list((root/"src").glob("*.py"))+list((root/"scripts").glob("*.py")):
    if p.name == "deep_audit.py":
        continue
    txt=p.read_text(encoding="utf-8",errors="ignore")
    for pat in patterns:
        if re.search(pat,txt,re.I|re.M):
            hits.append((str(p.relative_to(root)),pat))
print("PLACEHOLDER_HITS",len(hits))
for h in hits: print("HIT",h)

# output smoke history
try:
    h=json.loads((root/"outputs/smoke_history.json").read_text())
    print("SMOKE_HISTORY_ROWS",len(h))
    if h: print("SMOKE_LAST",json.dumps(h[-1]))
except Exception as e: print("SMOKE_HISTORY_ERROR",repr(e))

# diagrams zip
zp=root/"docs/diagrams/DeepGait_Project_Diagrams_Package.zip"
with zipfile.ZipFile(zp) as z:
    print("DIAGRAM_ZIP_BAD",z.testzip())
    print("DIAGRAM_ZIP_ITEMS",len(z.namelist()))
    print("DIAGRAM_ZIP_NAMES","|".join(z.namelist()))

# config sanity
cfg=(root/"configs/project.yaml").read_text()
for token in ["image_height: 64","image_width: 44","train_subjects: [1, 74]","test_subjects: [75, 124]","frames_per_sequence: 30","learning_rate: 0.0001"]:
    print("CONFIG",token,token in cfg)

# size sanity
for rel in ["outputs/smoke_checkpoint.pt","docs/diagrams/DeepGait_Project_Diagrams_Package.zip","docs/reference/GGCN_Base_Paper_2026.pdf","docs/review/DeepGait_Review_Presentation.pptx"]:
    q=root/rel
    print("FILE_SIZE",rel,q.stat().st_size if q.exists() else -1)

print("DEEP_AUDIT_PASS",len(missing)==0 and len(hits)==0)
