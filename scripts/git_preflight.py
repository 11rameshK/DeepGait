from pathlib import Path
import re, subprocess, json, os
root=Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\DeepGait_Project")
large=[]
for f in root.rglob("*"):
    if f.is_file() and ".git" not in f.parts:
        try:
            sz=f.stat().st_size
        except OSError:
            continue
        if sz>=50*1024*1024:
            large.append((str(f.relative_to(root)),sz))
print("LARGE_GE_50MB",len(large))
for x in sorted(large,key=lambda x:-x[1]): print("LARGE",x[0],x[1])

patterns={
 "github_token":re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
 "private_key":re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
 "aws_key":re.compile(r"AKIA[0-9A-Z]{16}"),
 "password_assignment":re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
}
hits=[]
for f in root.rglob("*"):
    if not f.is_file() or ".git" in f.parts:
        continue
    if f.suffix.lower() not in {".py",".md",".txt",".yaml",".yml",".json",".bat",".ps1",".gitignore"}:
        continue
    try:
        txt=f.read_text(encoding="utf-8",errors="ignore")
    except Exception:
        continue
    for name,rx in patterns.items():
        if rx.search(txt):
            hits.append((str(f.relative_to(root)),name))
print("SECRET_HITS",len(hits))
for h in hits: print("SECRET",*h)

r=subprocess.run(["git","status","--short","--ignored"],cwd=root,text=True,capture_output=True)
print("GIT_STATUS_BEGIN")
print(r.stdout)
print("GIT_STATUS_END")
