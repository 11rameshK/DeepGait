from pathlib import Path
import zipfile, textwrap
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(r"C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\DeepGait_Project")
OUT=ROOT/"docs"/"diagrams"; OUT.mkdir(parents=True,exist_ok=True)
W,H=1600,900
try:
    F=ImageFont.truetype("arial.ttf",20); FB=ImageFont.truetype("arialbd.ttf",20); FT=ImageFont.truetype("arialbd.ttf",30)
except:
    F=FB=FT=ImageFont.load_default()
COL={"blue":"#D9EAF7","green":"#E2F0D9","yellow":"#FFF2CC","orange":"#FCE4D6","purple":"#EADCF8","gray":"#F3F3F3","line":"#35506B","navy":"#1F4E78"}

def wrap(s,n=24):
    out=[]
    for q in s.split("\n"): out+=textwrap.wrap(q,n,break_long_words=False) or [""]
    return out

def box(draw,xy,text,fill):
    x,y,w,h=xy
    draw.rounded_rectangle((x,y,x+w,y+h),12,fill=fill,outline=COL["line"],width=2)
    lines=wrap(text,max(12,w//10)); lh=22; yy=y+(h-len(lines)*lh)/2
    for line in lines:
        bb=draw.textbbox((0,0),line,font=FB); draw.text((x+(w-(bb[2]-bb[0]))/2,yy),line,font=FB,fill="#111"); yy+=lh

def arrow(draw,a,b):
    draw.line((a,b),fill=COL["line"],width=3)
    import math
    ang=math.atan2(b[1]-a[1],b[0]-a[0]); L=13
    p1=(b[0]+L*math.cos(ang+2.6),b[1]+L*math.sin(ang+2.6))
    p2=(b[0]+L*math.cos(ang-2.6),b[1]+L*math.sin(ang-2.6))
    draw.polygon((b,p1,p2),fill=COL["line"])

def center_right(r): x,y,w,h=r; return (x+w,y+h/2)
def center_left(r): x,y,w,h=r; return (x,y+h/2)
def center_top(r): x,y,w,h=r; return (x+w/2,y)
def center_bottom(r): x,y,w,h=r; return (x+w/2,y+h)

pages=[]

# name, nodes [(id,text,rect,fill)], edges [(a,b)]
pages.append(("System Architecture",[
("raw","CASIA-B Raw Silhouettes\n124 Subjects | 11 Views | NM/BG/CL",(60,160,260,100),COL["blue"]),
("pre","Preprocessing\nIntegrity Check\nNormalize + Resize 64x44",(380,150,300,120),COL["yellow"]),
("pkl","Processed PKL Repository\n13,572 valid sequences",(760,100,280,100),COL["purple"]),
("train","Training Pipeline\n30-frame sampling\nMapping + Triplet Loss",(760,250,280,120),COL["orange"]),
("weights","Model Weights / Checkpoints",(1120,260,280,90),COL["green"]),
("query","Query / Probe Sequence",(60,400,260,90),COL["green"]),
("gen","Generate Network",(400,430,240,80),COL["blue"]),
("enc","Encoder Network",(720,430,240,80),COL["blue"]),
("map","Feature Mapping\nAFRE + TPFE\nEmbedding 256x63",(1040,415,280,110),COL["blue"]),
("match","Euclidean Distance Matching",(1040,610,280,90),COL["yellow"]),
("out","Predicted Subject ID",(1040,760,280,80),COL["green"]),
("gallery","Gallery Embeddings",(1360,600,200,90),COL["purple"])
],[("raw","pre"),("pre","pkl"),("pkl","train"),("train","weights"),("query","pre"),("pre","gen"),("gen","enc"),("enc","map"),("weights","map"),("map","match"),("gallery","match"),("match","out"),("weights","gallery")]))

pages.append(("Methodology Workflow",[
("a","Start",(690,90,220,60),COL["green"]),
("b","Acquire CASIA-B Dataset",(620,190,360,70),COL["blue"]),
("c","Extract + Integrity Verify",(620,300,360,70),COL["yellow"]),
("d","Preprocess Silhouettes\nCrop | Normalize | 64x44",(620,410,360,80),COL["blue"]),
("e","Create + Validate PKLs",(620,530,360,70),COL["orange"]),
("f","Train/Test Split\n001-074 / 075-124",(620,640,360,80),COL["green"]),
("g","Load 30-Frame Batches",(620,760,360,70),COL["blue"]),
("h","Train DeepGait\nGenerate + Encoder + AFRE/TPFE",(1080,700,360,90),COL["yellow"]),
("i","Compute Mapping + Triplet Loss",(1080,560,360,80),COL["orange"]),
("j","Save Checkpoints",(1080,430,360,70),COL["purple"]),
("k","Build Gallery/Probe Embeddings",(1080,300,360,80),COL["blue"]),
("l","Evaluate Euclidean Matching\nNM / BG / CL Rank-1",(1080,170,360,80),COL["green"]),
("m","End",(1150,70,220,60),COL["green"])
],[("a","b"),("b","c"),("c","d"),("d","e"),("e","f"),("f","g"),("g","h"),("h","i"),("i","j"),("j","k"),("k","l"),("l","m")]))

pages.append(("Logical Component Diagram",[
("d1","Data Layer\nCASIA-B + Processed PKLs",(60,150,280,95),COL["blue"]),
("d2","Preprocessing\nIntegrity | Normalize | 64x44",(390,150,280,95),COL["yellow"]),
("t1","Batch Builder\n30-frame NM/BG/CL",(720,150,260,95),COL["orange"]),
("m1","Generate Network",(1030,150,240,80),COL["blue"]),
("m2","Encoder Network",(1320,150,240,80),COL["blue"]),
("m3","AFRE + TPFE",(1320,300,240,80),COL["blue"]),
("m4","Embedding Fusion\n256x63",(1030,300,240,90),COL["blue"]),
("t2","Loss Engine\nMapping + Triplet",(720,440,260,90),COL["orange"]),
("t3","Optimizer + Checkpoints",(390,440,280,90),COL["purple"]),
("e1","Gallery Builder",(1030,540,240,80),COL["orange"]),
("e2","Distance Matcher",(1320,540,240,80),COL["yellow"]),
("o","Results + Predicted\nSubject ID",(1320,700,240,90),COL["green"])
],[("d1","d2"),("d2","t1"),("t1","m1"),("m1","m2"),("m2","m3"),("m3","m4"),("m4","t2"),("t2","t3"),("t3","m1"),("m4","e1"),("e1","e2"),("m4","e2"),("e2","o")]))

pages.append(("Behavioural Activity Diagram",[
("s","Start",(80,100,200,60),COL["green"]),
("task","Choose Training or Identification",(80,220,260,80),COL["yellow"]),
("train","Training Path\nLoad PKLs -> Batch -> Model -> Loss -> Update -> Checkpoint",(480,150,500,130),COL["orange"]),
("infer","Identification Path\nLoad Query -> Preprocess -> Embedding -> Gallery Match",(480,390,500,120),COL["blue"]),
("repo","Repositories\nPKLs | Checkpoints | Gallery Embeddings",(1120,260,350,120),COL["purple"]),
("result","View Predicted Subject ID",(480,640,500,80),COL["green"]),
("end","End",(620,780,220,60),COL["green"])
],[("s","task"),("task","train"),("task","infer"),("train","repo"),("repo","infer"),("infer","result"),("result","end")]))

pages.append(("Data Flow Diagram Level 0",[
("u","Student / Researcher",(60,330,260,80),COL["green"]),
("p1","P1 Dataset Preparation",(450,120,300,80),COL["blue"]),
("p2","P2 Model Training",(450,300,300,80),COL["yellow"]),
("p3","P3 Evaluation & Identification",(450,480,320,80),COL["orange"]),
("p4","P4 Reporting Assets",(450,660,300,80),COL["purple"]),
("d1","D1 Raw CASIA-B Dataset",(980,120,320,80),COL["blue"]),
("d2","D2 Processed PKLs",(980,300,320,80),COL["blue"]),
("d3","D3 Model Checkpoints",(980,480,320,80),COL["blue"]),
("d4","D4 Results / Diagrams / PPT",(980,660,320,80),COL["blue"]),
("g","Project Guide / PRC",(1360,660,200,80),COL["green"])
],[("u","p1"),("u","p2"),("u","p3"),("u","p4"),("p1","d1"),("d1","p1"),("p1","d2"),("d2","p2"),("p2","d3"),("d2","p3"),("d3","p3"),("p3","d4"),("p4","d4"),("d4","g")]))

pages.append(("Data Repository Structure - ER Not Applicable",[
("a","Source Archives\nZIP / TAR.GZ",(80,260,260,90),COL["blue"]),
("b","Raw CASIA-B Folders\nSubject -> Sequence -> View -> PNG",(390,250,390,120),COL["yellow"]),
("c","Processed PKL Repository\nSubject -> Sequence -> View -> PKL",(830,250,390,120),COL["green"]),
("d","Model Checkpoints\nWeights + Optimizer + Config",(1270,250,300,120),COL["purple"]),
("e","Evaluation Outputs\nLogs + Results + Diagrams + PPT",(650,520,400,110),COL["orange"]),
("n","No SQL database is present in the current implementation. Therefore a classical ER diagram would fabricate entities/tables. This repository structure is the accurate storage model.",(300,720,1000,100),COL["gray"])
],[("a","b"),("b","c"),("c","d"),("c","e"),("d","e")]))

# draw PNGs
for idx,(title,nodes,edges) in enumerate(pages,1):
    img=Image.new("RGB",(W,H),"white"); dr=ImageDraw.Draw(img)
    dr.rounded_rectangle((20,20,W-20,70),14,fill=COL["navy"])
    bb=dr.textbbox((0,0),title,font=FT); dr.text(((W-(bb[2]-bb[0]))/2,28),title,font=FT,fill="white")
    by={i:(t,r,f) for i,t,r,f in nodes}
    for i,t,r,f in nodes: box(dr,r,t,f)
    for a,b in edges:
        ra=by[a][1]; rb=by[b][1]
        scx,scy=ra[0]+ra[2]/2,ra[1]+ra[3]/2
        tcx,tcy=rb[0]+rb[2]/2,rb[1]+rb[3]/2
        dx,dy=tcx-scx,tcy-scy
        if abs(dx) >= abs(dy):
            if dx >= 0:
                ax,ay=center_right(ra); bx,byy=center_left(rb)
            else:
                ax,ay=center_left(ra); bx,byy=center_right(rb)
        else:
            if dy >= 0:
                ax,ay=center_bottom(ra); bx,byy=center_top(rb)
            else:
                ax,ay=center_top(ra); bx,byy=center_bottom(rb)
        arrow(dr,(ax,ay),(bx,byy))
    img.save(OUT/f"{idx:02d}_{title.replace(' ','_').replace('/','_')}.png")

# drawio XML
mx=Element("mxfile",{"host":"app.diagrams.net","version":"24.7.17","type":"device"})
for pi,(title,nodes,edges) in enumerate(pages,1):
    dia=SubElement(mx,"diagram",{"id":f"p{pi}","name":title})
    model=SubElement(dia,"mxGraphModel",{"page":"1","pageWidth":"1800","pageHeight":"1000","grid":"1","gridSize":"10"})
    rt=SubElement(model,"root"); SubElement(rt,"mxCell",{"id":"0"}); SubElement(rt,"mxCell",{"id":"1","parent":"0"})
    ids={}
    n=2
    for key,text,r,fill in nodes:
        cid=str(n); n+=1; ids[key]=cid
        cell=SubElement(rt,"mxCell",{"id":cid,"value":text.replace("\n","&#xa;"),"style":f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={COL['line']};fontSize=16;align=center;verticalAlign=middle;","vertex":"1","parent":"1"})
        x,y,w,h=r; SubElement(cell,"mxGeometry",{"x":str(x),"y":str(y),"width":str(w),"height":str(h),"as":"geometry"})
    for a,b in edges:
        cell=SubElement(rt,"mxCell",{"id":str(n),"style":f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=2;strokeColor={COL['line']};","edge":"1","parent":"1","source":ids[a],"target":ids[b]}); n+=1
        SubElement(cell,"mxGeometry",{"relative":"1","as":"geometry"})
xml=minidom.parseString(tostring(mx,encoding="utf-8")).toprettyxml(indent="  ")
(OUT/"DeepGait_Project_Diagrams.drawio").write_text(xml,encoding="utf-8")
(OUT/"README.txt").write_text("DeepGait verified diagrams. ER note: current project is file-based and has no relational database tables.\n",encoding="utf-8")
zip_path=OUT/"DeepGait_Project_Diagrams_Package.zip"
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.name!=zip_path.name: z.write(p,p.name)
print("DIAGRAMS_OK",zip_path)
