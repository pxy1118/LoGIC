"""Rebuild the five result figures from the frozen geometry, FEA and CFD studies."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import sys

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import pymupdf

INK="#203646"; MUT="#435E6D"; GRID="#D8E3E7"; TEAL="#227F82"; BLUE="#266F9F"
CLAY="#B96139"; PURPLE="#846492"; GREEN="#657E49"
TOPO=["Gyroid","Diamond","Primitive","Mixed"]
TC=[BLUE,CLAY,"#AD557F",TEAL]; MARK=["o","D","^","s"]
SPEED=LinearSegmentedColormap.from_list("speed",["#253B65","#286E88","#369792","#8AC393","#F2D382"])
PRESS=LinearSegmentedColormap.from_list("pressure",["#D8DCEC","#B4BEDC","#8797C3","#666CA2","#423E72"])
ERROR=LinearSegmentedColormap.from_list("error",["#F5E2BE","#E5AE74","#C4744C","#984440","#612C43"])
mpl.rcParams.update({"font.family":"Arial","font.size":7.7,"axes.labelsize":7.7,"xtick.labelsize":7.1,
 "ytick.labelsize":7.1,"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,"ytick.color":MUT,
 "axes.edgecolor":"#8098A4","axes.linewidth":.65,"xtick.major.size":2.6,"ytick.major.size":2.6,
 "axes.spines.top":False,"axes.spines.right":False,"mathtext.fontset":"stix","pdf.fonttype":42,
 "svg.fonttype":"none","svg.hashsalt":"logic-results-20261118","image.composite_image":False,
 "savefig.facecolor":"white","path.simplify":False})

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))
def col(data,key):
    return np.array([float(x[key]) for x in data])
def dt(v):
    return f"{v:g}".replace("-","−")

class Page:
    """Layout in millimetres from the upper left of a fixed 183 mm artboard."""
    def __init__(self,num,height):
        self.num=num; self.h=height; self.w=183
        self.fig=plt.figure(figsize=(183/25.4,height/25.4),dpi=300)
        self.axes={}
        self.info={}
    def pos(self,x,y):
        return x/self.w,1-y/self.h
    def ax(self,x,y,w,h,ident=None):
        a=self.fig.add_axes([x/self.w,1-(y+h)/self.h,w/self.w,h/self.h])
        if ident: self.axes[ident]=a
        a.set_axisbelow(True)
        a.tick_params(pad=2)
        return a
    def text(self,x,y,s,fs=7.4,bold=False,ha="left",color=INK):
        return self.fig.text(*self.pos(x,y),s,ha=ha,va="center",fontsize=fs,
                             color=color,weight="bold" if bold else "normal")
    def panel(self,x,y,letter,title):
        self.text(x+2.2,y+3.65,letter,10.2,True)
        self.text(x+8.4,y+3.65,title,8.45,True)
    def key(self,x,y,label,color,mark=None,ls="-",width=6,band=None):
        fx,fy=self.pos(x,y); tx,_=self.pos(x+width,y)
        if band:
            self.fig.add_artist(Rectangle(self.pos(x,y+1.0),width/self.w,2/self.h,
                                transform=self.fig.transFigure,fc=band,ec="none"))
        self.fig.add_artist(Line2D([fx,tx],[fy,fy],color=color,ls=ls,lw=1.35,
                transform=self.fig.transFigure))
        if mark:
            self.fig.add_artist(Line2D([(fx+tx)/2],[fy],marker=mark,color=color,ms=4.4,ls="none",
                                       transform=self.fig.transFigure))
        self.text(x+width+1.5,y,label,7.1)
    def colorbar(self,x,y,w,cmap,lo,hi,ticks):
        a=self.ax(x,y,w,2.3)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(lo,hi),cmap=cmap),
                          cax=a,orientation="horizontal",ticks=ticks)
        a.tick_params(labelsize=6.8,pad=1,length=2)
        a.spines["outline"].set_visible(False)
    def save(self,rg,cg,files):
        out=OUT/f"fig{self.num}"; qa=out/"provenance"
        out.mkdir(parents=True,exist_ok=True); qa.mkdir(parents=True,exist_ok=True)
        stem=out/f"Fig{self.num}"
        self.fig.savefig(stem.with_suffix(".pdf"),dpi=600,metadata={"Title":f"LoGIC Figure {self.num}"})
        svg=stem.with_suffix(".svg")
        self.fig.savefig(svg,dpi=600,metadata={"Date":None})
        txt=re.sub(r'<!DOCTYPE\s+svg\s+PUBLIC[^>]*>\s*','',svg.read_text(encoding="utf-8"))
        def font(m):
            el=m[0]; size=re.search(r"font-size:\s*([0-9.]+)px;?",el)
            if size is None: return el
            el=el[:size.start()]+el[size.end():]
            return el.replace(" ",f' font-size="{size[1]}" ',1)
        svg.write_text(re.sub(r'<(?:text|tspan)\b[^>]*>',font,txt),encoding="utf-8")
        plt.close(self.fig)
        with pymupdf.open(stem.with_suffix(".pdf")) as doc:
            doc[0].get_pixmap(dpi=600,alpha=False).save(stem.with_suffix(".png"))
        with Image.open(stem.with_suffix(".png")) as im:
            im.save(stem.with_suffix(".tif"),compression="tiff_lzw",dpi=(600,600))
            im.thumbnail((1250,1000),Image.Resampling.LANCZOS); im.save(qa/"preview.png")
        (qa/"source.json").write_text(json.dumps(dict(self.info,files={
            source_name(f):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files}),indent=2),encoding="utf-8")

def grid(a):
    a.grid(axis="y",lw=.52,color=GRID)
def parity(a,lo,hi,ticks):
    a.plot([lo,hi],[lo,hi],ls=(0,(4,3)),lw=.85,color="#768994",zorder=1)
    a.set(xlim=(lo,hi),ylim=(lo,hi),xticks=ticks,yticks=ticks)
def twins(a,y,m1,m2,col1=TEAL,col2=CLAY):
    a.hlines(y,m1,m2,color="#AFC1C9",lw=1.2)
    a.scatter(m2,y,s=23,marker="D",c=col2,zorder=3,lw=.55,edgecolor="white")
    a.scatter(m1,y,s=15,marker="o",c=col1,zorder=4,lw=.45,edgecolor="white")

def fig7():
    p=Page(7,144)
    for x,y,l,s in [(2,2,"a","Prescribed orientation fields"),(93,2,"b","Error across rotation spans"),
                       (2,72,"c","Phase-Jacobian stretch"),(93,72,"d","Target and realized axes")]:
        p.panel(x,y,l,s)
    path=G/"orientation.csv"; data={r["case_id"]:r for r in rows(path) if r["resolution"]=="240"}
    assert len(data)==12
    base=[data[f"orientation_constant_{i}"] for i in [0,15,30,45]]
    others=[data["orientation_"+s] for s in ["gradient_z_15","gradient_z_30","gradient_z_40","gradient_z_60",
              "gradient_x_30","gradient_y_30","twist_z_30","multiaxis"]]
    a=p.ax(26,19,58,40,"a"); mean="local_z_error_mean_deg"; mx="local_z_error_max_deg"
    mm=np.r_[col(base,mean).mean(),col(others,mean)]; high=np.r_[col(base,mx).max(),col(others,mx)]
    y=np.arange(len(mm)); twins(a,y,mm,high)
    a.axvline(10,ls=(0,(3,2)),lw=.85,color="#72858B")
    a.set(yticks=y,yticklabels=["Constant (4)","z, 0–15°","z, 0–30°","z, 0–40°","z, 0–60°","x, 0–30°",
                                "y, 0–30°","z twist","Multiaxis"],ylim=(8.55,-.55),xlim=(-.5,15.2),
          xticks=[0,5,10,15],xlabel="Local z-axis error (°)")
    a.tick_params(axis="y",length=0)
    p.key(9,13.2,"Mean",TEAL,"o","none",4);p.key(31,13.2,"Maximum",CLAY,"D","none",4)
    p.key(59,13.2,"10° threshold","#72858B",ls="--")
    spans=np.array([15,30,40,60]); b=p.ax(109,19,65,40,"b")
    for vals,c,m in [(col(others[:4],mean),TEAL,"o"),(col(others[:4],mx),CLAY,"D")]:
        b.plot(spans,vals,color=c,marker=m,ms=4.3,lw=1.45)
    grid(b); b.set(xticks=spans,xlim=(11,64),ylim=(0,12),yticks=[0,3,6,9,12],
                  xlabel="Prescribed z-gradient span (°)",ylabel="Local z-axis error (°)")
    p.key(110,13.2,"Mean",TEAL,"o");p.key(137,13.2,"Maximum",CLAY,"D")
    c=p.ax(26,90,58,41,"c")
    for k,co,m in [("min_phase_stretch",TEAL,"o"),("max_phase_stretch",CLAY,"D")]:
        c.plot(spans,col(others[:4],k),color=co,marker=m,lw=1.45,ms=4.3)
    c.axhline(1,color="#7C929B",lw=.8,ls="--");grid(c)
    c.set(xticks=spans,xlim=(11,64),ylim=(.60,1.40),yticks=[.6,.8,1,1.2,1.4],
          xlabel="Prescribed z-gradient span (°)",ylabel="Singular value of the Jacobian")
    p.key(12,83.7,"Minimum",TEAL,"o");p.key(47,83.7,"Maximum",CLAY,"D")
    folder=G/"cases/orientation_gradient_z_40"; files=[folder/"geometry.npz",folder/"orientation_nodes.npz"]
    with np.load(files[0]) as v: mask=v["mask"]
    with np.load(files[1]) as v:
        mid=list(v["indices"]).index(mask.shape[1]//2);sel=(slice(1,-1,2),mid,slice(1,-1,2))
        pos=v["positions_mm"][sel]; target=v["target_axes"][sel];actual=v["actual_axes"][sel]
    d=p.ax(123.5,90,41,41,"d")
    d.imshow(mask[:,mask.shape[1]//2,:].T,cmap=ListedColormap(["#FFFFFF","#D1DBDE"]),origin="lower",
             extent=(0,7.5,0,7.5),interpolation="nearest")
    for mat,co,width in [(target,BLUE,.013),(actual,CLAY,.0055)]:
        d.quiver(pos[...,0],pos[...,2],mat[...,0,2],mat[...,2,2],angles="xy",scale_units="xy",
                 scale=1.7,width=width,color=co,pivot="mid",headwidth=2.8)
    d.set(xlabel="x (mm)",ylabel="z (mm)",xticks=[0,2.5,5,7.5],yticks=[0,2.5,5,7.5])
    p.key(103,83.7,"Target",BLUE);p.key(129,83.7,"Realized",CLAY);p.text(157,83.7,"0–40° field",7.0,color=MUT)
    p.info={"cases":12,"constant_cases_grouped":4,"slice_glyphs":int(pos.shape[0]*pos.shape[1])}
    p.save([["a","b"],["c","d"]],[["a","c"]],[path,*files])

SPATIAL=["topology_gradient","density_gradient_x","density_gradient_y","density_gradient_z",
         "coupled_three_controls","coupled_three_topologies"]
def fig8():
    p=Page(8,145)
    for x,y,l,s in [(2,2,"a","Uniform density targets"),(93,2,"b","Global and regional errors"),
                       (2,73,"c","Cell-scale density control"),(93,73,"d","Coupled density profile")]:
        p.panel(x,y,l,s)
    path=G/"density.csv"; dens=[r for r in rows(path) if r["resolution"]=="240"]
    uniform=[r for r in dens if r["section"]=="uniform_density"]; assert len(uniform)==20
    a=p.ax(25,23,59,32,"a"); types=["Gyroid","Diamond","Primitive","Gyroid+Diamond"]
    values=np.array([col(sorted([r for r in uniform if r["topology"]==t],key=lambda q:float(q["target_volume_mean"])),"mesh_svf")*100 for t in types])
    targets=np.array([25,30,35,40,50]);err=values-targets
    cm=LinearSegmentedColormap.from_list("signed",["#A9CEDE","#F7F8F5","#F5C09E"])
    a.imshow(err,cmap=cm,vmin=-.1,vmax=.1,aspect="auto",interpolation="nearest")
    for i,j in np.ndindex(values.shape):
        a.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",fontsize=6.9)
    for z in np.arange(.5,4.5): a.axvline(z,color="white",lw=1.15)
    for z in np.arange(.5,3.5): a.axhline(z,color="white",lw=1.15)
    a.set(xticks=range(5),xticklabels=targets,yticks=range(4),yticklabels=types[:3]+["G + D"],xlabel="Target solid volume fraction (%)")
    a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]
    p.text(4,14.8,"Signed error (pp)",7.1)
    p.colorbar(45,12,32,cm,-.1,.1,[-.1,0,.1])
    p.text(25,20,"Cell values give exported SVF (%)",7,color=MUT)
    b=p.ax(118,23,58,32,"b")
    lookup={r["case_id"]:r for r in dens};sp=[lookup[n] for n in SPATIAL]
    twins(b,np.arange(6),abs(col(sp,"global_error_pp")),col(sp,"regional_mae_pp"))
    b.set(yticks=np.arange(6),yticklabels=["Topology","Density x","Density y","Density z","All fields","G/D/P + R"],
          ylim=(5.6,-.6),xlim=(-.18,6),xticks=[0,2,4,6],xlabel="SVF error (percentage points)")
    b.tick_params(axis="y",length=0);p.key(103,15,"Global |error|",TEAL,"o","none",4);p.key(137,15,"Regional MAE",CLAY,"D","none",4)
    regpath=G/"regional_density.csv"; regs=[r for r in rows(regpath) if r["kind"]=="generated" and r["bins"]=="3" and r["case_id"] in SPATIAL]
    assert all(sum(r["case_id"]==n for r in regs)==27 for n in SPATIAL)
    c=p.ax(28,92,39,39,"c"); parity(c,19,49,[20,30,40])
    assert np.all((col(regs,"measured_svf")>=.19)&(col(regs,"measured_svf")<=.49))
    assert np.all((col(regs,"target_svf")>=.19)&(col(regs,"target_svf")<=.49))
    for names,co,m in [([SPATIAL[0]],BLUE,"o"),(SPATIAL[1:4],GREEN,"^"),(SPATIAL[4:],CLAY,"D")]:
        rr=[r for r in regs if r["case_id"] in names]
        c.scatter(col(rr,"target_svf")*100,col(rr,"measured_svf")*100,s=19,marker=m,facecolor="none",
                  edgecolor=co,linewidth=.75,alpha=.84,zorder=3)
    c.set(xlabel="Regional target SVF (%)",ylabel="Regional measured SVF (%)")
    p.key(5,85.1,"Topology",BLUE,"o","none",4);p.key(33,85.1,"Density",GREEN,"^","none",4);p.key(59,85.1,"Coupled",CLAY,"D","none",4)
    profile=G/"cases/coupled_three_controls/profiles.npz"
    d=p.ax(109,92,65,39,"d")
    with np.load(profile) as src:
        z,tar,meas=[src["2_"+s] for s in ["coordinate_mm","target_svf","measured_svf"]]
    d.plot(z,meas*100,color=TEAL,lw=1.6);d.plot(z,tar*100,color=INK,lw=1.15,ls=(0,(4,2)))
    grid(d);d.set(xlabel="Slab centre, z (mm)",ylabel="Slab-averaged SVF (%)",xlim=(1,6.5),xticks=[1,2,3,4,5,6],
                  ylim=(20,50),yticks=[20,30,40,50])
    p.key(109,85.1,"Target",INK,ls="--");p.key(141,85.1,"Measured",TEAL)
    p.info={"uniform_structures":20,"spatial_structures":6,"regions":len(regs),"slab_width_mm":float(src_width(profile))}
    p.save([["a","b"],["c","d"]],[],[path,regpath,profile])

def src_width(path):
    with np.load(path) as z: return z["2_window_mm"].item()

def fig9():
    p=Page(9,146)
    pred=F/"test_curves.npz"; samplefile=F/"test_structures.csv"
    with np.load(pred) as z:
        ids=z["sample_ids"]; truth=z["truth"]; members=z["members"]
    assert members.shape==(5,62,1001) and truth.shape==(62,1001)
    mean=members.mean(0); sd=members.std(0,ddof=1); delta=mean-truth
    rmse=np.sqrt(np.mean(delta**2,axis=1)); rank=np.argsort(rmse,kind="stable")
    chosen=rank[[0,(len(ids)-1)//2,-1]]; assert ids[chosen].tolist()==read(F/"figure9_examples.json")["public_sample_ids"]
    xstrain=np.linspace(0,5,1001)
    p.key(7,14.1,"FEA",INK,width=6)
    p.key(28,14.1,"Ensemble mean",TEAL,ls="--",width=6)
    p.key(68,14.1,"±1 SD across five seeds","#82B9B6",ls="none",width=6,band="#C2DED9")
    p.key(123,14.1,"Δ = prediction − FEA",CLAY,width=6)
    chosen_metrics=[]
    for j,(idx,title) in enumerate(zip(chosen,["Best","Median rank","Worst"])):
        x=2+60*j; aletter="abc"[j];p.panel(x,2,aletter,f"{title} · ID {ids[idx]}")
        a=p.ax(x+11,22,43,28,aletter)
        a.fill_between(xstrain,mean[idx]-sd[idx],mean[idx]+sd[idx],color="#9FCCC6",alpha=.80,lw=0)
        a.plot(xstrain,truth[idx],color=INK,lw=1.30)
        a.plot(xstrain,mean[idx],color=TEAL,lw=1.25,ls=(0,(4,2.6)))
        a.set(xlim=(0,5),ylim=(0,260),xticks=range(6),yticks=[0,100,200])
        if j==0: a.set_ylabel("Stress (MPa)")
        grid(a); a.tick_params(axis="x",bottom=False,labelbottom=False)
        p.text(x+11,19.1,f"RMSE  {rmse[idx]:.2f} MPa",7.35,True,color=MUT)
        # The identical signed-residual scale makes the ranked examples comparable.
        b=p.ax(x+11,53.4,43,10.4,"r"+aletter)
        b.axhline(0,color="#80959F",lw=.65,ls="--")
        b.plot(xstrain,delta[idx],color=CLAY,lw=1.05)
        b.set(xlim=(0,5),xticks=range(6),ylim=(-4.5,25),yticks=[0,20],xlabel="Strain (%)")
        if j==0: b.set_ylabel("Δ (MPa)",fontsize=7.2)
        chosen_metrics.append(dict(id=int(ids[idx]),rmse_MPa=float(rmse[idx])))
        assert delta[idx].min()>-4.5 and delta[idx].max()<25
        assert truth[idx].max()<260 and (mean[idx]+sd[idx]).max()<260
    pairs=[r for r in rows(samplefile) if r["member"]=="ensemble"]
    assert len(pairs)==len(set(int(r["sample_id"]) for r in pairs))==62
    assert set(ids)==set(int(r["sample_id"]) for r in pairs)
    specs=[("elastic_modulus_MPa",.001,"Effective modulus","GPa",7,33,[10,20,30]),
           ("yield_strength_MPa",1,"0.2% offset yield strength","MPa",80,270,[100,150,200,250]),
           ("W5_MJ_m3",1,"Loading work to 5%","MJ/m³",4,14,[4,7,10,13])]
    for j,(k,fac,name,unit,lo,hi,ticks) in enumerate(specs):
        x=2+60*j; letter="def"[j];p.panel(x,77.4,letter,name)
        obs=col(pairs,"reference_"+k)*fac; yh=col(pairs,"predicted_"+k)*fac
        assert np.isfinite([obs,yh]).all() and min(obs.min(),yh.min())>=lo and max(obs.max(),yh.max())<=hi
        mape=np.mean(abs((yh-obs)/obs))*100
        p.text(x+11,90.0,f"MAPE  {mape:.2f}%",7.35,True,color=MUT)
        a=p.ax(x+11,94.2,39,39,letter);parity(a,lo,hi,ticks)
        a.scatter(obs,yh,s=15,fc=TEAL,ec="white",lw=.38,alpha=.80,zorder=3)
        a.set(xlabel=f"FEA ({unit})",ylabel=f"Predicted ({unit})")
    p.info={"structures":62,"ensemble_members":5,"curves":chosen_metrics,
            "band":"pointwise sample SD (ddof=1)","residual":"ensemble minus FEA, MPa"}
    p.save([list("abc"),["ra","rb","rc"],list("def")],[["a","ra"],["b","rb"],["c","rc"]],[pred,samplefile])

def fig10():
    p=Page(10,112)
    source=C/"per_structure.csv"; dat=rows(source); ids=[int(r["sample_id"]) for r in dat]
    assert len(set(ids))==len(ids)==75
    snapshots=[CFD/f"samples/{i}/sample.json" for i in ids]
    fam=[read(f)["topology_family"] for f in snapshots]
    parts=[set("+".join(a).split("+")) for a in fam]
    assert all(a and a<=set(TOPO[:3]) for a in parts)
    groups=np.array([next(iter(a)) if len(a)==1 else "Mixed" for a in parts])
    assert set(groups)==set(TOPO)
    stats=C/"main_metrics.csv"; recorded={r["metric"]:r for r in rows(stats)}
    ensemble_metrics={}
    for j,(key,mul,title,unit,lim,ticks) in enumerate([
        ("K",1e9,"Apparent flow-domain permeability",r"$10^{-9}$ m²",11,[0,2,4,6,8,10]),
        ("DP",1,"Pressure drop","Pa",6.25,[0,1,2,3,4,5,6])]):
        x=2+91*j; l="ab"[j];p.panel(x,2,l,title)
        u=col(dat,key+"_true")*mul; v=col(dat,key+"_pred")*mul
        assert np.all(np.r_[u,v]>=0) and max(u.max(),v.max())<lim
        e=(v/u-1)*100; mape=abs(e).mean(); r2=1-((v-u)**2).sum()/((u-u.mean())**2).sum()
        assert abs(mape-float(recorded[key+"_mape_percent"]["ensemble"]))<1e-5
        p.text(x+15,14.1,f"R²  {r2:.3f}     MAPE  {mape:.2f}%",7.8,True,color=MUT)
        a=p.ax(x+21,19,48,48,l);parity(a,0,lim,ticks);a.set_ylabel(f"Predicted ({unit})")
        a.tick_params(axis="x",bottom=False,labelbottom=False)
        b=p.ax(x+21,72,48,14,l+"r")
        b.axhline(0,color=MUT,lw=.7,ls="--")
        b.set(xlim=(0,lim),ylim=(-32,32),xticks=ticks,yticks=[-25,0,25],
              xlabel=f"CFD reference ({unit})",ylabel="Error (%)")
        for famname,co,m in zip(TOPO,TC,MARK):
            sel=groups==famname
            for ax,xx,yy,size in [(a,u[sel],v[sel],23),(b,u[sel],e[sel],14)]:
                ax.scatter(xx,yy,s=size,marker=m,c=co,ec="white",lw=.48,alpha=.85,zorder=3)
        ensemble_metrics[key]={"R2":float(r2),"MAPE_percent":float(mape)}
    for x,n,co,m in zip([15,54,96,139],TOPO,TC,MARK):
        p.key(x,103,f"{n} (n = {(groups==n).sum()})",co,m,"none",4)
    p.info={"structures":75,"ensemble_members":3,"ensemble":ensemble_metrics,
            "residual":"100 × (ensemble / CFD − 1)","mixed":"any mixture of two or three topology families"}
    p.save([["a","b"],["ar","br"]],[["a","ar"],["b","br"]],[source,stats,*snapshots])

def fig11():
    p=Page(11,149)
    sid=38; folder=CFD/f"samples/{sid}"
    truth_file=folder/"field.npz"; metrics=folder/"metrics.json"; pred_file=C/f"predictions/{sid}.npz"
    protocol=C/"test_design.json"; assert sid in read(protocol)["representative_test_ids"]
    with np.load(truth_file) as src,np.load(pred_file) as pr:
        k=src["mask"].shape[-1]//2
        mask=src["mask"][:,:,k].astype(bool)
        assert np.array_equal(mask,pr["target_valid"][:,:,k]) and int(pr["sample_id"])==sid
        ori,ds=src["origin_mm"],src["spacing_mm"]
        ref=np.stack([src[n][:,:,k] for n in ["u","v","w","p"]]);ref[3]-=read(metrics)["pressure_outlet_Pa"]
        pred=pr["field"][:,:,:,k]
    # Array values are defined at query nodes; the four bounds are pixel edges.
    lims=np.ravel(np.column_stack([ori[:2]-.5*ds[:2],ori[:2]+(np.array(mask.shape)-.5)*ds[:2]]))
    specs=[
       (np.linalg.norm(ref[:3],axis=0)*1000,np.linalg.norm(pred[:3],axis=0)*1000,SPEED,"speed","mm/s"),
       (ref[3],pred[3],PRESS,"pressure","Pa")]
    ranges=[]
    for row,(obs,hat,cmap,kind,unit) in enumerate(specs):
        y=2+73.4*row
        err=np.abs(hat-obs); lo=float(min(obs[mask].min(),hat[mask].min()))
        hi=float(max(obs[mask].max(),hat[mask].max())); emax=float(err[mask].max())
        assert np.isfinite([lo,hi,emax]).all()
        ranges.append(dict(quantity=kind,shared=[lo,hi],absolute_error=[0,emax]))
        for j,(arr,title,cm,low,high) in enumerate([(obs,f"CFD {kind}",cmap,lo,hi),
                (hat,f"Predicted {kind}",cmap,lo,hi),(err,f"Absolute {kind} error",ERROR,0,emax)]):
            x=2+60*j;letter="abcdef"[row*3+j]
            p.panel(x,y,letter,title)
            a=p.ax(x+10.9,y+12,39,39,letter)
            c=cm.copy();c.set_bad("white")
            a.imshow(np.ma.array(arr.T,mask=~mask.T),origin="lower",extent=lims,cmap=c,
                     vmin=low,vmax=high,interpolation="nearest",aspect="auto")
            a.set(xticks=[0,2,4,6],yticks=[0,2,4,6],xlabel="x (mm)")
            if j==0: a.set_ylabel("y (mm)")
            else: a.tick_params(axis="y",labelleft=False)
        ticks=[0,2,4,6,8] if row==0 else [.5,.6,.7,.8]
        eticks=[0,1,2,3] if row==0 else [0,.1,.2]
        p.colorbar(15,y+61,93,cmap,lo,hi,ticks); p.colorbar(136,y+61,33,ERROR,0,emax,eticks)
        text=("Speed" if row==0 else "Outlet-referenced pressure")
        p.text(61.5,y+70.6,f"{text} ({unit}) · shared scale",7.1,ha="center")
        p.text(152.5,y+70.6,f"Absolute error ({unit})",7.1,ha="center")
    p.info={"sample":38,"ensemble_members":3,"slice":"native x–y plane at z index 32",
            "valid_pixels":int(mask.sum()),"scales":ranges,"interpolation":"nearest","mask":"original valid fluid mask"}
    p.save([list("abc"),list("def")],[["a","d"],["b","e"],["c","f"]],[truth_file,metrics,pred_file,protocol])

def source_name(path):
    for prefix, parent in [("results", RESULTS), ("cfd", CFD)]:
        if path.is_relative_to(parent):
            return prefix + "/" + path.relative_to(parent).as_posix()
    raise ValueError(path)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("figures",nargs="*",type=int,choices=None)
    parser.add_argument("--release", type=Path, required=True, help="LoGIC-Dataset repository root")
    parser.add_argument("--data", type=Path, required=True, help="Directory containing the unpacked cfd folder")
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    RESULTS=args.release.resolve()/"results"
    G,F,C=[RESULTS/name for name in ("geometry","fea","cfd")]
    CFD=args.data.resolve()/"cfd"
    OUT=args.output.resolve()
    nums=args.figures or list(range(7,12))
    makers={7:fig7,8:fig8,9:fig9,10:fig10,11:fig11}
    for num in nums:
        makers[num]()
        print(f"Exported Fig. {num}",flush=True)
