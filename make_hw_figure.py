import glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
def load(p): return np.array([float(l) for l in open(p) if l.strip() and not l.startswith('#')])
base="tf/traces_final/"
pairs=[("bsort",base+"bsort_cache_on.csv",base+"bsort_cache_off.csv"),
       ("insertsort",base+"insertsort_cache_on.csv",base+"insertsort_cache_off.csv"),
       ("ludcmp",base+"ludcmp_cache_on.csv",base+"ludcmp_cache_off.csv")]
fig,axes=plt.subplots(3,2,figsize=(9,9))
for r,(name,pon,poff) in enumerate(pairs):
    on,off=load(pon),load(poff)
    ax=axes[r,0]
    for v,lab,c in ((on,"L1 I+D cache enabled","tab:blue"),(off,"caches disabled","tab:red")):
        ax.hist(v,bins=120,density=True,alpha=0.55,color=c,label=lab)
    ax.set_yscale("log"); ax.set_xlabel("execution time (cycles)")
    ax.set_ylabel("density"); ax.set_title(f"{name}: measured distribution",fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax=axes[r,1]
    for v,lab,c in ((on,"cache on","tab:blue"),(off,"cache off","tab:red")):
        s=np.sort(v); ex=1.0-np.arange(1,len(s)+1)/len(s); m=ex>0
        ax.plot(s[m]/np.median(s),ex[m],color=c,lw=1.4,label=lab)
    ax.set_yscale("log"); ax.set_xlabel("execution time / median")
    ax.set_ylabel(r"$P(T>t)$"); ax.set_title(f"{name}: upper tail",fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout()
for e in ("pdf","png"): fig.savefig(f"figures/hw_benchmarks.{e}",dpi=200,bbox_inches="tight")
print("wrote 3-kernel figure")
