"""Step 4: numbers and figures from results/ (pairs.csv, judge_*.csv, thumbs/). Writes figures/ and results/summary.txt. Runs on a laptop."""
import glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ---- knobs ----
RES = "results"
FIG = "figures"
DETACH = 0.01                 # the geometry rule flags an edit when this share of the surface comes loose
N_BOOT = 2000
N_GALLERY = 8
EXAMPLE = None                # object for the "what the damage looks like" figure; None = the one whose edits are all most visible
GALLERY_KINDS = ["remove", "float", "rotate", "shrink"]   # grown parts can look like a plausible design, so the gallery leaves them out
EDITS = ["remove", "float", "shrink", "grow", "rotate"]
EDIT_LABEL = {"remove": "part\nremoved", "float": "part\nfloating", "shrink": "part\nshrunk 0.4×", "grow": "part\ngrown 2×", "rotate": "part\nturned 90°"}
JUDGES = {"rule": "Geometry rule (loose part)", "clip": "CLIP text match", "qwen7b_score": "Qwen2.5-VL-7B, yes/no", "qwen7b_pair": "Qwen2.5-VL-7B, A vs B",
          "qwen32b_score": "Qwen2.5-VL-32B, yes/no", "qwen32b_pair": "Qwen2.5-VL-32B, A vs B", "combo": "Rule first, then 32B A vs B"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
# ---------------
os.makedirs(FIG, exist_ok=True)
if not os.path.isdir(f"{RES}/thumbs"):          # the renders are shipped as one zip to keep the file count small
    import zipfile; zipfile.ZipFile(f"{RES}/thumbs.zip").extractall(RES)
pairs = pd.read_csv(f"{RES}/pairs.csv"); pairs["key"] = pairs.oid + "__" + pairs.kind
j = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/judge_*.csv")) if "smoke" not in f])
rule = pairs[["oid", "kind", "key"]].copy()
rule["pref"] = np.where(pairs.detached_after >= DETACH, 1.0, 0.5)   # a rule has no opinion when nothing comes loose
rule["judge"] = "rule"
best = j[j.judge == "qwen32b_pair"][["key", "pref"]].set_index("key").pref
combo = rule.copy(); combo["judge"] = "combo"
combo["pref"] = np.where(pairs.detached_after.values >= DETACH, 1.0, combo.key.map(best).values)   # rule vetoes loose parts, the VLM decides the rest
j = pd.concat([j, rule] + ([combo] if len(best) else [])).merge(pairs.drop(columns=["oid", "kind"]), on="key")
j["right"] = np.where(j.pref > .5, 1.0, np.where(j.pref < .5, 0.0, 0.5))
judges = [k for k in JUDGES if k in set(j.judge)]
lines = []
def say(x=""):
    print(x); lines.append(str(x))

# sanity: fail loud on judges that do not answer or do not vary
say(f"{pairs.oid.nunique()} objects, {len(pairs)} pairs; damage visible: {pairs[pairs.damage].visible.mean():.0%} (invisible ones dropped)")
for k in judges:
    x = j[j.judge == k]
    m = f" answer mass min {x['mass'].min():.3f}" if "mass" in x and x["mass"].notna().any() else ""
    say(f"  {k:15s} pref std {x.pref.std():.3f}, ties {(x.pref == .5).mean():.0%}{m}")
    assert x.pref.std() > 1e-3 or k == "rule", f"{k} does not vary"
    if "mass" in x and x["mass"].notna().any():
        assert x["mass"].median() > .5, f"{k}: the judge mostly answers with other words (median answer mass {x['mass'].median():.2f})"
dmg = j[j.damage & j.visible]

def boot(x, by="oid"):
    """Accuracy and 95% CI, resampling objects."""
    g = x.groupby(by).right.agg(["sum", "count"])
    r = np.random.default_rng(0); s, c = g["sum"].values, g["count"].values
    b = [s[i].sum() / c[i].sum() for i in (r.integers(len(g), size=len(g)) for _ in range(N_BOOT))]
    return s.sum() / c.sum(), *np.percentile(b, [2.5, 97.5])

say("\nPicks the undamaged original (accuracy, 95% CI over objects; 50% = chance)")
tab = {}
for k in judges:
    a, lo, hi = boot(dmg[dmg.judge == k]); tab[k] = {e: dmg[(dmg.judge == k) & (dmg.kind == e)].right.mean() for e in EDITS}
    tab[k]["all"] = a
    tab[k]["turn"] = j[(j.judge == k) & (j.kind == "turn")].right.mean()
    same = j[(j.judge == k) & (j.kind == "same")]
    bias = f" | identical pair: picks 'first' {same['first'].mean():.0%}" if "first" in same and same["first"].notna().any() else ""
    turn = j[(j.judge == k) & (j.kind == "turn")].right.mean()
    say(f"{JUDGES[k]:28s} {a:.0%} [{lo:.0%}, {hi:.0%}]  " + "  ".join(f"{e} {tab[k][e]:.0%}" for e in EDITS) + f" | harmless turn: prefers original {turn:.0%}{bias}")
vlm = [k for k in judges if k.startswith("qwen")]   # the VLM judges alone, for the gallery
if vlm:
    w = dmg[dmg.judge.isin(vlm)].groupby("key").right.mean()
    say(f"\nDamaged objects that every VLM judge prefers to the original: {(w == 0).sum()} of {len(w)} ({(w == 0).mean():.0%})")
    say(f"Damage the geometry rule catches: {(pairs[pairs.damage & pairs.visible].detached_after >= DETACH).mean():.0%}")
open(f"{RES}/summary.txt", "w").write("\n".join(lines) + "\n")

# figure 1: report card, judges x damage type
cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f2f1ed", "#2a78d6"])
cols = EDITS + ["all", "turn"]
M = np.array([[tab[k][e] for e in cols] for k in judges])
Mc = M.copy(); Mc[:, -1] = np.nan        # the harmless control is not coloured as right/wrong
fig, ax = plt.subplots(figsize=(8.2, .55 * len(judges) + 1.5), dpi=170)
ax.imshow(np.ma.masked_invalid(Mc), cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for c in range(M.shape[1]):
        ax.text(c, i, f"{M[i, c]:.0%}", ha="center", va="center", fontsize=9, color=INK, fontweight="bold" if cols[c] == "all" else None)
ax.set_xticks(range(len(cols)), [EDIT_LABEL.get(e, {"all": "all\ndamage", "turn": "harmless turn\n(fair = 50%)"}.get(e)) for e in cols], fontsize=8.5, color=MUTED)
ax.set_yticks(range(len(judges)), [JUDGES[k] for k in judges], fontsize=9, color=INK)
ax.axvline(len(cols) - 2.5, color="white", lw=4); ax.axvline(len(cols) - 1.5, color="white", lw=4)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("How often each judge picks the undamaged original over the damaged one\n50% = coin flip, red = prefers the broken object", fontsize=9.5, color=INK, loc="left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_report_card.png"); plt.close(fig)

# figure 2: accuracy against the size of the damaged part
bins = [0, .02, .05, .1, .2, 1]
dmg = dmg.assign(sz=pd.cut(dmg.part_share, bins, labels=["1-2%", "2-5%", "5-10%", "10-20%", ">20%"]))
COL = {"rule": "#8a8984", "clip": "#1baf7a", "qwen7b_score": "#f2a37f", "qwen7b_pair": "#eb6834", "qwen32b_score": "#8fb8ec", "qwen32b_pair": "#2a78d6", "combo": "#0b0b0b"}
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=170)
for k in judges:
    m = dmg[dmg.judge == k].groupby("sz", observed=False).right.mean()
    ax.plot(range(len(m)), m.values * 100, color=COL[k], lw=2, marker="o", ms=5, label=JUDGES[k])
ax.axhline(50, color=MUTED, lw=1, ls=(0, (3, 3))); ax.text(-.1, 51, "coin flip", fontsize=8, color=MUTED)
ax.set_xticks(range(len(bins) - 1), m.index.astype(str))
ax.set_xlabel("Size of the damaged part (share of the surface)", fontsize=9.5, color=MUTED)
ax.set_ylabel("Picks the original (%)", fontsize=9.5, color=MUTED); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, length=0); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_part_size.png"); plt.close(fig)

# figure 3: the misses - clearly visible damage that every VLM judge prefers to the original
if vlm:
    v = dmg[dmg.judge.isin(vlm)]
    w = v.groupby("key").agg(right=("right", "mean"), pref=("pref", "mean")).join(pairs.set_index("key"))
    x = w[(w.right == 0) & w.kind.isin(GALLERY_KINDS)].sort_values("pix_diff", ascending=False).drop_duplicates("oid").head(N_GALLERY)
    if len(x):
        fig, axs = plt.subplots(2, len(x), figsize=(2.1 * len(x), 4.9), dpi=170, squeeze=False)
        for i, (key, r) in enumerate(x.iterrows()):
            for row, k in ((0, f"{r.oid}__orig"), (1, key)):
                ax = axs[row, i]
                p = f"{RES}/thumbs/{k}.jpg"
                assert os.path.exists(p), f"missing thumbnail {p}"
                ax.imshow(Image.open(p))
                ax.set_xticks([]); ax.set_yticks([]); [s.set_color(GRID) for s in ax.spines.values()]
            axs[0, i].set_title(r["name"], fontsize=8.5, color=INK)
            axs[1, i].set_xlabel(f"{r.part.replace('_', ' ')}: {EDIT_LABEL[r.kind].replace(chr(10), ' ')}", fontsize=7.2, color=INK)
        axs[0, 0].set_ylabel("Original", fontsize=9, color=MUTED); axs[1, 0].set_ylabel("Damaged", fontsize=9, color=MUTED)
        fig.suptitle(f"Every VLM judge ({len(vlm)} judge settings) prefers the damaged object (bottom) to the original (top)", fontsize=10, color=INK, x=.01, ha="left")
        fig.tight_layout(); fig.savefig(f"{FIG}/fig3_misses.png"); plt.close(fig)
# figure 0: what each edit looks like, on one object
e = pairs[pairs.kind.isin(EDITS)].groupby("oid").pix_diff.min()
ex = EXAMPLE or e.idxmax()
r = pairs[pairs.oid == ex].set_index("kind")
cols0 = [("orig", "original")] + [(k, f"{r.loc[k, 'part'].replace('_', ' ')}\n{EDIT_LABEL[k].replace(chr(10), ' ')}") for k in EDITS] + [("turn", "control:\nwhole object turned 45°")]
fig, axs = plt.subplots(1, len(cols0), figsize=(1.9 * len(cols0), 2.9), dpi=170)
for ax, (k, lab) in zip(axs, cols0):
    ax.imshow(Image.open(f"{RES}/thumbs/{ex}__{k}.jpg")); ax.set_xticks([]); ax.set_yticks([])
    [sp.set_color(GRID) for sp in ax.spines.values()]
    ax.set_xlabel(lab, fontsize=7.5, color=INK)
fig.suptitle(f"One benchmark object ({r['name'].iloc[0]}) and its edits; each image is the 4-view render a judge sees", fontsize=9.5, color=INK, x=.01, ha="left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig0_edits.png"); plt.close(fig)
print("figures written to", FIG)
