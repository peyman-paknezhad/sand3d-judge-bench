"""Step 2: build the benchmark pairs from Sand3D programs.

Each program is run, its output is split into named part groups (table_leg.001, table_leg.002 ... -> table_leg),
and one group is damaged in a known way. The pair (original, damaged) has an exact answer: the original is better.
Two controls have no right answer (identical copy; whole object turned 45 degrees): a fair judge splits them 50/50.

Writes work/img/*.png (1024 px judge inputs), results/thumbs/*.jpg and results/pairs.csv.
Usage: python build_pairs.py          (SMOKE=3 python build_pairs.py for a quick test)
"""
import itertools, os, re, zlib
import multiprocessing as mp
import numpy as np, pandas as pd, trimesh

# ---- knobs ----
N_OBJECTS = 150
MIN_PARTS, MAX_PARTS = 3, 150
MIN_PART_SHARE = 0.01            # a damaged part group must carry at least this share of the surface area
EDITS = ["remove", "float", "shrink", "grow", "rotate"]   # damage: the original should win
CONTROLS = ["same", "turn"]                               # no right answer
FLOAT_DIST = 0.15                # float: move the part this share of the object diagonal away from the centre
SHRINK, GROW, TURN_DEG = 0.4, 2.0, 45
MIN_PIX = 0.005                  # damage must change at least this share of the pixels to count as visible
MAX_TRIES = 4                    # part groups tried per damage kind before giving up on visibility
TOUCH = 0.004                    # parts closer than this share of the diagonal count as touching
DETACH_START = 0.01              # objects that already have a loose part are skipped
DATA = "data/val.pkl"
WORK = "work"
OUT = "results"
WORKERS = max(1, os.cpu_count() - 2)
BUILD_TIMEOUT = 240
SMOKE = int(os.environ.get("SMOKE", 0))   # >0: only this many objects (a test, not results)
# ---------------
Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def worker_init():
    import logging
    logging.disable(logging.CRITICAL)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.dup2(null, 2)    # Blender prints a line per exported part


def named_parts(code, entry, path):
    """Run a Sand3D program; return [(part name, Z-up mesh)] with each part in world position."""
    import sand3d as s3d
    s3d.empty_scene()
    ns = {}
    exec(compile(code, "program", "exec"), ns)
    s3d.export(ns[entry](), path)
    sc = trimesh.load(path, force="scene")
    out = []
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]
        m = sc.geometry[g]
        if not isinstance(m, trimesh.Trimesh) or not len(m.faces):
            continue
        m = m.copy(); m.apply_transform(T); m.apply_transform(Y_UP_TO_Z_UP)
        out.append((node, m))
    os.remove(path)
    return out


def group_of(name):
    """table_leg.003 -> table_leg: copies of one part form a group and are damaged together."""
    return re.sub(r"\.\d+$", "", name)


def detached(ms):
    """Share of surface area outside the largest group of touching parts (0 = everything is connected)."""
    n = len(ms)
    diag = np.linalg.norm(trimesh.util.concatenate(ms).extents)
    root = list(range(n))
    def find(i):
        while root[i] != i:
            i = root[i]
        return i
    tol = TOUCH * diag
    for i, j in itertools.combinations(range(n), 2):
        if np.any(ms[i].bounds[1] + tol < ms[j].bounds[0]) or np.any(ms[j].bounds[1] + tol < ms[i].bounds[0]):
            continue
        cm = trimesh.collision.CollisionManager()
        cm.add_object("a", ms[i])
        if cm.min_distance_single(ms[j]) < tol:
            root[find(i)] = find(j)
    area = np.array([m.area for m in ms])
    groups = {}
    for i in range(n):
        groups[find(i)] = groups.get(find(i), 0) + area[i]
    return 1 - max(groups.values()) / area.sum()


def edit(parts, kind, idx, rng):
    """Return a new part list with the parts in idx (one group) damaged, or the whole object turned (control)."""
    allm = trimesh.util.concatenate([m for _, m in parts])
    cen, diag = allm.bounds.mean(0), np.linalg.norm(allm.extents)
    out = [(n, m.copy()) for n, m in parts]
    if kind == "same":
        return out
    if kind == "turn":
        R = trimesh.transformations.rotation_matrix(np.radians(TURN_DEG), [0, 0, 1], cen)
        for _, m in out: m.apply_transform(R)
        return out
    if kind == "remove":
        return [x for j, x in enumerate(out) if j not in idx]
    gc = trimesh.util.concatenate([out[j][1] for j in idx]).bounds.mean(0)
    if kind == "float":
        d = gc - cen
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-3 * diag else np.array([0, 0, 1.0])
        T = trimesh.transformations.translation_matrix(d * FLOAT_DIST * diag)
    elif kind == "rotate":
        T = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0] if rng.random() < .5 else [0, 1, 0], gc)
    for j in idx:
        m = out[j][1]
        if kind in ("shrink", "grow"):   # each copy about its own centre: thinner legs, fatter handle
            m.apply_transform(trimesh.transformations.scale_matrix(SHRINK if kind == "shrink" else GROW, m.bounds.mean(0)))
        else:
            m.apply_transform(T)
    return out


def save(img, key):
    img.save(f"{WORK}/img/{key}.png")
    img.resize((200, 200)).save(f"{OUT}/thumbs/{key}.jpg", quality=88)


def make_object(task):
    """Build one object and write its original and edited renders. Any failure skips the object."""
    from render import grid
    oid, code, entry = task
    try:
        parts = named_parts(code, entry, f"{WORK}/{oid}.glb")
        if not (MIN_PARTS <= len(parts) <= MAX_PARTS):
            return dict(oid=oid, ok=False, error="parts out of range")
        ms = [m for _, m in parts]
        if detached(ms) >= DETACH_START:
            return dict(oid=oid, ok=False, error="detached at start")
        area = np.array([m.area for m in ms])
        groups = {}
        for j, (n, _) in enumerate(parts):
            groups.setdefault(group_of(n), []).append(j)
        share = {g: area[ix].sum() / area.sum() for g, ix in groups.items()}
        big = max(share, key=share.get)
        cand = sorted(g for g in groups if share[g] >= MIN_PART_SHARE and g != big)   # never the main body
        if not cand:
            return dict(oid=oid, ok=False, error="no editable part")
        rng = np.random.default_rng(zlib.crc32(oid.encode()))
        g0 = grid(ms); save(g0, f"{oid}__orig")
        a0 = np.asarray(g0.convert("L"), float)
        rows = []
        for kind in EDITS + CONTROLS:
            tries = [str(x) for x in rng.permutation(cand)[:MAX_TRIES]] if kind in EDITS else [""]
            for gname in tries:     # damage must be visible: take the first group whose edit changes enough pixels
                new = edit(parts, kind, groups.get(gname, []), rng)
                g = grid([m for _, m in new])
                pix = float((np.abs(np.asarray(g.convert("L"), float) - a0) > 20).mean())
                if pix >= MIN_PIX:
                    break
            save(g, f"{oid}__{kind}")
            rows.append(dict(oid=oid, kind=kind, damage=kind in EDITS, part=gname, n_copies=len(groups.get(gname, [])), part_share=share.get(gname, 0.0),
                             n_parts=len(parts), n_groups=len(groups), pix_diff=pix, visible=pix >= MIN_PIX,
                             detached_after=detached([m for _, m in new])))
        return dict(oid=oid, ok=True, rows=rows)
    except Exception as e:
        return dict(oid=oid, ok=False, error=f"{type(e).__name__}: {str(e)[:120]}")


def main():
    os.makedirs(f"{WORK}/img", exist_ok=True); os.makedirs(f"{OUT}/thumbs", exist_ok=True)
    n = SMOKE or N_OBJECTS
    df = pd.read_pickle(DATA).sample(frac=1, random_state=0)
    df = df.drop_duplicates("entry")                      # one object per name: more variety
    tried = df.head(n * 3)
    pool = mp.get_context("spawn").Pool(WORKERS, initializer=worker_init, maxtasksperchild=20)
    jobs = {r.id: pool.apply_async(make_object, ((r.id, r.program, r.entry),)) for r in tried.itertuples()}
    info = tried.set_index("id")
    rows, fails, kept = [], {}, 0
    for oid, j in jobs.items():
        try:
            x = j.get(timeout=BUILD_TIMEOUT)
        except mp.TimeoutError:
            x = dict(oid=oid, ok=False, error="timeout")
        if x["ok"] and kept < n:
            kept += 1
            for r in x["rows"]:
                rows.append(dict(r, entry=info.loc[oid, "entry"], name=info.loc[oid, "entry"].replace("_", " "), family=info.loc[oid, "family"]))
        elif not x["ok"]:
            e = x["error"].split(":")[0]; fails[e] = fails.get(e, 0) + 1
    pool.terminate()
    pd.DataFrame(rows).to_csv(f"{OUT}/pairs{'_smoke' if SMOKE else ''}.csv", index=False)
    print(f"{kept} objects, {len(rows)} pairs, from {len(tried)} tried; skipped: {fails}", flush=True)
    assert kept == n, "not enough usable objects"


if __name__ == "__main__":
    main()
