"""Small CPU z-buffer renderer (no OpenGL needed): four grey clay views of a mesh in one 2x2 image."""
import numpy as np, trimesh
from PIL import Image

TILE = 512                   # pixels per view; the 2x2 grid is 1024 x 1024
AZ = (45, 135, 225, 315)     # four cameras around the object
EL = 25                      # camera elevation in degrees
LIGHT = np.array([.4, -.5, .77])


def camera(az, el):
    a, e = np.radians(az), np.radians(el)
    eye = 2.6 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = -eye / np.linalg.norm(eye)
    s = np.cross(f, (0, 0, 1)); s /= np.linalg.norm(s)
    return eye, np.stack([s, np.cross(s, f), f])


def project(v, eye, R, size=TILE):
    c = (v - eye) @ R.T
    k = 4.4 * size / 2       # focal length in pixels; the object fills about 85% of the view
    return np.stack([size / 2 + k * c[:, 0] / c[:, 2], size / 2 - k * c[:, 1] / c[:, 2], c[:, 2]], 1)


def raster(tri, size=TILE):
    """tri: (F,3,3) screen-space triangles. Returns the face id seen at each pixel (-1 = background)."""
    ids, z = np.full((size, size), -1, int), np.full((size, size), np.inf)
    for i, t in enumerate(tri):
        x0, x1 = int(max(0, np.floor(t[:, 0].min()))), int(min(size - 1, np.ceil(t[:, 0].max())))
        y0, y1 = int(max(0, np.floor(t[:, 1].min()))), int(min(size - 1, np.ceil(t[:, 1].max())))
        if x0 > x1 or y0 > y1:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-9:
            continue
        w0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / d
        w1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / d
        w2 = 1 - w0 - w1
        ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        dep = w0 * az + w1 * bz + w2 * cz
        sub = z[y0:y1 + 1, x0:x1 + 1]
        upd = ins & (dep < sub)
        sub[upd] = dep[upd]
        ids[y0:y1 + 1, x0:x1 + 1][upd] = i
    return ids


def clay(ids, normals):
    img = np.ones((*ids.shape, 3))
    hit = ids >= 0
    img[hit] = (.25 + .7 * np.abs(normals[ids[hit]] @ LIGHT))[:, None] * np.array([.82, .80, .76])
    return img


def grid(meshes):
    """Four views of a list of meshes, centred and scaled to the same size, as one RGB image."""
    m = trimesh.util.concatenate(meshes)
    cen, scale = m.bounds.mean(0), 1 / np.linalg.norm(m.extents)
    V, F, N = (m.vertices - cen) * scale, m.faces, m.face_normals
    out = Image.new("RGB", (TILE * 2, TILE * 2), "white")
    for i, az in enumerate(AZ):
        eye, R = camera(az, EL)
        ids = raster(project(V, eye, R)[F])
        t = Image.fromarray((np.clip(clay(ids, N), 0, 1) * 255).astype(np.uint8))
        out.paste(t, ((i % 2) * TILE, (i // 2) * TILE))
    return out
