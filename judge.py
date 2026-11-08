"""Step 3: score every benchmark pair with one judge. Usage: python judge.py <clip | qwen7b | qwen32b>
  qwen7b / qwen32b   two ways each: 'score' = yes/no correctness score on each object alone,
                     'pair' = both objects shown, judge picks 1 or 2; asked in both orders and averaged
  clip               CLIP similarity between each view and "a 3D model of a <name>" (text-to-3D metric)
Writes results/judge_<judge>.csv with one row per pair: pref = how much the judge prefers the original (0..1, 0.5 = no preference)."""
import os, sys
import numpy as np, pandas as pd
from PIL import Image

# ---- knobs ----
MODELS = {"qwen7b": "Qwen/Qwen2.5-VL-7B-Instruct", "qwen32b": "Qwen/Qwen2.5-VL-32B-Instruct"}
CLIP = ("ViT-L-14", "laion2b_s32b_b82k")
PIXELS = 1024 * 1024        # per image: the full 2x2 render
BATCH = {"qwen7b": 1, "qwen32b": 1}   # batch 1 on purpose: padded batches shifted the scores of identical images by ~0.007, as big as the effects
WORK = "work"
OUT_DIR = "results"
PAIRS = os.environ.get("PAIRS", f"{OUT_DIR}/pairs.csv")
PAIR_PREFIX = "The correctly built one is Image "   # starts the answer; without it the 32B model begins "To determine..." and rarely says 1 or 2
# ---------------
SPEC = "all parts attached where they belong, no part missing or absurdly thin, no part passing through another, no holes or inside-out surfaces"
img = lambda key: Image.open(f"{WORK}/img/{key}.png").convert("RGB")


class Qwen:
    def __init__(self, name):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        assert torch.cuda.is_available(), "No GPU found."
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODELS[name], padding_side="left", min_pixels=PIXELS, max_pixels=PIXELS)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODELS[name], dtype=torch.bfloat16, device_map="cuda").eval()
        self.model.lm_head.float()   # bfloat16 logits would round different objects to the same score
        tok = self.proc.tokenizer
        self.ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", "No", "no", "1", "2")}

    def probs(self, texts, images, prefix=""):
        """texts[i] goes with the list images[i]; prefix starts the answer for the model. Returns next-token probabilities."""
        chats = [[{"role": "user", "content": [{"type": "image"}] * len(ims) + [{"type": "text", "text": t}]}] for t, ims in zip(texts, images)]
        chat = [self.proc.apply_chat_template(c, tokenize=False, add_generation_prompt=True) + prefix for c in chats]
        inp = self.proc(text=chat, images=[i for ims in images for i in ims], return_tensors="pt", padding=True).to("cuda")
        with self.torch.no_grad():
            h = self.model.model(**inp).last_hidden_state[:, -1]
            return self.model.lm_head(h.float()).softmax(-1).cpu().numpy()

    def p(self, pr, words):
        return pr[:, [self.ids[w] for w in words]].sum(1)

    def score(self, items, bs):
        """items: (name, key). Score = mean of P(Yes | correctly built?) and P(No | has a defect?)."""
        out = []
        for b in range(0, len(items), bs):
            batch = items[b:b + bs]
            ims = [[img(k)] for _, k in batch]
            pre = [f"Four views of a 3D model that should be a {n}. " for n, _ in batch]
            pos = self.probs([x + f"Is it correctly built ({SPEC})? Answer Yes or No." for x in pre], ims)
            neg = self.probs([x + f"Does it have a construction defect (the opposite of: {SPEC})? Answer Yes or No." for x in pre], ims)
            y, n = self.p(pos, ["Yes", "yes"]), self.p(pos, ["No", "no"])
            y2, n2 = self.p(neg, ["Yes", "yes"]), self.p(neg, ["No", "no"])
            out += [dict(score=(a / (a + c) + d / (e + d)) / 2, mass=min(a + c, e + d)) for a, c, e, d in zip(y, n, y2, n2)]
            print(f"score {b + len(batch)}/{len(items)}", flush=True)
        return out

    def pair(self, items, bs):
        """items: (name, key_a, key_b). Returns P(judge picks a), asked with a first and with a second, averaged."""
        q = lambda n: (f"Image 1 and Image 2 each show four views of a 3D model that should be a {n}. "
                       f"Which one is built correctly ({SPEC})? Answer 1 or 2.")
        out = []
        for b in range(0, len(items), bs):
            batch = items[b:b + bs]
            texts = [q(n) for n, _, _ in batch]
            ab = self.probs(texts, [[img(a), img(c)] for _, a, c in batch], PAIR_PREFIX)
            ba = self.probs(texts, [[img(c), img(a)] for _, a, c in batch], PAIR_PREFIX)
            p1, p2 = self.p(ab, ["1"]), self.p(ab, ["2"])
            q1, q2 = self.p(ba, ["1"]), self.p(ba, ["2"])
            out += [dict(pref=(x / (x + y) + w / (z + w)) / 2, first=x / (x + y), second=w / (z + w), mass=min(x + y, z + w)) for x, y, z, w in zip(p1, p2, q1, q2)]
            print(f"pair {b + len(batch)}/{len(items)}", flush=True)
        return out


def clip_scores(items):
    import torch, open_clip
    model, _, prep = open_clip.create_model_and_transforms(CLIP[0], pretrained=CLIP[1], device="cuda")
    tok = open_clip.get_tokenizer(CLIP[0])
    out = []
    with torch.no_grad():
        for n, k in items:
            g = img(k); s = g.size[0] // 2
            tiles = [g.crop((x, y, x + s, y + s)) for y in (0, s) for x in (0, s)]
            fi = model.encode_image(torch.stack([prep(t) for t in tiles]).cuda())
            ft = model.encode_text(tok([f"a 3D model of a {n}"]).cuda())
            fi, ft = fi / fi.norm(dim=-1, keepdim=True), ft / ft.norm(dim=-1, keepdim=True)
            out.append(dict(score=float((fi @ ft.T).mean())))
    return out


def main(judge):
    d = pd.read_csv(PAIRS)
    d["key"] = d.oid + "__" + d.kind
    objs = d.drop_duplicates("oid")
    singles = [(r.name, f"{r.oid}__orig") for r in objs.itertuples()] + [(r.name, r.key) for r in d.itertuples()]
    rows = d[["oid", "kind", "key"]].copy()
    if judge == "clip":
        s = dict(zip([k for _, k in singles], clip_scores(singles)))
        rows["s_orig"] = [s[f"{o}__orig"]["score"] for o in d.oid]
        rows["s_edit"] = [s[k]["score"] for k in d.key]
        rows["pref"] = np.where(rows.s_orig > rows.s_edit, 1.0, np.where(rows.s_orig < rows.s_edit, 0.0, 0.5))
        rows["judge"] = "clip"
        out = rows
    else:
        q = Qwen(judge)
        s = dict(zip([k for _, k in singles], q.score(singles, BATCH[judge])))
        a = rows.copy()
        a["s_orig"] = [s[f"{o}__orig"]["score"] for o in d.oid]
        a["s_edit"] = [s[k]["score"] for k in d.key]
        a["mass"] = [min(s[f"{o}__orig"]["mass"], s[k]["mass"]) for o, k in zip(d.oid, d.key)]
        a["pref"] = np.where(a.s_orig > a.s_edit, 1.0, np.where(a.s_orig < a.s_edit, 0.0, 0.5))
        a["judge"] = f"{judge}_score"
        p = pd.DataFrame(q.pair([(r.name, f"{r.oid}__orig", r.key) for r in d.itertuples()], BATCH[judge]))
        b = pd.concat([rows.reset_index(drop=True), p], axis=1)
        b["judge"] = f"{judge}_pair"
        out = pd.concat([a, b])
    out.to_csv(f"{OUT_DIR}/judge_{judge}{'_smoke' if 'smoke' in PAIRS else ''}.csv", index=False)
    print(f"JUDGE_DONE {judge} {len(out)} rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
