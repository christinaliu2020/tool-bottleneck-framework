#!/usr/bin/env python3
# Minimal VLM tool selector: per-image selection -> CSV + JSONL
# Usage example:
#   python -m tbm.vlm_select \
#     --base http://localhost:8000/v1 \
#     --model google/medgemma-4b-it \
#     --api_key EMPTY \
#     --task "Detect tumor vs normal" \
#     --toolbox_json toolboxes/histo_tools.json \
#     --dataset_dir /path/to/images --exts jpg,png \
#     --out_csv data/vlm_selected.csv \
#     --out_jsonl data/vlm_selected.jsonl \
#     --top_k 3

from __future__ import annotations
import argparse, json, csv, os, re, io, datetime
from pathlib import Path
from typing import Dict, List, Optional
from mimetypes import guess_type

try:
    from PIL import Image
except Exception:
    Image = None

from openai import OpenAI

# -------------------------
# I/O helpers
# -------------------------
def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load_toolbox(json_path: Path) -> Dict[str, str]:
    """
    Toolbox file format (simple):
      {
        "histo_nuc_type": "Return nucleus type label",
        "histo_nuc_bbox": "Return nucleus bounding boxes [x1,y1,w,h]",
        ...
      }
    Keys are tool IDs; values are one-line purposes.
    """
    obj = json.loads(Path(json_path).read_text())
    if not isinstance(obj, dict) or not all(isinstance(v, str) for v in obj.values()):
        raise SystemExit("toolbox_json must be a dict: {tool_id: purpose_str, ...}")
    return obj

def scan_images(root: Path, exts: List[str], recursive: bool) -> List[Path]:
    exts = {e.lower().lstrip(".") for e in exts if e}
    if not exts:
        exts = {"jpg","jpeg","png","webp","tif","tiff"}
    files = (list(root.rglob("*")) if recursive else list(root.glob("*")))
    files = [p for p in files if p.is_file() and p.suffix.lower().lstrip(".") in exts]
    files.sort()
    return files

# -------------------------
# Image encoding (base64 data URLs)
# -------------------------
def infer_image_format(path: str, fallback: str="jpeg") -> str:
    mime, _ = guess_type(path)
    if mime and mime.startswith("image/"):
        ext = mime.split("/")[-1].lower()
        return "tiff" if ext in {"tif","tiff"} else ext
    ext = os.path.splitext(path)[-1].lower().strip(".")
    return "tiff" if ext in {"tif","tiff"} else (ext or fallback)

def encode_image_to_base64(path: str, resize_max: Optional[int]=None, jpeg_quality: int=85) -> str:
    fmt = infer_image_format(path)
    # For TIFFs or resizing, prefer PIL; else raw bytes are fine.
    use_pil = fmt in {"tif","tiff"} or (resize_max is not None)
    if use_pil:
        if Image is None:
            raise RuntimeError("PIL not available; required for TIFF or resizing.")
        img = Image.open(path)
        try:
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
        except Exception:
            pass
        if img.mode not in ("RGB","L"):
            img = img.convert("RGB")
        if resize_max:
            w,h = img.size
            s = resize_max / max(w,h)
            if s < 1.0:
                img = img.resize((max(1,int(w*s)), max(1,int(h*s))), Image.LANCZOS)
        buf = io.BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = buf.getvalue()
        mime = "image/jpeg"
    else:
        with open(path, "rb") as f:
            b64 = f.read()
        mime = f"image/{fmt}"
    import base64
    return f"data:{mime};base64,{base64.b64encode(b64).decode('utf-8')}"

# -------------------------
# OpenAI/vLLM client + call
# -------------------------
def get_client(base: str, api_key: str, timeout: float=60.0, max_retries: int=2) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base, timeout=timeout, max_retries=max_retries)

def selector_system(max_k: int) -> str:
    return (
        "You are a medical imaging expert. From a provided toolbox, select up to "
        f"{max_k} tools that best support the specified task on the given image.\n"
        "Return ONLY this JSON object:\n"
        "{ \"task_modality\": \"histopathology|dermatology|xray|unknown\","
        "  \"task\": string,"
        "  \"selected_tools\": [ {\"id\": string, \"rank\": int, \"confidence\": number, \"reason\": string} ],"
        "  \"abstain\": boolean }\n"
        "Rules: unique tool ids; ranks are 1..N contiguous; reason ≤ 12 words. No markdown."
    )

def build_messages(task: str, image_data_url: Optional[str], toolbox: Dict[str,str], top_k: int) -> List[dict]:
    # Single user message keeps alternation simple for vLLM
    catalog_lines = [f"- {tid}: {purpose}" if purpose else f"- {tid}"
                     for tid, purpose in toolbox.items()]
    content = [
        {"type":"text","text":f"Task: {task}"},
        {"type":"text","text":"Tool catalog (ID → purpose):"},
        *[{"type":"text","text":line} for line in catalog_lines],
        {"type":"text","text":"Choose a subset (max {k}) from: ".format(k=top_k) + ", ".join(toolbox.keys())},
        {"type":"text","text":"Return JSON ONLY with keys: task_modality, task, selected_tools, abstain."},
        {"type":"text","text":"Keep each reason ≤ 12 words; short noun phrases."},
    ]
    if image_data_url:
        content.insert(1, {"type":"text","text":"Image:"})
        content.insert(2, {"type":"image_url","image_url":{"url": image_data_url, "detail":"high"}})
    return [
        {"role":"system","content":[{"type":"text","text":selector_system(top_k)}]},
        {"role":"user","content":content},
    ]

# -------------------------
# Minimal JSON validation
# -------------------------
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE | re.DOTALL)
def _strip_fences(s: str) -> str:
    return FENCE_RE.sub("", s).strip()

def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception:
        s = _strip_fences(content)
        # extract first {...}
        start = s.find("{")
        if start == -1: raise
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{": depth += 1
            if c == "}":
                depth -= 1
                if depth == 0:
                    blob = s[start:i+1]
                    break
        return json.loads(blob)

def validate_selection(obj: dict, allowed_ids: List[str], top_k: int) -> List[str]:
    # Coerce missing top-level keys if model omitted
    obj.setdefault("task_modality", "unknown")
    obj.setdefault("task", "")
    obj.setdefault("selected_tools", [])
    obj.setdefault("abstain", not bool(obj.get("selected_tools")))
    # Basic checks
    if obj.get("abstain") is True:
        return []
    sel = obj.get("selected_tools", [])
    if not isinstance(sel, list):
        return []
    # Normalize + filter
    out_ids = []
    seen = set()
    for e in sel[:top_k]:
        if not isinstance(e, dict): continue
        tid = str(e.get("id","")).strip()
        if not tid or tid in seen: continue
        if tid not in allowed_ids: continue
        seen.add(tid); out_ids.append(tid)
    return out_ids

def main():
    ap = argparse.ArgumentParser(description="Minimal VLM tool selector → CSV/JSONL")
    ap.add_argument("--base", required=True, help="OpenAI-compatible base URL (vLLM, etc.)")
    ap.add_argument("--model", required=True, help="Model name (e.g., google/medgemma-4b-it)")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--task", required=True, help="Task text, e.g., 'Classify tumor vs normal'")
    ap.add_argument("--toolbox_json", type=Path, required=True, help="JSON: {tool_id: purpose, ...}")
    # images: either --dataset_dir or explicit --images
    ap.add_argument("--dataset_dir", type=Path)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--exts", default="jpg,jpeg,png,webp,tif,tiff")
    ap.add_argument("--images", nargs="*", help="Optional explicit list of image paths")
    # output
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--out_jsonl", type=Path, required=True)
    # generation knobs
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=640)
    # image knobs
    ap.add_argument("--resize_max", type=int, default=None)
    ap.add_argument("--jpeg_quality", type=int, default=85)
    args = ap.parse_args()

    toolbox = load_toolbox(args.toolbox_json)
    allowed_ids = list(toolbox.keys())

    # Gather image list
    imgs: List[Path] = []
    if args.images:
        imgs = [Path(p) for p in args.images]
    elif args.dataset_dir:
        imgs = scan_images(args.dataset_dir, [e.strip() for e in args.exts.split(",")], recursive=args.recursive)
    if not imgs:
        raise SystemExit("No images provided. Use --images ... or --dataset_dir ...")

    # Output files
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(args.out_csv, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["image_path","tool_ids"])

    jsonl_f = open(args.out_jsonl, "w", encoding="utf-8")

    client = get_client(args.base, args.api_key)

    for i, img_path in enumerate(imgs, 1):
        img_str = str(img_path)
        try:
            data_url = encode_image_to_base64(img_str, resize_max=args.resize_max, jpeg_quality=args.jpeg_quality)
            messages = build_messages(args.task, data_url, toolbox, args.top_k)
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                response_format={"type":"json_object"},
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            content = resp.choices[0].message.content
            obj = _parse_json(content)
            tool_ids = validate_selection(obj, allowed_ids, args.top_k)

            # CSV row
            csv_w.writerow([img_str, ",".join(tool_ids)])

            # JSONL audit
            record = {
                "timestamp": now_iso(),
                "image_path": img_str,
                "task": args.task,
                "top_k": args.top_k,
                "selected_tool_ids": tool_ids,
                "raw_selection": obj,  # the parsed JSON from the model
                "model": args.model,
                "base_url": args.base,
            }
            jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[{i}/{len(imgs)}] OK: {img_path.name} -> {tool_ids}")

        except Exception as e:
            # On any failure, write empty selection (robust for training)
            csv_w.writerow([img_str, ""])
            err = {
                "timestamp": now_iso(),
                "image_path": img_str,
                "task": args.task,
                "error": str(e),
                "model": args.model,
                "base_url": args.base,
            }
            jsonl_f.write(json.dumps(err, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(imgs)}] ERROR: {img_path.name} -> {e}")

    csv_f.close(); jsonl_f.close()
    print(f"\nSaved CSV:   {args.out_csv}")
    print(f"Saved JSONL: {args.out_jsonl}")

if __name__ == "__main__":
    main()
