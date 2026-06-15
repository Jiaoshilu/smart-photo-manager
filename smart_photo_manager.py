#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Photo Manager (Bilingual + AI, Library Grid, Editor++, Suggestions)
-------------------------------------------------------------------------
- GUI: Tkinter
- DB: SQLite (auto-migration)
- Image: Pillow + OpenCV
- Charts: Matplotlib
- AI: Google Cloud Vision API (GOOGLE_API_KEY via env/keyring/local file)
- Library: scrollable grid of thumbnails with filter by FINAL AI label (ai_top_label)
- Editor++: EV/gamma/contrast/saturation/temperature/tint/clarity/denoise/grain
            + highlights/shadows/vibrance/vignette/sharpen
            + rotate/flip/auto white balance/center-crop (1:1, 16:9)
            + Before/After toggle, ISO presets
- Suggestions: exposure/sharpness + scene-aware hints
- Batch import: import folder, call AI, persist to DB, final label list auto-built
- i18n: zh/en; all UI strings go through t()

Author: ChatGPT
License: MIT
"""

from __future__ import annotations

# stdlib
import os, json, csv, math, time, base64, sqlite3, threading, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# tk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog

# third party
from PIL import Image, ImageTk, ExifTags
import numpy as np
import cv2
import requests
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# --- MUSIQ (TF Hub) ---
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    # 避免和 Tk GPU/驱动冲突：不用 GPU
    try: tf.config.set_visible_devices([], "GPU")
    except Exception: pass
    _TF_AVAILABLE = True
except Exception:
    tf = hub = None
    _TF_AVAILABLE = False

MUSIQ_URL = "https://tfhub.dev/google/musiq/ava/1"  # 偏审美
_MUSIQ_MODEL = None
_MUSIQ_LOCK = threading.Lock()

def _load_musiq_model_blocking():
    """阻塞加载（会命中缓存），只在后台线程调用。"""
    global _MUSIQ_MODEL
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow / TF-Hub 未安装")
    with _MUSIQ_LOCK:
        if _MUSIQ_MODEL is None:
            _MUSIQ_MODEL = hub.load(MUSIQ_URL)
    return _MUSIQ_MODEL

def musiq_score_ndarray_rgb(rgb: np.ndarray) -> tuple[Optional[float], Optional[float], str]:
    if not _TF_AVAILABLE:
        return None, None, "TF/Hub not available"
    try:
        model = _load_musiq_model_blocking()

        # 统一尺寸到短边 512（经验上 MUSIQ 更稳）
        h, w = rgb.shape[:2]
        short = min(h, w)
        if short > 512:
            s = 512.0 / short
            rgb = cv2.resize(rgb, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

        # ==== 路线 A：serving_default(image_bytes=...) ====
        sig = getattr(model, "signatures", {}).get("serving_default") if hasattr(model, "signatures") else None
        if sig is not None:
            in_keys = list(getattr(sig, "structured_input_signature", (None, {}))[1].keys())
            if "image_bytes" in in_keys or "bytes" in in_keys:
                try:
                    rgb8 = tf.convert_to_tensor(rgb, dtype=tf.uint8)
                    jpeg = tf.io.encode_jpeg(rgb8)          # -> scalar string
                    b = tf.expand_dims(jpeg, 0)             # [1]
                    kwargs = {"image_bytes": b} if "image_bytes" in in_keys else {"bytes": b}
                    out = sig(**kwargs)
                    for k in ("mean","score","quality_score","scores","output_0","default"):
                        if k in out:
                            v = float(out[k].numpy().reshape(-1)[0])
                            if v > 10: v /= 10.0
                            if 0 <= v <= 1: v *= 10.0
                            s = None
                            for ks in ("std","stddev"):
                                if ks in out:
                                    s = float(out[ks].numpy().reshape(-1)[0])
                                    if s > 10: s /= 10.0
                                    if 0 <= s <= 1: s *= 10.0
                                    break
                            return v, s, ""
                except Exception:
                    pass

        # ==== 路线 B：直接可调用模块（吃 uint8/float32 张量） ====
        x8  = tf.convert_to_tensor(rgb, dtype=tf.uint8)[None, ...]
        x32 = tf.cast(x8, tf.float32) / 255.0
        for xin in (x8, x32):
            try:
                y = model(xin)
                if isinstance(y, dict) and y:
                    any_t = list(y.values())[0]
                    v = float(any_t.numpy().reshape(-1)[0])
                else:
                    v = float(y.numpy().reshape(-1)[0])
                if v > 10: v /= 10.0
                if 0 <= v <= 1: v *= 10.0
                return v, None, ""
            except Exception:
                pass

        # ==== 路线 C：KerasLayer（有些 hub 模块必须用这个） ====
        try:
            kl = hub.KerasLayer(MUSIQ_URL, trainable=False)
            for xin in (x32, x8):
                try:
                    y = kl(xin)
                    v = float(y.numpy().reshape(-1)[0])
                    if v > 10: v /= 10.0
                    if 0 <= v <= 1: v *= 10.0
                    return v, None, ""
                except Exception:
                    pass
        except Exception:
            pass

        return None, None, "Unsupported signature"
    except Exception as e:
        return None, None, str(e)


def _debug_log_musiq_signatures(model, ui_log=print):
    try:
        ui_log("[MUSIQ] type=" + str(type(model)))
        if hasattr(model, "signatures"):
            keys = list(model.signatures.keys())
            ui_log("[MUSIQ] signatures: " + ", ".join(keys) if keys else "[MUSIQ] signatures: (none)")
            for name, fn in model.signatures.items():
                try:
                    ins = list(fn.structured_input_signature[1].keys())
                except Exception:
                    ins = []
                try:
                    outs = list(fn.structured_outputs.keys())
                except Exception:
                    outs = []
                ui_log(f"[MUSIQ] signature {name}: inputs={ins or '-'}  outputs={outs or '-'}")
        else:
            ui_log("[MUSIQ] model has no .signatures")
    except Exception as e:
        ui_log(f"[MUSIQ] signature introspection error: {e}")



# keyring optional
try:
    import keyring
except Exception:
    keyring = None

# ---------------- i18n ----------------

LANGS = {
    "zh": {
        "title": "智能照片管理器 (中英双语+AI)",
        "tab_log": "日志",
        "tab_lib": "库",
        "tab_meta": "EXIF 与特征",
        "tab_sug": "建议",
        "btn_open_image": "打开图片",
        "btn_open_folder": "导入文件夹",
        "btn_charts": "图表",
        "btn_search": "搜索",
        "btn_edit": "编辑",
        "dlg_select_image": "选择图片",
        "dlg_select_folder": "选择文件夹",
        "notice_title": "提示",
        "notice_no_images": "没有可用的图片",
        "completed_title": "完成",
        "completed_msg": "处理完成！结果已保存到：\n{path}",
        "exif_header": "-- EXIF 信息 --",
        "exif_none": "没有 EXIF 数据",
        "sharp_header": "-- 清晰度 --",
        "progress_fmt": "进度: {i}/{total}",
        "charts_title": "图表（直方图/色彩/预览）",
        "charts_count_ylabel": "数量",
        "library": "库",
        "filter": "筛选",
        "filter_all": "全部",
        "import_folder": "导入文件夹",
        "refresh": "刷新",
        "edit_dehaze": "去雾",
        "edit_rotate": "旋转（角度）",
        "edit_hsl_section": "分色段 (HSL)",
        "edit_hue_shift": "色相偏移（°）",
        "edit_hsl_sat": "分色饱和度",
        "edit_hsl_luma": "分色明度",
        "edit_side_by_side": "并排对比",
        "grid_tip": "提示：滚动浏览，点击缩略图在左侧预览。",
        "ai_menu": "AI",
        "ai_toggle": "启用/禁用 云识别",
        "ai_set_key": "设置 Google API Key…",
        "ai_save_key": "保存密钥（持久化）",
        "ai_clear_key": "清除已保存密钥",
        "ai_need_key": "GOOGLE_API_KEY 未设置。请在系统环境变量或 AI 菜单中设置。",
        "search_ai": "按最终分类（ai_top_label）搜索，例如：flower",
        "ai_header": "-- 分类（AI）--",
        "ai_final": "最终: {label}（置信度 {conf:.2f}）",
        "ai_topk": "Top-5：",
        "ai_no_result": "无 AI 结果",
        "menu_lang": "语言",
        "menu_edit": "编辑",
        "menu_edit_open": "打开编辑器",
        "sug_regen": "重算建议",
        "batch_done": "批量导入完成，共 {n} 张。",
        "edit_title": "编辑",
        "edit_exposure": "曝光补偿 (EV)",
        "edit_gamma": "伽马",
        "edit_contrast": "对比度",
        "edit_saturation": "饱和度",
        "edit_temp": "色温",
        "edit_tint": "色调(偏绿/偏洋红)",
        "edit_clarity": "清晰度(微对比)",
        "edit_denoise": "降噪",
        "edit_grain": "胶片颗粒",
        "edit_highlights": "高光",
        "edit_shadows": "阴影",
        "edit_vibrance": "自然饱和度",
        "edit_vignette": "暗角",
        "edit_sharpen": "锐化",
        "edit_dehaze": "去雾",
        "edit_rotate": "旋转（角度）",
        "edit_hsl_section": "分色段 (HSL)",
        "edit_hsl_band": "分色段",
        "edit_hue_shift": "色相偏移（°）",
        "edit_hsl_sat": "分色饱和度",
        "edit_hsl_luma": "分色明度",
        "edit_iso_up": "提亮预设",
        "edit_iso_down": "降噪预设",
        "edit_flip_h": "水平翻转",
        "edit_flip_v": "垂直翻转",
        "edit_rotate_left": "左转 90°",
        "edit_rotate_right": "右转 90°",
        "edit_auto_wb": "自动白平衡",
        "edit_crop_sq": "居中裁剪 1:1",
        "edit_crop_169": "居中裁剪 16:9",
        "edit_save_preset": "保存预设…",
        "edit_load_preset": "加载预设…",
        "edit_undo": "撤销",
        "edit_redo": "重做",
        "edit_reset": "重置参数",
        "edit_saveas": "另存为…",
        "edit_show_original": "显示原图",
        "edit_side_by_side": "并排对比"
    },
    "en": {
        "title": "Smart Photo Manager (Bilingual + AI)",
        "tab_log": "Log",
        "tab_lib": "Library",
        "tab_meta": "EXIF & Features",
        "tab_sug": "Suggestions",
        "btn_open_image": "Open Image",
        "btn_open_folder": "Import Folder",
        "btn_charts": "Charts",
        "btn_search": "Search",
        "btn_edit": "Edit",
        "dlg_select_image": "Select Image",
        "dlg_select_folder": "Select Folder",
        "edit_dehaze": "Dehaze",
        "edit_rotate": "Rotate (deg)",
        "edit_hsl_section": "HSL Band",
        "edit_hue_shift": "Hue Shift (°)",
        "edit_hsl_sat": "Band Saturation",
        "edit_hsl_luma": "Band Luminance",
        "edit_side_by_side": "Side-by-side",
        "notice_title": "Notice",
        "notice_no_images": "No images available",
        "completed_title": "Completed",
        "completed_msg": "Finished! CSV saved to:\n{path}",
        "exif_header": "-- EXIF --",
        "exif_none": "No EXIF data",
        "sharp_header": "-- Sharpness --",
        "progress_fmt": "Progress: {i}/{total}",
        "charts_title": "Charts (Histogram/Color/Preview)",
        "charts_count_ylabel": "Count",
        "library": "Library",
        "filter": "Filter",
        "filter_all": "All",
        "import_folder": "Import Folder",
        "refresh": "Refresh",
        "grid_tip": "Tip: scroll to browse, click a thumbnail to preview on the left.",
        "ai_menu": "AI",
        "ai_toggle": "Enable/Disable Cloud Tagging",
        "ai_set_key": "Set Google API Key…",
        "ai_save_key": "Save Key (persist)",
        "ai_clear_key": "Clear Saved Key",
        "ai_need_key": "GOOGLE_API_KEY not set. Use AI menu to set it.",
        "search_ai": "Search by final class (ai_top_label), e.g., flower",
        "ai_header": "-- AI Classification --",
        "ai_final": "Final: {label} (conf {conf:.2f})",
        "ai_topk": "Top-5:",
        "ai_no_result": "No AI result",
        "menu_lang": "Language",
        "menu_edit": "Edit",
        "menu_edit_open": "Open Editor",
        "sug_regen": "Regen Tips",
        "batch_done": "Batch import finished: {n} images.",
        "edit_title": "Edit",
        "edit_exposure": "Exposure (EV)",
        "edit_gamma": "Gamma",
        "edit_contrast": "Contrast",
        "edit_saturation": "Saturation",
        "edit_temp": "Temperature",
        "edit_tint": "Tint (G/M)",
        "edit_clarity": "Clarity (local contrast)",
        "edit_denoise": "Denoise",
        "edit_grain": "Film Grain",
        "edit_highlights": "Highlights",
        "edit_shadows": "Shadows",
        "edit_vibrance": "Vibrance",
        "edit_vignette": "Vignette",
        "edit_sharpen": "Sharpen",
        "edit_dehaze": "Dehaze",
        "edit_rotate": "Rotate (degree)",
        "edit_hsl_section": "HSL (Color Range)",
        "edit_hsl_band": "Color Band",
        "edit_hue_shift": "Hue Shift (°)",
        "edit_hsl_sat": "Band Saturation",
        "edit_hsl_luma": "Band Luma",
        "edit_iso_up": "ISO Up Preset",
        "edit_iso_down": "ISO Down Preset",
        "edit_flip_h": "Flip Horizontal",
        "edit_flip_v": "Flip Vertical",
        "edit_rotate_left": "Rotate Left 90°",
        "edit_rotate_right": "Rotate Right 90°",
        "edit_auto_wb": "Auto White Balance",
        "edit_crop_sq": "Center Crop 1:1",
        "edit_crop_169": "Center Crop 16:9",
        "edit_save_preset": "Save Preset…",
        "edit_load_preset": "Load Preset…",
        "edit_undo": "Undo",
        "edit_redo": "Redo",
        "edit_reset": "Reset",
        "edit_saveas": "Save As…",
        "edit_show_original": "Show Original",
        "edit_side_by_side": "Side-by-Side"
    },
}

def t(app, key, **kwargs):
    lang = getattr(app, "lang", "zh")
    s = LANGS.get(lang, LANGS["zh"]).get(key, key)
    return s.format(**kwargs) if kwargs else s

# -------------- API key persistence --------------

SERVICE_NAME = "SmartPhotoAI"
ACCOUNT_NAME = "google_vision_api_key"

def _config_path() -> Path:
    p = Path.home() / ".smartphoto"
    p.mkdir(exist_ok=True)
    return p / "config.json"

def load_api_key_persisted():
    k = os.environ.get("GOOGLE_API_KEY", "").strip()
    if k:
        return k, "env"
    if keyring is not None:
        try:
            k = keyring.get_password(SERVICE_NAME, ACCOUNT_NAME)
            if k:
                return k.strip(), "keyring"
        except Exception:
            pass
    cfg = _config_path()
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            k = (data.get("GOOGLE_API_KEY") or "").strip()
            if k:
                return k, "file"
        except Exception:
            pass
    return "", ""

def save_api_key_persisted(k: str) -> str:
    k = (k or "").strip()
    if not k:
        return ""
    if keyring is not None:
        try:
            keyring.set_password(SERVICE_NAME, ACCOUNT_NAME, k)
            return "keyring"
        except Exception:
            pass
    cfg = _config_path()
    try:
        cfg.write_text(json.dumps({"GOOGLE_API_KEY": k}, ensure_ascii=False, indent=2), encoding="utf-8")
        return "file"
    except Exception:
        return ""

def clear_saved_api_key():
    if keyring is not None:
        try: keyring.delete_password(SERVICE_NAME, ACCOUNT_NAME)
        except Exception: pass
    cfg = _config_path()
    if cfg.exists():
        try: cfg.unlink()
        except Exception: pass

# ---------------- DB ----------------

DB_PATH = Path(__file__).resolve().with_name("smartphoto.db")


# Lightweight thread pool to keep UI responsive
EXECUTOR = ThreadPoolExecutor(max_workers=3)
SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT UNIQUE,
  width INTEGER, height INTEGER, aspect REAL,
  faces INTEGER, brightness REAL,
  iso INTEGER, exposure_s REAL,
  exif_model TEXT, exif_lens TEXT, exif_dt TEXT,
  focal REAL,
  sharp_var REAL,
  -- AI final label only
  ai_provider TEXT,
  ai_raw TEXT,
  ai_top_label TEXT,
  ai_conf REAL,
  ai_ts TEXT,
  -- MUSIQ aesthetic
  musiq_ava REAL,
  musiq_std REAL,
  musiq_ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_ai_label ON images(ai_top_label);
"""

def _migrate(conn: sqlite3.Connection):
    for sql in [
        "ALTER TABLE images ADD COLUMN sharp_var REAL;",
        "ALTER TABLE images ADD COLUMN ai_provider TEXT;",
        "ALTER TABLE images ADD COLUMN ai_raw TEXT;",
        "ALTER TABLE images ADD COLUMN ai_top_label TEXT;",
        "ALTER TABLE images ADD COLUMN ai_conf REAL;",
        "ALTER TABLE images ADD COLUMN ai_ts TEXT;",
        "ALTER TABLE images ADD COLUMN musiq_ava REAL;",
        "ALTER TABLE images ADD COLUMN musiq_std REAL;",
        "ALTER TABLE images ADD COLUMN musiq_ts TEXT;",
    ]:
        try: conn.execute(sql)
        except Exception: pass
    try: conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_label ON images(ai_top_label);")
    except Exception: pass

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# ------------- Image utils -------------

def imread_unicode(path, flag=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0: return None
    return cv2.imdecode(data, flag)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_exif_dict(img):
    exif_raw = getattr(img, "_getexif", lambda: None)()
    return {} if not exif_raw else {ExifTags.TAGS.get(k,k): v for k,v in exif_raw.items()}

def exposure_time_to_float(val):
    try:
        if isinstance(val, tuple) and len(val)==2:
            num, den = val
            return float(num)/float(den) if den else None
        return float(val)
    except Exception:
        return None

def resize_max_edge(bgr, max_edge=1200):
    h, w = bgr.shape[:2]
    m = max(h,w)
    if m <= max_edge: return bgr
    s = max_edge / float(m)
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def assess_sharpness(image_path):
    img_gray = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return 0.0, "Cannot read"
    img_gray = resize_max_edge(img_gray, 1200)
    lap = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    q = "Sharp" if lap>1000 else ("Normal" if lap>300 else "Blurred")
    return float(lap), q

# ---------- Color / Histogram helpers ----------

def color_stats_bgr(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    total = H.size if H.size else 1
    blue_mask  = (90 <= H) & (H <= 130) & (S >= 40) & (V >= 80)
    green_mask = (40 <= H) & (H <= 85)  & (S >= 40) & (V >= 60)
    white_mask = (S < 40) & (V > 180)
    vivid_mask = (S > 100) & (V > 90)
    non_bg     = ~(((90 <= H) & (H <= 130)) | ((40 <= H) & (H <= 85)))
    vivid_non_bg = vivid_mask & non_bg
    return {
        "blue_ratio": round(np.count_nonzero(blue_mask)/total, 3),
        "green_ratio": round(np.count_nonzero(green_mask)/total, 3),
        "lowSat_highVal_ratio": round(np.count_nonzero(white_mask)/total, 3),
        "vivid_non_blue_green_ratio": round(np.count_nonzero(vivid_non_bg)/total, 3),
    }

def rgb_hist_curves(bgr_small):
    rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
    hist_r = cv2.calcHist([rgb], [0], None, [256], [0,256]).flatten()
    hist_g = cv2.calcHist([rgb], [1], None, [256], [0,256]).flatten()
    hist_b = cv2.calcHist([rgb], [2], None, [256], [0,256]).flatten()
    def _norm(h):
        m = h.max() if h.size and h.max() > 0 else 1.0
        return (h / m).astype(np.float32)
    return _norm(hist_r), _norm(hist_g), _norm(hist_b)

# 安全转换
def _safe_int(x):
    try:
        if x is None: return None
        return int(float(x))
    except Exception:
        return None

def _safe_float(x):
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

# ------------- Suggestions -------------

def _safe_shutter_for(focal, faces):
    base = 1/125 if faces else 1/60
    if focal and focal>0: base = max(base, 1.0/float(focal))
    return base

def _stops(a,b):
    try:
        if not a or not b or a<=0 or b<=0: return 0.0
        return math.log2(b/a)
    except Exception:
        return 0.0

def _fmt(s): s=round(s,1); return f"{'+' if s>0 else ''}{s} EV"

def generate_suggestions(feats, sharp_var=None, ai_top=None, ai_pairs=None, lang="zh"):
    """Richer bilingual suggestions, returns list of lines."""
    # Text templates
    T = {
        "zh": {
            "too_slow": "快门 {exp:.3f}s 偏慢（安全≈{safe:.3f}s）：建议提速{ev}或提高 ISO/开大光圈。",
            "portrait_fast": "人像/运动：尽量 ≥1/250s，避免拖影。",
            "dark": "整体偏暗：+0.7~+1.3EV 或提升 1–2 档 ISO。",
            "slightly_dark": "轻微偏暗：+0.3~+0.7EV 或提升 1 档 ISO。",
            "very_bright": "过亮：-0.7~-1.3EV，或使用 ND/CPL。",
            "bright": "较亮：尝试 -0.3~-0.7EV 保护高光。",
            "iso_3200": "ISO≥3200：建议降 1–2 档（至 800–1600），降噪更干净。",
            "iso_1600": "ISO≥1600：考虑 -1 档并配合慢门/大光圈。",
            "soft": "清晰度较低：三脚架/提高快门 1~2 档；单点对焦检查对准。",
            "moderate": "清晰度中等：提升快门 0.5–1 档或连拍。",
            "scene_portrait1": "人像：肤色优先，色温略偏暖；室内保持 ≥1/160s。",
            "scene_portrait2": "背景虚化：50–85mm，开到 ƒ1.8~ƒ2.8。",
            "scene_macro1": "微距：景深很浅，建议堆栈 5–15 张或缩小到 ƒ8–ƒ11。",
            "scene_macro2": "微距手持：快门 ≥1/200s 或使用微距闪光/补光灯。",
            "scene_land1": "风光：使用 CPL 减反光提饱和；三脚架 + 低 ISO。",
            "scene_land2": "日出日落：先保高光，-0.3~-1EV；可包围曝光 3~5 张 HDR。",
            "scene_night1": "夜景：建议 ƒ1.8, 4s, ISO200 起步；直方图中值约 35%。",
            "scene_night2": "城市夜景：关闭 OIS 上三脚架；用延时 2s 避免震动。",
            "scene_food": "美食：近距离 + 斜 45°；适当提升自然饱和度与清晰度。",
            "scene_arch1": "建筑：注意垂直线；若畸变明显，使用透视矫正。",
            "scene_arch2": "室内：提高 ISO 保持 ≥1/125s，避免手震。",
            "scene_bird": "野生动物/鸟类：快门 ≥1/1000s；连拍 + 追焦(AF-C)。",
            "scene_sport": "运动：快门 ≥1/1000s；提高 ISO 保证冻结动作。",
            "overall_ok": "整体观感良好，可适当微调对比与饱和度。"
        },
        "en": {
            "too_slow": "Shutter {exp:.3f}s is slow (safe≈{safe:.3f}s): speed up {ev} or raise ISO / open aperture.",
            "portrait_fast": "Portrait/action: keep ≥1/250s to avoid motion blur.",
            "dark": "Overall dark: +0.7~+1.3 EV or +1–2 ISO stops.",
            "slightly_dark": "Slightly dark: +0.3~+0.7 EV or +1 ISO stop.",
            "very_bright": "Very bright: -0.7~-1.3 EV or use ND/CPL.",
            "bright": "Bright: try -0.3~-0.7 EV to protect highlights.",
            "iso_3200": "ISO≥3200: reduce 1–2 stops (to 800–1600) for cleaner noise.",
            "iso_1600": "ISO≥1600: consider -1 stop with slower shutter/wider aperture.",
            "soft": "Low sharpness: tripod or +1~2 shutter stops; use single-point AF.",
            "moderate": "Moderate sharpness: +0.5–1 shutter stop or shoot in burst.",
            "scene_portrait1": "Portrait: prioritize skin tones; slightly warmer WB; indoor ≥1/160s.",
            "scene_portrait2": "Background blur: 50–85mm at ƒ1.8–ƒ2.8.",
            "scene_macro1": "Macro: depth is shallow; try focus stacking 5–15 shots or stop to ƒ8–ƒ11.",
            "scene_macro2": "Handheld macro: ≥1/200s or use macro flash/lighting.",
            "scene_land1": "Landscape: use CPL; tripod + low ISO.",
            "scene_land2": "Golden hour: protect highlights, -0.3~-1 EV; bracket 3–5 shots for HDR.",
            "scene_night1": "Night: start at ƒ1.8, 4s, ISO200; center histogram ~35%.",
            "scene_night2": "City night: turn off OIS on tripod; 2s self-timer.",
            "scene_food": "Food: close-up at ~45°; increase vibrance and clarity slightly.",
            "scene_arch1": "Architecture: keep verticals straight; use perspective correction if needed.",
            "scene_arch2": "Indoors: raise ISO to keep ≥1/125s to avoid camera shake.",
            "scene_bird": "Wildlife/birds: ≥1/1000s; burst + AF-C tracking.",
            "scene_sport": "Sports: ≥1/1000s; raise ISO to freeze motion.",
            "overall_ok": "Looks good overall; consider slight tweaks to contrast and saturation."
        }
    }[lang if lang in ("zh","en") else "zh"]

    tips=[]
    iso=feats.get("iso")
    exp=feats.get("exposure_s")
    focal=feats.get("focal")
    bri = feats.get("brightness")
    faces=int(feats.get("faces") or 0)

    if isinstance(exp,(int,float)):
        safe=_safe_shutter_for(focal,faces)
        if exp>safe:
            tips.append(T["too_slow"].format(exp=exp, safe=safe, ev=_fmt(_stops(exp,safe))))
        elif faces and exp<1/250:
            tips.append(T["portrait_fast"])
    if isinstance(bri,(int,float)):
        if bri<70: tips.append(T["dark"])
        elif bri<120: tips.append(T["slightly_dark"])
        elif bri>200: tips.append(T["very_bright"])
        elif bri>170: tips.append(T["bright"])
    if isinstance(iso,int):
        if iso>=3200: tips.append(T["iso_3200"])
        elif iso>=1600: tips.append(T["iso_1600"])
    if isinstance(sharp_var,(int,float)):
        if sharp_var<300: tips.append(T["soft"])
        elif sharp_var<1000: tips.append(T["moderate"])

    labs = [l.lower() for (l,_) in (ai_pairs or []) if l]
    s = (ai_top or "").lower() if ai_top else ""
    def any_kw(*kws): 
        return any(kw in s for kw in kws) or any(any(kw in lab for kw in kws) for lab in labs)

    if any_kw("portrait","person","people","face"):
        tips += [T["scene_portrait1"], T["scene_portrait2"]]
    if any_kw("flower","macro","insect","blossom"):
        tips += [T["scene_macro1"], T["scene_macro2"]]
    if any_kw("sky","cloud","sunset","landscape","mountain","sea","ocean"):
        tips += [T["scene_land1"], T["scene_land2"]]
    if any_kw("night","astronomy","milky way","cityscape","neon"):
        tips += [T["scene_night1"], T["scene_night2"]]
    if any_kw("food","dish","plate"):
        tips += [T["scene_food"]]
    if any_kw("architecture","building","bridge","interior"):
        tips += [T["scene_arch1"], T["scene_arch2"]]
    if any_kw("wildlife","bird","animal","pet"):
        tips += [T["scene_bird"]]
    if any_kw("sports","running","ball","soccer","basketball","tennis"):
        tips += [T["scene_sport"]]

    out=[]; seen=set()
    for x in tips:
        if x and x not in seen:
            seen.add(x); out.append(x[:220])
    return out or [T["overall_ok"]]

# ------------- Google Vision -------------

GOOGLE_VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

def img_to_b64_for_api(path: str, max_edge: int = 1200) -> str:
    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    if bgr is None: raise RuntimeError("Cannot read image")
    bgr = resize_max_edge(bgr, max_edge)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok: raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def vision_label_detect_b64(b64: str, api_key: str, max_results: int = 10):
    if not api_key: raise RuntimeError("Missing GOOGLE_API_KEY")
    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [{"type": "LABEL_DETECTION", "maxResults": max_results}]
        }]
    }
    r = requests.post(f"{GOOGLE_VISION_ENDPOINT}?key={api_key}", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    anns = data.get("responses", [{}])[0].get("labelAnnotations", []) or []
    return [(it.get("description",""), float(it.get("score",0.0))) for it in anns]

# ------------- App -------------

class App(tk.Tk):

    def open_from_path(self, path: str):
        """Sync wrapper used by library grid / regen; internally delegates to async worker."""
        self.open_from_path_async(path)

    def open_from_path_async(self, path: str):
        """主线程只做预览与状态，重活丢到后台线程"""
        import threading
        self._last_opened_path = path
        self.status.set(f"Selected: {path}")
        self.text.delete(1.0, tk.END)
        self.sug_list.delete(0, tk.END)
        for i in self.meta_tree.get_children():
            self.meta_tree.delete(i)

        # 先快速预览
        try:
            with Image.open(path) as img:
                img.thumbnail((900, 900))
                self._last_img_tk = ImageTk.PhotoImage(img)
            self.preview.config(image=self._last_img_tk, text="")
        except Exception as e:
            self.preview.config(text="Cannot preview", image="")
            self.ui_log(f"[Error] Preview failed: {e}")

        # 转菊花
        try:
            self.prog["mode"] = "indeterminate"
            self.prog.start(12)
        except Exception:
            pass
        self.status.set("Loading…")

        threading.Thread(target=self._open_worker, args=(path,), daemon=True).start()

    def _open_worker(self, path: str):
        # —— 阶段 1：迅速拿到能拿到的一切（本地计算）——
        exif = {}
        sharp_val, sharp_q = 0.0, "N/A"
        feats = {"faces":0, "brightness":None, "iso":None, "exposure_s":None, "focal":None}
        bgr = imread_unicode(path, cv2.IMREAD_COLOR)

        # EXIF
        try:
            with Image.open(path) as img2:
                exif = get_exif_dict(img2)
        except Exception:
            pass
        feats["iso"] = _safe_int(exif.get("ISOSpeedRatings") if isinstance(exif, dict) else None)
        feats["exposure_s"] = _safe_float(exposure_time_to_float(exif.get("ExposureTime"))) if isinstance(exif,dict) else None
        fl = exif.get("FocalLength") if isinstance(exif, dict) else None
        if isinstance(fl, tuple) and len(fl)==2:
            feats["focal"] = _safe_float(float(fl[0])/float(fl[1]) if fl[1] else None)
        elif fl is not None:
            feats["focal"] = _safe_float(fl)

        # 清晰度 + 亮度（本地很快）
        try:
            sharp_val, sharp_q = assess_sharpness(path)
        except Exception:
            pass
        if bgr is not None:
            gimg = cv2.cvtColor(resize_max_edge(bgr, 1200), cv2.COLOR_BGR2GRAY)
            feats["brightness"] = float(np.mean(gimg))

        # —— 立即把“轻量信息”推到 UI（不用等 AI/MUSIQ）——
        def _show_fast():
            self.status.set("Ready")
            try:
                self.prog.stop(); self.prog["mode"]="determinate"; self.prog["value"]=0
            except Exception: pass

            self.text.delete(1.0, tk.END)
            for i in self.meta_tree.get_children():
                self.meta_tree.delete(i)

            # EXIF
            self.ui_log(t(self,"exif_header"))
            show = ["Model","LensModel","ISOSpeedRatings","ExposureTime","FNumber","FocalLength","DateTimeOriginal"]
            if exif:
                for k in show:
                    if k in exif:
                        self.ui_log(f"{k}: {exif[k]}")
                        self.meta_tree.insert("", tk.END, values=(k, exif[k]))
            else:
                self.ui_log(t(self,"exif_none"))

            # 清晰度
            self.ui_log("\n"+t(self,"sharp_header"))
            self.ui_log(f"Var: {sharp_val:.2f}  [{sharp_q}]")
            self.meta_tree.insert("", tk.END, values=("Sharpness Var", f"{sharp_val:.2f}"))
            self.meta_tree.insert("", tk.END, values=("Sharpness Quality", sharp_q))

            # 先产出一版建议（没有 AI 标签也能给）
            self.sug_list.delete(0, tk.END)
            tips = generate_suggestions(feats, sharp_var=sharp_val, ai_top=None, ai_pairs=None,
                                        lang=getattr(self,'lang','zh')) or []
            for tip in tips: self.sug_list.insert(tk.END, tip)
            self.ui_log(f"[Suggest] {len(tips)} tips generated (fast).")

        self.after(0, _show_fast)

        # —— 先把“本地阶段”的结果落库（AI/MUSIQ 先空着）——
        try:
            self._upsert_db_final(path, bgr, sharp_val, exif,
                                ai_pairs=None, ai_top_label=None, ai_conf=None, ai_provider=None,
                                musiq_ava=None, musiq_std=None)
            self.after(0, self.refresh_filters)
        except Exception as e:
            self.after(0, self.ui_log, f"[DB Error] {e}")

        # —— 阶段 2：AI 与 MUSIQ 并行异步，完成后各自增量更新 UI + DB —— 
        def _ai_worker():
            ai_pairs=[]; ai_top=None; ai_conf=0.0; ai_provider=None
            if self.ai_enabled:
                try:
                    if not self.ai_api_key:
                        raise RuntimeError(t(self,"ai_need_key"))
                    b64 = img_to_b64_for_api(path, 1200)
                    ai_pairs = vision_label_detect_b64(b64, self.ai_api_key, max_results=10)
                    if ai_pairs:
                        ai_top, ai_conf = ai_pairs[0][0], ai_pairs[0][1]
                    ai_provider="google_vision"
                except Exception as e:
                    self.after(0, self.ui_log, f"[AI Error] {e}")

            # UI 增量更新：AI 块 + 重新生成含场景的建议
            def _ai_ui():
                if ai_pairs:
                    self._log_ai_block(ai_pairs, ai_top, ai_conf)
                    self.ui_log(f"AI Final: {ai_top or '-'} (conf={ai_conf:.2f})")
                    # 覆盖建议
                    self.sug_list.delete(0, tk.END)
                    tips = generate_suggestions(feats, sharp_var=sharp_val,
                                                ai_top=ai_top, ai_pairs=ai_pairs,
                                                lang=getattr(self,'lang','zh')) or []
                    for tip in tips: self.sug_list.insert(tk.END, tip)
                    self.ui_log(f"[Suggest] updated with AI context.")
            self.after(0, _ai_ui)

            # DB 增量更新：写入 AI 字段
            try:
                self._upsert_db_final(path, bgr, sharp_val, exif,
                                    ai_pairs, ai_top, ai_conf, ai_provider,
                                    musiq_ava=None, musiq_std=None)
                self.after(0, self.refresh_filters)
            except Exception as e:
                self.after(0, self.ui_log, f"[DB Error] {e}")

        def _musiq_worker():
            musiq=None; musiq_std=None
            if self.musiq_enabled and bgr is not None:
                try:
                    rgb_small = cv2.cvtColor(resize_max_edge(bgr, 1200), cv2.COLOR_BGR2RGB)
                    musiq, musiq_std, err = musiq_score_ndarray_rgb(rgb_small)
                    if err:
                        self.after(0, self.ui_log, f"[MUSIQ] {err}")
                except Exception as e:
                    self.after(0, self.ui_log, f"[MUSIQ Error] {e}")

            # UI 增量更新：显示分数
            def _musiq_ui():
                if musiq is not None:
                    self.ui_log(f"MUSIQ(AVA): {musiq:.2f}" + (f"  (±{musiq_std:.2f})" if musiq_std else ""))
                    self.meta_tree.insert("", tk.END, values=("MUSIQ (AVA)", f"{musiq:.2f}"))
            self.after(0, _musiq_ui)

            # DB 增量更新：写 MUSIQ 字段
            try:
                self._upsert_db_final(path, bgr, sharp_val, exif,
                                    ai_pairs=None, ai_top_label=None, ai_conf=None, ai_provider=None,
                                    musiq_ava=musiq, musiq_std=musiq_std)
            except Exception as e:
                self.after(0, self.ui_log, f"[DB Error] {e}")

        # 并行跑
        threading.Thread(target=_ai_worker,    daemon=True).start()
        threading.Thread(target=_musiq_worker, daemon=True).start()



    # === 放在 class App 里面 ===

    def start_batch_thread(self):
        """选择文件夹并在后台线程批量处理，不阻塞 UI"""
        folder = filedialog.askdirectory(title=t(self, "dlg_select_folder"))
        if not folder:
            return

        # 清空右侧面板
        self.text.delete(1.0, tk.END)
        self.sug_list.delete(0, tk.END)
        for i in self.meta_tree.get_children():
            self.meta_tree.delete(i)

        # 进度动画
        self.status.set("Classifying…")
        try:
            self.prog["mode"] = "indeterminate"
            self.prog.start(12)
        except Exception:
            pass

        # 后台执行批处理
        threading.Thread(target=self._batch_worker, args=(folder,), daemon=True).start()

    # --- helpers ---
    def _rgb_hist(self, bgr_small):
        rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
        hist_r = cv2.calcHist([rgb], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([rgb], [1], None, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([rgb], [2], None, [256], [0, 256]).flatten()
        return hist_r, hist_g, hist_b

    def _exif_table_pairs(self, exif: dict):
        keys = ["Model","LensModel","ISOSpeedRatings","ExposureTime","FNumber",
                "FocalLength","DateTimeOriginal"]
        out=[]
        for k in keys:
            v = exif.get(k,"-")
            if isinstance(v, tuple) and len(v)==2 and all(v):
                try:
                    v = f"{float(v[0])/float(v[1]):.2f}"
                except Exception:
                    v = f"{v[0]}/{v[1]}"
            out.append((k,str(v)))
        return out

    def _log_ai_block(self, ai_pairs, final_label, final_conf, top_k: int = 5):
        def _conf_bar(score: float, width: int = 24) -> str:
            try:
                v = 0.0 if score is None else float(score)
            except Exception:
                v = 0.0
            v = max(0.0, min(1.0, v))
            filled = int(round(v * width))
            return "█" * filled + "·" * (width - filled)

        self.ui_log("")
        self.ui_log("-- AI --")
        if final_label:
            if final_conf is not None:
                self.ui_log(f"Final : {final_label}  (conf≈{final_conf:.2f})")
            else:
                self.ui_log(f"Final : {final_label}")
        else:
            self.ui_log("Final : (no result)")

        if ai_pairs:
            self.ui_log(f"Top-{min(top_k, len(ai_pairs))}:")
            for (lab, sc) in ai_pairs[:top_k]:
                lab = lab or "-"
                bar = _conf_bar(sc, 24)
                try:
                    self.ui_log(f"  {lab:<24} [{bar}]  {sc:>5.2f}")
                except Exception:
                    self.ui_log(f"  {str(lab)[:24]:<24} [{bar}]  {sc:>5.2f}")
        else:
            self.ui_log("Top-5: (no predictions)")
        self.ui_log("")

    # --- init ---
    def __init__(self):
        super().__init__()
        self.lang = "zh"
        self.conn = get_db()
        self.title(t(self,"title"))
        self.geometry("1280x840")
        self.minsize(1000,700)

        self._last_img_tk = None
        self._last_opened_path = None

        self.ai_api_key, src = load_api_key_persisted()
        self.ai_enabled = bool(self.ai_api_key)
        self.face_detect_enabled = False
        self.musiq_enabled = bool(_TF_AVAILABLE)

        try: self.tk.call('tk','scaling',1.25)
        except: pass

        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.palette = {"bg":"#0f1115","panel":"#171a21","fg":"#e6e8eb","muted":"#98a2b3","accent":"#5b9dff","border":"#242834"}
        self.configure(bg=self.palette["bg"])
        self._apply_styles()

        # Toolbar
        bar = ttk.Frame(self, style="Panel.TFrame", padding=(10,8)); bar.pack(side=tk.TOP, fill=tk.X)
        self.btn_open_image = ttk.Button(bar, text="📷 "+t(self,"btn_open_image"), command=self.open_single, style="Accent.TButton")
        self.btn_open_folder= ttk.Button(bar, text="🗂 "+t(self,"btn_open_folder"), command=self.start_batch_thread)
        self.btn_charts     = ttk.Button(bar, text="📊 "+t(self,"btn_charts"), command=self._charts_threaded)
        self.btn_search     = ttk.Button(bar, text="🔎 "+t(self,"btn_search"), command=self.search_dialog)
        self.btn_regen_sug = ttk.Button(bar, text="✨ "+t(self,"sug_regen"), command=self.regenerate_suggestions)
        self.btn_edit       = ttk.Button(bar, text="🛠 "+t(self,"btn_edit"), command=self.open_editor)
        for b in (self.btn_open_image,self.btn_open_folder,self.btn_charts,self.btn_search,self.btn_edit): b.pack(side=tk.LEFT, padx=4)

        rightbar = ttk.Frame(bar, style="Panel.TFrame"); rightbar.pack(side=tk.RIGHT)
        self.lang_var = tk.StringVar(value="中文")
        cb_lang = ttk.Combobox(rightbar, width=8, state="readonly", textvariable=self.lang_var, values=["中文","English"])
        cb_lang.bind("<<ComboboxSelected>>", self._on_lang_change); cb_lang.pack(side=tk.RIGHT, padx=6)

        self.theme_var = tk.StringVar(value="Dark")
        cb_theme = ttk.Combobox(rightbar, width=8, state="readonly", textvariable=self.theme_var, values=["Dark","Light"])
        cb_theme.bind("<<ComboboxSelected>>", self._on_theme_change); cb_theme.pack(side=tk.RIGHT, padx=6)

        # Split
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = ttk.Frame(main, style="Panel.TFrame", padding=10); main.add(left, weight=2)
        self.preview = tk.Label(left, bg=self.palette["panel"], fg=self.palette["muted"], width=60, height=25, text="⟵ "+t(self,"btn_open_image"))
        self.preview.pack(fill=tk.BOTH, expand=True)

        # Notebook
        self.nb = ttk.Notebook(main, style="TNotebook"); main.add(self.nb, weight=3)

        # Log tab
        self.tab_log = ttk.Frame(self.nb, style="Panel.TFrame", padding=10)
        self.text = tk.Text(self.tab_log, relief=tk.FLAT, bg=self.palette["panel"], fg=self.palette["fg"], insertbackground=self.palette["fg"], height=12)
        self.text.pack(fill=tk.BOTH, expand=True)

        # Library tab
        self.tab_lib = ttk.Frame(self.nb, style="Panel.TFrame", padding=10)
        self._build_library_ui(self.tab_lib)

        # Meta tab
        self.tab_meta = ttk.Frame(self.nb, style="Panel.TFrame", padding=10)
        self.meta_tree = ttk.Treeview(self.tab_meta, columns=("k","v"), show="headings", height=14)
        self.meta_tree.heading("k", text="Key"); self.meta_tree.heading("v", text="Value")
        self.meta_tree.column("k", width=220, anchor="w"); self.meta_tree.column("v", width=480, anchor="w")
        self.meta_tree.pack(fill=tk.BOTH, expand=True)

        # Suggestions tab
        self.tab_sug = ttk.Frame(self.nb, style="Panel.TFrame", padding=10)
        self.sug_list = tk.Listbox(self.tab_sug, relief=tk.FLAT, bg=self.palette["panel"], fg=self.palette["fg"], height=14)
        self.sug_list.pack(fill=tk.BOTH, expand=True)

        # add tabs
        self.nb.add(self.tab_log, text=t(self,"tab_log"))
        self.nb.add(self.tab_lib, text=t(self,"tab_lib"))
        self.nb.add(self.tab_meta, text=t(self,"tab_meta"))
        self.nb.add(self.tab_sug, text=t(self,"tab_sug"))

        # Status
        status = ttk.Frame(self, style="Panel.TFrame"); status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status, style="Muted.TLabel").pack(side=tk.LEFT, padx=10, pady=6)
        self.prog = ttk.Progressbar(status, mode="determinate", style="Horizontal.TProgressbar", length=200)
        self.prog.pack(side=tk.RIGHT, padx=10, pady=8)

        # Menu
        menubar = tk.Menu(self)
        mlang = tk.Menu(menubar, tearoff=0)
        mlang.add_command(label="中文", command=lambda:self.switch_lang("zh"))
        mlang.add_command(label="English", command=lambda:self.switch_lang("en"))
        menubar.add_cascade(label=t(self,"menu_lang"), menu=mlang)

        mai = tk.Menu(menubar, tearoff=0)
        def toggle_ai():
            self.ai_enabled = not self.ai_enabled
            messagebox.showinfo("AI", f"Cloud AI Tagging {'ENABLED' if self.ai_enabled else 'DISABLED'}")
        def set_api_key():
            k = simpledialog.askstring("Google Vision Key", "Enter GOOGLE_API_KEY:", initialvalue=self.ai_api_key, show="*")
            if k is not None:
                self.ai_api_key = k.strip()
                messagebox.showinfo("AI", "API key set for this session.")
        def save_current_key():
            src = save_api_key_persisted(self.ai_api_key or "")
            messagebox.showinfo("AI", f"Saved to {src or 'N/A'}")
        def clear_key():
            clear_saved_api_key()
            messagebox.showinfo("AI", "Cleared.")
        mai.add_command(label=t(self,"ai_toggle"), command=toggle_ai)
        mai.add_command(label=t(self,"ai_set_key"), command=set_api_key)
        mai.add_separator()
        mai.add_command(label=t(self,"ai_save_key"), command=save_current_key)
        mai.add_command(label=t(self,"ai_clear_key"), command=clear_key)
        menubar.add_cascade(label=t(self,"ai_menu"), menu=mai)

        # Face menu (toggle heavy face detection)
        mface = tk.Menu(menubar, tearoff=0)
        def toggle_face():
            self.face_detect_enabled = not self.face_detect_enabled
            messagebox.showinfo("Face", f"Face detection {'ENABLED' if self.face_detect_enabled else 'DISABLED'}")
        mface.add_command(label="Enable/Disable Face Detect", command=toggle_face)
        menubar.add_cascade(label="Face", menu=mface)


        medit = tk.Menu(menubar, tearoff=0)
        medit.add_command(label=t(self,"menu_edit_open"), command=self.open_editor)
        menubar.add_cascade(label=t(self,"menu_edit"), menu=medit)

        self.config(menu=menubar)

        # Shortcuts
        self.bind("<Control-o>", lambda e:self.open_single())
        self.bind("<Control-f>", lambda e:self.start_batch_thread())
        self.bind("<Control-s>", lambda e:self.search_dialog())
        self.bind("<Control-r>", lambda e:self.refresh_filters())
        self.bind("<Control-e>", lambda e:self.open_editor())

        self.ui_log(f"[AI] API key source: {src or 'none'}")
        if self.ai_enabled:
            self.ui_log("[AI] Cloud tagging auto-enabled")

        self.relabel_ui()
        self.refresh_filters()
        # Regenerate suggestions for the currently opened image in new language
        if getattr(self, "_last_opened_path", None):
            self.regenerate_suggestions()
        if _TF_AVAILABLE:
            try:
                m = _load_musiq_model_blocking()
                self.ui_log("[MUSIQ] Loaded.")
                _debug_log_musiq_signatures(m, ui_log=self.ui_log)
            except Exception as e:
                self.ui_log(f"[MUSIQ] Load error: {e}")

    # ---- styles / relabel ----
    def _apply_styles(self):
        p=self.palette
        self.style.configure("TFrame", background=p["bg"])
        self.style.configure("Panel.TFrame", background=p["panel"])
        self.style.configure("TLabel", background=p["bg"], foreground=p["fg"])
        self.style.configure("Panel.TLabel", background=p["panel"], foreground=p["fg"])
        self.style.configure("Muted.TLabel", background=p["bg"], foreground=p["muted"])
        self.style.configure("TButton", padding=8)
        self.style.configure("Accent.TButton", padding=8)
        self.style.map("Accent.TButton", foreground=[('active',p["fg"])], background=[('!disabled',p["accent"])])
        self.style.configure("TNotebook", background=p["bg"])
        self.style.configure("TNotebook.Tab", padding=(12,7))
        self.style.configure("Treeview", background=p["panel"], fieldbackground=p["panel"], foreground=p["fg"])
        self.style.configure("Horizontal.TProgressbar", troughcolor=p["panel"])

    def _on_theme_change(self, *_):
        if self.theme_var.get()=="Light":
            self.palette.update({"bg":"#F6F7FB","panel":"#FFFFFF","fg":"#101828","muted":"#667085","accent":"#3B82F6","border":"#E4E7EC"})
        else:
            self.palette.update({"bg":"#0f1115","panel":"#171a21","fg":"#e6e8eb","muted":"#98a2b3","accent":"#5b9dff","border":"#242834"})
        self.configure(bg=self.palette["bg"])
        self._apply_styles()
        self.preview.configure(bg=self.palette["panel"], fg=self.palette["muted"])
        self.text.configure(bg=self.palette["panel"], fg=self.palette["fg"], insertbackground=self.palette["fg"])
        self.sug_list.configure(bg=self.palette["panel"], fg=self.palette["fg"])

    def _on_lang_change(self, *_):
        self.switch_lang("zh" if self.lang_var.get()=="中文" else "en")

    
    def switch_lang(self, lang):
        self.lang = lang
        self.lang_var.set("中文" if lang=="zh" else "English")
        self.relabel_ui()
        # 更新 Tab 文案
        self.nb.tab(self.tab_log, text=t(self,"tab_log"))
        self.nb.tab(self.tab_lib, text=t(self,"tab_lib"))
        self.nb.tab(self.tab_meta, text=t(self,"tab_meta"))
        self.nb.tab(self.tab_sug, text=t(self,"tab_sug"))
        # 更新 Library 顶栏与下拉
        self._relabel_library_top()
        self.refresh_filters()
        # 语言切换后，重新生成当前图片的建议（如果有）
        try:
            if getattr(self, "_last_opened_path", None):
                self.regenerate_suggestions()
        except Exception:
            pass

    def regenerate_suggestions(self):
        if not getattr(self, '_last_opened_path', None):
            return
        self.open_from_path(self._last_opened_path)

    def relabel_ui(self):

        self.title(t(self,"title"))
        self.btn_open_image.config(text="📷 "+t(self,"btn_open_image"))
        self.btn_open_folder.config(text="🗂 "+t(self,"btn_open_folder"))
        self.btn_charts.config(text="📊 "+t(self,"btn_charts"))
        self.btn_search.config(text="🔎 "+t(self,"btn_search"))
        self.btn_regen_sug.config(text="✨ "+t(self,"sug_regen"))
        self.btn_edit.config(text="🛠 "+t(self,"btn_edit"))
        self.preview.config(text="⟵ "+t(self,"btn_open_image"))

    # ---- log helper ----
    def ui_log(self, s):
        self.text.insert(tk.END, s+"\n"); self.text.see(tk.END)

    # ---- single image flow ----

    def open_single(self):
        path = filedialog.askopenfilename(
            title=t(self, "dlg_select_image"),
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")],
        )
        if not path:
            return
        self.open_from_path_async(path)

    # (duplicate of worker consolidated above) — omitted here

    def _charts_threaded(self):
        if not getattr(self, "_last_opened_path", None):
            messagebox.showinfo(t(self, "notice_title"), t(self, "dlg_select_image"))
            return
        threading.Thread(target=self._build_and_show_charts, daemon=True).start()

    def _build_and_show_charts(self):
        try:
            img_path = getattr(self, "_last_opened_path", None)
            if not img_path:
                self.after(0, lambda: messagebox.showinfo(t(self, "notice_title"), t(self, "dlg_select_image")))
                return

            bgr = imread_unicode(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                self.after(0, lambda: messagebox.showerror("Error", "Cannot read image."))
                return

            gray_small = cv2.cvtColor(resize_max_edge(bgr, 1200), cv2.COLOR_BGR2GRAY)
            bgr_small  = resize_max_edge(bgr, 1200)

            hist_gray = cv2.calcHist([gray_small], [0], None, [256], [0, 256]).flatten()
            hist_r, hist_g, hist_b = rgb_hist_curves(bgr_small)

            sharp_var, sharp_q = assess_sharpness(img_path)

            exif_pairs = []
            try:
                with Image.open(img_path) as img2:
                    exif = get_exif_dict(img2)
                for k in ["Model", "LensModel", "ISOSpeedRatings", "ExposureTime",
                        "FNumber", "FocalLength", "DateTimeOriginal"]:
                    if k in exif:
                        v = exif[k]
                        if isinstance(v, tuple) and len(v) == 2 and all(v):
                            try:
                                v = f"{float(v[0]) / float(v[1]):.2f}"
                            except Exception:
                                v = f"{v[0]}/{v[1]}"
                        exif_pairs.append((k, str(v)))
            except Exception:
                pass

            cs = color_stats_bgr(bgr_small)
            rgb_preview = cv2.cvtColor(resize_max_edge(bgr, 720), cv2.COLOR_BGR2RGB)

            fig = Figure(figsize=(10.8, 6.4), tight_layout=True)

            ax1 = fig.add_subplot(221)
            ax1.fill_between(np.arange(256), hist_gray, step="pre")
            ax1.set_title("Grayscale Histogram")
            ax1.set_xlabel("Gray (0~255)")
            ax1.set_ylabel("Count")

            ax2 = fig.add_subplot(222)
            x = np.arange(256)
            ax2.plot(x, hist_r, label="R")
            ax2.plot(x, hist_g, label="G")
            ax2.plot(x, hist_b, label="B")
            ax2.set_title("RGB Curves (normalized)")
            ax2.set_xlim(0, 255)
            ax2.set_ylim(0, 1.05)
            ax2.legend(loc="upper right")

            ax3 = fig.add_subplot(223)
            ax3.axis("off")
            lines = [f"Sharpness: {sharp_var:.2f} [{sharp_q}]"]
            try:
                cur = self.conn.cursor()
                cur.execute("SELECT ai_top_label, ai_conf FROM images WHERE path=?", (str(img_path),))
                row = cur.fetchone()
                if row:
                    ai_lab, ai_conf = row
                    if ai_lab:
                        if ai_conf is not None:
                            lines.append(f"AI Final: {ai_lab} (conf={ai_conf:.2f})")
                        else:
                            lines.append(f"AI Final: {ai_lab}")
            except Exception:
                pass

            if exif_pairs:
                lines.append("")
                lines.append("-- EXIF --")
                for k, v in exif_pairs[:8]:
                    lines.append(f"{k}: {v}")

            lines.append("")
            lines.append("-- Color Ratios --")
            lines.append(
                f"Blue: {cs['blue_ratio']:.3f} | "
                f"Green: {cs['green_ratio']:.3f} | "
                f"White(hiV lowS): {cs['lowSat_highVal_ratio']:.3f}"
            )
            lines.append(f"Vivid(!Blue&!Green): {cs['vivid_non_blue_green_ratio']:.3f}")

            ax3.text(0.02, 0.98, "\n".join(lines), va="top", ha="left")

            ax4 = fig.add_subplot(224)
            ax4.axis("off")
            ax4.imshow(rgb_preview)
            ax4.set_title(Path(img_path).name)

            def show():
                win = tk.Toplevel(self)
                win.title(t(self, "charts_title"))
                canvas = FigureCanvasTkAgg(fig, master=win)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas.draw()

            self.after(0, show)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

    # ---- search ----
    def search_dialog(self):
        q = simpledialog.askstring(t(self,"btn_search"), t(self,"search_ai"))
        if not q: return
        cur = self.conn.cursor()
        cur.execute("SELECT path, ai_top_label, ai_conf FROM images WHERE LOWER(ai_top_label)=LOWER(?) ORDER BY path", (q.strip(),))
        res=cur.fetchall()
        out=[f"{Path(p).name}: {lab or '-'} ({conf if conf else '-'})" for (p,lab,conf) in res]
        win=tk.Toplevel(self); win.title(t(self,"btn_search"))
        txt=tk.Text(win,width=100,height=22); txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, "\n".join(out) if out else "No results")

    # ---- editor ----
    def open_editor(self):
        if not getattr(self, "_last_opened_path", None):
            messagebox.showinfo(t(self,"notice_title"), t(self,"dlg_select_image"))
            return
        try:
            bgr = imread_unicode(self._last_opened_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("Cannot read image")
            EditDialog(self, bgr, self._last_opened_path)
        except Exception as e:
            messagebox.showerror("Edit", str(e))

    # ---- library UI ----
    def _build_library_ui(self, parent: tk.Widget):
        self.lib_top = ttk.Frame(parent, style="Panel.TFrame"); self.lib_top.pack(fill=tk.X)
        self.lbl_filter = ttk.Label(self.lib_top, text=t(self,"filter"), style="Panel.TLabel")
        self.lbl_filter.pack(side=tk.LEFT)
        self.filter_var = tk.StringVar(value=t(self,"filter_all"))
        self.filter_combo = ttk.Combobox(self.lib_top, textvariable=self.filter_var, width=22, state="readonly")
        self.filter_combo.pack(side=tk.LEFT, padx=8)
        self.filter_combo.bind("<<ComboboxSelected>>", lambda e:self.rebuild_gallery())
        self.btn_refresh = ttk.Button(self.lib_top, text=t(self,"refresh"), command=self.refresh_filters)
        self.btn_refresh.pack(side=tk.LEFT, padx=6)
        self.btn_import = ttk.Button(self.lib_top, text=t(self,"import_folder"), command=self.start_batch_thread)
        self.btn_import.pack(side=tk.LEFT, padx=6)
        self.lbl_tip = ttk.Label(self.lib_top, text=t(self,"grid_tip"), style="Muted.TLabel")
        self.lbl_tip.pack(side=tk.RIGHT)

        self.lib_canvas = tk.Canvas(parent, highlightthickness=0, bg=self.palette["panel"], height=440)
        self.lib_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.lib_canvas.yview)
        self.lib_canvas.configure(yscrollcommand=self.lib_scroll.set)
        self.lib_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lib_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.grid_frame = ttk.Frame(self.lib_canvas, style="Panel.TFrame")
        self.lib_canvas.create_window((0,0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", lambda e: self.lib_canvas.configure(scrollregion=self.lib_canvas.bbox("all")))
        self.lib_canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

    def _relabel_library_top(self):
        self.lbl_filter.config(text=t(self,"filter"))
        self.btn_refresh.config(text=t(self,"refresh"))
        self.btn_import.config(text=t(self,"import_folder"))
        self.lbl_tip.config(text=t(self,"grid_tip"))

    def _on_mouse_wheel(self, event):
        delta = -1*(event.delta//120)
        self.lib_canvas.yview_scroll(delta, "units")

    def refresh_filters(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM images")
        total = cur.fetchone()[0] or 0
        cur.execute("""
            SELECT DISTINCT TRIM(LOWER(ai_top_label))
            FROM images
            WHERE ai_top_label IS NOT NULL AND ai_top_label!=''
            ORDER BY 1
        """)
        labels = [row[0] for row in cur.fetchall() if row[0]]
        values = [t(self, "filter_all")] + labels
        self.filter_combo.configure(values=values)
        if self.filter_var.get() not in values:
            self.filter_var.set(t(self, "filter_all"))
        self.ui_log(f"[Library] {total} photos, {len(labels)} classes: {', '.join(labels[:10])}{' ...' if len(labels)>10 else ''}")
        self.rebuild_gallery()

    def _gallery_query(self):
        cur = self.conn.cursor()
        flt = (self.filter_var.get() or "").strip()
        if flt == t(self, "filter_all"):
            cur.execute(
                "SELECT path, ai_top_label, ai_conf FROM images WHERE ai_top_label IS NOT NULL AND ai_top_label!='' ORDER BY path"
            )
        else:
            cur.execute(
                "SELECT path, ai_top_label, ai_conf FROM images WHERE ai_top_label IS NOT NULL AND TRIM(LOWER(ai_top_label)) = ? ORDER BY path",
                (flt.strip().lower(),)
            )
        return cur.fetchall()

    def rebuild_gallery(self):
        try: self.lib_canvas.yview_moveto(0.0)
        except: pass
        for w in self.grid_frame.winfo_children():
            w.destroy()
        rows = self._gallery_query()
        if not rows:
            ttk.Label(self.grid_frame, text="(empty)", style="Muted.TLabel")\
                .grid(row=0, column=0, padx=10, pady=10, sticky="w")
            return
        cell_w, cell_h = 180, 200
        width_now = max(self.grid_frame.winfo_width(), 600)
        cols = max(3, min(6, width_now // cell_w))
        for i, (path, ai_lab, ai_conf) in enumerate(rows):
            r, c = divmod(i, cols)
            cell = ttk.Frame(self.grid_frame, style="Panel.TFrame")
            cell.grid(row=r, column=c, padx=8, pady=10, sticky="n")
            tkimg = None
            try:
                with Image.open(path) as img:
                    img.thumbnail((cell_w - 20, cell_h - 60))
                    tkimg = ImageTk.PhotoImage(img)
            except Exception:
                ph = tk.Canvas(cell, width=cell_w - 20, height=cell_h - 60,
                               bg=self.palette["panel"], highlightthickness=1,
                               highlightbackground=self.palette["border"])
                ph.create_text((cell_w - 20)//2, (cell_h - 60)//2,
                               text="(Preview failed)", fill=self.palette["muted"])
                ph.pack()
            if tkimg is not None:
                lbl = tk.Label(cell, image=tkimg, bg=self.palette["panel"])
                lbl.image = tkimg
                lbl.pack()
                lbl.bind("<Button-1>", lambda e, p=path: self.open_from_path(p))
            sub = f"{ai_lab or '-'} ({ai_conf:.2f})" if ai_lab else "-"
            ttk.Label(cell, text=f"{Path(path).name}\n[{sub}]",
                      style="Panel.TLabel", justify="center").pack(pady=(4, 0))

    # ---- DB helpers ----
    def _upsert_db_final(
        self,
        path,
        bgr,
        sharp_val,
        exif,
        ai_pairs,
        ai_top_label,
        ai_conf,
        ai_provider,
        musiq_ava: float | None = None,
        musiq_std: float | None = None,
    ):
        # —— 使用短连接，允许在后台线程写库 ——
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(SCHEMA)
            _migrate(conn)
            cur = conn.cursor()

            width  = int(bgr.shape[1]) if bgr is not None else None
            height = int(bgr.shape[0]) if bgr is not None else None
            aspect = (width/height) if width and height else None

            exif_model = exif.get("Model") if isinstance(exif, dict) else None
            exif_lens  = exif.get("LensModel") if isinstance(exif, dict) else None
            exif_dt    = exif.get("DateTimeOriginal") if isinstance(exif, dict) else None

            focal = None
            fl = exif.get("FocalLength") if isinstance(exif, dict) else None
            if isinstance(fl, tuple) and len(fl) == 2 and fl[1]:
                try:
                    focal = float(fl[0]) / float(fl[1])
                except Exception:
                    focal = None
            elif fl is not None:
                try:
                    focal = float(fl)
                except Exception:
                    focal = None

            ai_raw_json = json.dumps(ai_pairs or [], ensure_ascii=False)

            ts = datetime.datetime.now().isoformat(timespec="seconds")
            musiq_ts = ts if musiq_ava is not None else None  # 只有有分数时写入时间

            cur.execute("""
            INSERT INTO images(
                path, width, height, aspect, faces, brightness, iso, exposure_s,
                exif_model, exif_lens, exif_dt, focal, sharp_var,
                ai_provider, ai_raw, ai_top_label, ai_conf, ai_ts,
                musiq_ava, musiq_std, musiq_ts
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
                width=excluded.width,
                height=excluded.height,
                aspect=excluded.aspect,
                faces=excluded.faces,
                brightness=excluded.brightness,
                iso=excluded.iso,
                exposure_s=excluded.exposure_s,
                exif_model=excluded.exif_model,
                exif_lens=excluded.exif_lens,
                exif_dt=excluded.exif_dt,
                focal=excluded.focal,
                sharp_var=excluded.sharp_var,
                ai_provider=excluded.ai_provider,
                ai_raw=excluded.ai_raw,
                ai_top_label=excluded.ai_top_label,
                ai_conf=excluded.ai_conf,
                ai_ts=excluded.ai_ts,
                musiq_ava=excluded.musiq_ava,
                musiq_std=excluded.musiq_std,
                musiq_ts=excluded.musiq_ts
            """, (
                str(path), width, height, aspect, 0, None, None, None,
                exif_model, exif_lens, exif_dt, focal, float(sharp_val or 0.0),
                ai_provider, ai_raw_json, ai_top_label, ai_conf, ts,
                float(musiq_ava) if musiq_ava is not None else None,
                float(musiq_std) if musiq_std is not None else None,
                musiq_ts
            ))
            conn.commit()
        finally:
            conn.close()


    def _batch_worker(self, folder: str):
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
        paths = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
        total = len(paths)
        done = 0
        for p in paths:
            try:
                bgr = imread_unicode(str(p), cv2.IMREAD_COLOR)
                sharp_val, _ = assess_sharpness(str(p))
                # EXIF
                exif = {}
                try:
                    with Image.open(p) as im:
                        exif = get_exif_dict(im)
                except Exception:
                    pass
                # AI
                ai_pairs=[]; ai_top=None; ai_conf=None; provider=None
                if self.ai_enabled and bgr is not None:
                    try:
                        b64 = img_to_b64_for_api(str(p), 1200)
                        ai_pairs = vision_label_detect_b64(b64, self.ai_api_key, max_results=10)
                        if ai_pairs:
                            ai_top, ai_conf = ai_pairs[0]
                        provider="google_vision"
                    except Exception as e:
                        self.after(0, self.ui_log, f"[AI Error] {p.name}: {e}")
                # MUSIQ
                musiq = None; musiq_std = None
                if self.musiq_enabled and bgr is not None:
                    try:
                        rgb_small = cv2.cvtColor(resize_max_edge(bgr, 1200), cv2.COLOR_BGR2RGB)
                        musiq, musiq_std, err = musiq_score_ndarray_rgb(rgb_small)
                        if err:
                            self.after(0, self.ui_log, f"[MUSIQ] {p.name}: {err}")
                    except Exception as e:
                        self.after(0, self.ui_log, f"[MUSIQ Error] {p.name}: {e}")
                self._upsert_db_final(str(p), bgr, sharp_val, exif, ai_pairs, ai_top, ai_conf, provider,
                        musiq_ava=musiq, musiq_std=musiq_std)
            except Exception as e:
                self.after(0, self.ui_log, f"[Batch Error] {p}: {e}")
            finally:
                done += 1
                if done % 5 == 0 or done == total:
                    self.after(0, self.status.set, t(self,"progress_fmt", i=done, total=total))
        self.after(0, self.status.set, t(self,"batch_done", n=total))
        self.after(0, self.refresh_filters)
        self.after(0, lambda: (self.prog.stop(), setattr(self.prog, "mode", "determinate")))

# ---- Editor Dialog ----

class _VScrollFrame(ttk.Frame):
    """
    纵向/横向可滚动的容器。
    - lock_width=True  时：内部宽度锁定为可视宽度（只出现竖向滚动，适合表单）
    - lock_width=False 时：内部保持自然宽度，若超出则出现横向滚动条
    """
    def __init__(self, master, lock_width=True, **kw):
        super().__init__(master, **kw)
        self.lock_width = lock_width

        # 画布 + 滚动条
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical",
                                  command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal",
                                  command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set,
                              xscrollcommand=self.hbar.set)

        # 内部真正放控件的 Frame
        self.inner = ttk.Frame(self.canvas, style="Panel.TFrame")
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")

        # 布局：网格避免拉伸问题
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 同步滚动区域
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 鼠标滚轮：竖向滚动；按住 Shift 时横向滚动
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_wheel)

    def _on_inner_configure(self, _event=None):
        # 更新滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self.lock_width:
            # 锁定宽度：让内部宽度等于可视宽度（只出现竖向滚动）
            try:
                self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width())
            except Exception:
                pass

    def force_min_width(self, width: int):
            """强制内部窗口最小宽度，以便触发横向滚动条。"""
            self.lock_width = False                 # 确保不锁定宽度
            try:
                self.canvas.itemconfigure(self._win, width=width)
            except Exception:
                pass
            # 让画布重新计算滚动区域
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
    def _on_canvas_resize(self, _event=None):
        if self.lock_width:
            try:
                self.canvas.itemconfigure(self._win, width=self.canvas.winfo_width())
            except Exception:
                pass

    def _on_mouse_wheel(self, event):
        # Windows 正常是 event.delta=120 的倍数
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shift_wheel(self, event):
        # 按住 Shift + 滚轮 -> 横向滚动
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")



class EditDialog(tk.Toplevel):
    """
    非破坏式编辑器（后期模拟 ISO/曝光等）
    增强版 v3：
    - 高光/阴影/自然饱和度/暗角/锐化
    - 旋转（任意角度）/翻转/自动白平衡/居中裁剪
    - HSL 分色（按色相段：红/橙/黄/绿/青/蓝/紫，调 Hue/Sat/Luma）
    - 去雾（简化 Dark Channel）
    - 预设 保存/加载（JSON）
    - 撤销/重做（参数）
    - 并排对比（原图 vs 结果）
    """
    def __init__(self, app: App, bgr_src: np.ndarray, src_path: str):
        super().__init__(app)
        self.app = app
        self.title(t(self.app, "edit_title"))
        self.geometry("1260x780")
        self.configure(bg=app.palette["bg"])

        self.bgr_src = resize_max_edge(bgr_src, 2400)  # 原始可变（旋转/裁剪会更新它）
        self.preview_size = 900
        self._preview_tk = None
        self.src_path = src_path

        # 参数（放在 dict，方便保存/撤销）
        self.params = {
            "exposure": 0.0, "gamma": 1.0, "contrast": 1.0, "sat": 1.0,
            "temp": 0.0, "tint": 0.0, "clarity": 0.0, "denoise": 0.0,
            "grain": 0.0, "high": 0.0, "shadow": 0.0, "vibrance": 1.0,
            "vignette": 0.0, "sharpen": 0.0, "rotate": 0.0, "dehaze": 0.0,
            "hsl_band": "Red", "hsl_hue": 0.0, "hsl_sat": 1.0, "hsl_luma": 1.0,
            "show_orig": False, "side_by_side": False
        }
        self.undo_stack = []
        self.redo_stack = []

        # Tk 变量映射
        self.var = { k: tk.DoubleVar(value=v) if isinstance(v,(int,float)) else tk.BooleanVar(value=v) if isinstance(v,bool) else tk.StringVar(value=v)
                     for k,v in self.params.items() }

        # —— 左右可拖动分割 —— 
        # —— 不用分隔条；左右按比例自适应 —— 
        container = ttk.Frame(self, style="Panel.TFrame")
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=0)

        # 左右两列比例：左=1，右=2；可改成 1:1、2:3 等
        container.grid_columnconfigure(0, weight=1, minsize=420)  # 左栏最小宽度
        container.grid_columnconfigure(1, weight=2)               # 右侧更宽
        container.grid_rowconfigure(0, weight=1)

        # 左侧滚动容器（只纵向滚动，控件会横向填满）
        left_holder = ttk.Frame(container, style="Panel.TFrame")
        left_holder.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_wrap = _VScrollFrame(left_holder, padding=0, style="Panel.TFrame", lock_width=True)
        left_wrap.pack(fill=tk.BOTH, expand=True)

        # 右侧预览区
        right = ttk.Frame(container, padding=10, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")

        # 之后的控件都往 left/right 放
        left = left_wrap.inner

        # right 已经是上面创建的 ttk.Frame

        def add_slider(key, text, a, b, step=0.01):
            ttk.Label(left, text=text, style="Panel.TLabel").pack(anchor="w", pady=(6,0))
            s = ttk.Scale(left, from_=a, to=b, orient=tk.HORIZONTAL, variable=self.var[key], command=lambda *_: self._on_change())
            s.pack(fill=tk.X, pady=2)
            return s

        add_slider("exposure", t(self.app,"edit_exposure"), -2.0, 2.0)
        add_slider("gamma", t(self.app,"edit_gamma"), 0.5, 1.5)
        add_slider("contrast", t(self.app,"edit_contrast"), 0.5, 1.8)
        add_slider("sat", t(self.app,"edit_saturation"), 0.0, 2.0)
        add_slider("temp", t(self.app,"edit_temp"), -100, 100)
        add_slider("tint", t(self.app,"edit_tint"), -100, 100)
        add_slider("clarity", t(self.app,"edit_clarity"), 0.0, 2.0)
        add_slider("denoise", t(self.app,"edit_denoise"), 0.0, 20.0)
        add_slider("grain", t(self.app,"edit_grain"), 0.0, 30.0)
        add_slider("high", t(self.app,"edit_highlights"), -100, 100)
        add_slider("shadow", t(self.app,"edit_shadows"), -100, 100)
        add_slider("vibrance", t(self.app,"edit_vibrance"), 0.0, 2.0)
        add_slider("vignette", t(self.app,"edit_vignette"), 0.0, 80.0)
        add_slider("sharpen", t(self.app,"edit_sharpen"), 0.0, 2.0)
        add_slider("dehaze", t(self.app, "edit_dehaze"), 0.0, 1.0)
        add_slider("rotate", t(self.app, "edit_rotate"), -45, 45)

        # HSL 分段
        box = ttk.Frame(left, style="Panel.TFrame"); box.pack(fill=tk.X, pady=(8,0))
        ttk.Label(box, text=t(self.app, "edit_hsl_section"), style="Panel.TLabel").pack(anchor="w")
        self.var["hsl_band"] = tk.StringVar(value="Red")
        self.cmb = ttk.Combobox(box, state="readonly", values=["Red","Orange","Yellow","Green","Cyan","Blue","Magenta"], textvariable=self.var["hsl_band"])
        self.cmb.pack(fill=tk.X, pady=2)
        self.cmb.bind("<<ComboboxSelected>>", lambda e: self._on_change())

        add_slider("hsl_hue",  t(self.app, "edit_hue_shift"), -30, 30)
        add_slider("hsl_sat",  t(self.app, "edit_hsl_sat"),   0.0, 2.0)
        add_slider("hsl_luma", t(self.app, "edit_hsl_luma"),  0.5, 1.5)

        # 预设/撤销/翻转裁剪行
        row2 = ttk.Frame(left, style="Panel.TFrame"); row2.pack(fill=tk.X, pady=(8,0))
        ttk.Button(row2, text=t(self.app,"edit_iso_up"), command=self.preset_iso_up).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(row2, text=t(self.app,"edit_iso_down"), command=self.preset_iso_down).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        row3 = ttk.Frame(left, style="Panel.TFrame"); row3.pack(fill=tk.X, pady=(4,0))
        ttk.Button(row3, text=t(self.app,"edit_flip_h"), command=lambda:self._transform('flip_h')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(row3, text=t(self.app,"edit_flip_v"), command=lambda:self._transform('flip_v')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        row4 = ttk.Frame(left, style="Panel.TFrame"); row4.pack(fill=tk.X, pady=(4,0))
        ttk.Button(row4, text=t(self.app,"edit_rotate_left"), command=lambda:self._transform('rot_l')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(row4, text=t(self.app,"edit_rotate_right"), command=lambda:self._transform('rot_r')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        row5 = ttk.Frame(left, style="Panel.TFrame")
        row5.pack(fill=tk.X, pady=(4,0))
 
        row5 = ttk.Frame(left, style="Panel.TFrame"); row5.pack(fill=tk.X, pady=(4,0))
        ttk.Button(row5, text=t(self.app,"edit_auto_wb"), command=lambda:self._transform('awb')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(row5, text=t(self.app,"edit_crop_sq"), command=lambda:self._transform('crop_1_1')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(row5, text=t(self.app,"edit_crop_169"), command=lambda:self._transform('crop_16_9')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # --- row6: 预设 ---
        row6 = ttk.Frame(left, style="Panel.TFrame"); row6.pack(fill=tk.X, pady=(6,0))
        self.btn_preset_save = ttk.Button(row6, text=t(self.app, "edit_save_preset"), command=self.save_preset)
        self.btn_preset_save.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_preset_load = ttk.Button(row6, text=t(self.app, "edit_load_preset"), command=self.load_preset)
        self.btn_preset_load.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # --- row7: 撤销/重做 ---
        row7 = ttk.Frame(left, style="Panel.TFrame"); row7.pack(fill=tk.X, pady=(4,0))
        self.btn_undo = ttk.Button(row7, text=t(self.app, "edit_undo"), command=self.undo)
        self.btn_undo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_redo = ttk.Button(row7, text=t(self.app, "edit_redo"), command=self.redo)
        self.btn_redo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        row8 = ttk.Frame(left, style="Panel.TFrame"); row8.pack(fill=tk.X, pady=(6,0))
        ttk.Button(row8, text=t(self.app,"edit_reset"), command=self.reset_params).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(row8, text=t(self.app,"edit_saveas"), command=self.save_as).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Checkbutton(left, text=t(self.app,"edit_show_original"), variable=self.var["show_orig"], command=self._on_change).pack(anchor="w", pady=(6,0))
        ttk.Checkbutton(left, text=t(self.app, "edit_side_by_side"),
                        variable=self.var["side_by_side"], command=self._on_change)\
            .pack(anchor="w", pady=(2,0))

        # 画布
        self.lbl = tk.Label(right, bg=self.app.palette["panel"])
        self.lbl.pack(fill=tk.BOTH, expand=True)

        # 首次渲染
        self._push_undo()  # 初始状态入栈
        self.refresh_preview()

    # -------------- 状态栈 --------------
    def _push_undo(self):
        # 存入当前参数快照
        snap = {k:(v.get() if hasattr(v, 'get') else v) for k,v in self.var.items()}
        self.undo_stack.append(snap)
        # 清空 redo
        self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) <= 1: return
        cur = self.undo_stack.pop()
        self.redo_stack.append(cur)
        prev = self.undo_stack[-1]
        self._apply_snapshot(prev)
        self.refresh_preview()

    def redo(self):
        if not self.redo_stack: return
        snap = self.redo_stack.pop()
        self._apply_snapshot(snap)
        self.undo_stack.append(snap)
        self.refresh_preview()

    def _apply_snapshot(self, snap):
        for k,v in snap.items():
            if k in self.var:
                if isinstance(self.var[k], tk.BooleanVar):
                    self.var[k].set(bool(v))
                elif isinstance(self.var[k], tk.StringVar):
                    self.var[k].set(str(v))
                else:
                    self.var[k].set(float(v))

    # -------------- 事件 --------------
    def _on_change(self):
        self.refresh_preview()

    def _transform(self, op):
        # 对原始图进行几何/WB/裁剪等不可逆操作；操作后入栈
        if op == 'flip_h':
            self.bgr_src = cv2.flip(self.bgr_src, 1)
        elif op == 'flip_v':
            self.bgr_src = cv2.flip(self.bgr_src, 0)
        elif op == 'rot_l':
            self.bgr_src = cv2.rotate(self.bgr_src, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif op == 'rot_r':
            self.bgr_src = cv2.rotate(self.bgr_src, cv2.ROTATE_90_CLOCKWISE)
        elif op == 'awb':
            b,g,r = cv2.split(self.bgr_src.astype(np.float32))
            eps = 1e-6
            mb, mg, mr = b.mean()+eps, g.mean()+eps, r.mean()+eps
            k = (mb+mg+mr)/3.0
            b = np.clip(b * (k/mb), 0, 255)
            g = np.clip(g * (k/mg), 0, 255)
            r = np.clip(r * (k/mr), 0, 255)
            self.bgr_src = cv2.merge([b,g,r]).astype(np.uint8)
        elif op == 'crop_1_1':
            h, w = self.bgr_src.shape[:2]
            side = min(h, w)
            y0 = (h - side)//2; x0 = (w - side)//2
            self.bgr_src = self.bgr_src[y0:y0+side, x0:x0+side].copy()
        elif op == 'crop_16_9':
            h, w = self.bgr_src.shape[:2]
            target = 16/9.0
            cur = w/float(h)
            if cur > target:
                new_w = int(h*target); x0 = (w-new_w)//2
                self.bgr_src = self.bgr_src[:, x0:x0+new_w].copy()
            else:
                new_h = int(w/target); y0 = (h-new_h)//2
                self.bgr_src = self.bgr_src[y0:y0+new_h, :].copy()

        self._push_undo()
        self.refresh_preview()

    # -------------- 处理管线 --------------
    def _apply_hsl(self, img_bgr: np.ndarray) -> np.ndarray:
        band = self.var["hsl_band"].get()
        hue_shift = float(self.var["hsl_hue"].get())
        sat_scale = float(self.var["hsl_sat"].get())
        luma_scale = float(self.var["hsl_luma"].get())
        if abs(hue_shift) < 1e-3 and abs(sat_scale-1.0) < 1e-3 and abs(luma_scale-1.0) < 1e-3:
            return img_bgr
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        H,S,V = cv2.split(hsv)
        # hue in [0,179]; map color bands (approximate)
        ranges = {
            "Red": [(0,10),(170,179)],
            "Orange":[(10,25)],
            "Yellow":[(25,35)],
            "Green":[(35,85)],
            "Cyan":[(85,100)],
            "Blue":[(100,130)],
            "Magenta":[(130,170)]
        }
        mask = np.zeros_like(H, dtype=np.uint8)
        for (a,b) in ranges.get(band,[]):
            mask = cv2.bitwise_or(mask, cv2.inRange(H, a, b))
        # soft mask by saturation to avoid affecting grays
        soft = (mask/255.0) * (S/255.0)
        # apply hue shift (scale 0..179 => degrees ~ 2)
        H = (H + (hue_shift/2.0)*soft).clip(0,179)
        S = (S * (1.0 + (sat_scale-1.0)*soft)).clip(0,255)
        V = (V * (1.0 + (luma_scale-1.0)*soft)).clip(0,255)
        out = cv2.merge([H,S,V]).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)

    def _simple_dehaze(self, img_bgr: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0.001: return img_bgr
        I = img_bgr.astype(np.float32)/255.0
        # estimate dark channel
        min_rgb = I.min(axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
        dark = cv2.erode(min_rgb, kernel)
        # estimate atmosphere A as top 0.1% brightest in dark channel
        flat = dark.flatten()
        thresh = np.percentile(flat, 99.9)
        A = I.reshape(-1,3)[dark.reshape(-1) >= thresh].mean(axis=0)
        A = np.maximum(A, 0.1)
        # transmission
        w = 0.95
        t = 1 - w*dark
        t = np.clip(t, 0.1, 1.0)
        # recover
        J = (I - A) / t[...,None] + A
        J = np.clip(J, 0, 1)
        # blend by strength
        out = (I*(1-strength) + J*strength)
        return (out*255).astype(np.uint8)

    def apply_edits(self, bgr_in: np.ndarray) -> np.ndarray:
        # rotate arbitrary
        angle = float(self.var["rotate"].get())
        img = bgr_in.copy()
        if abs(angle) > 0.01:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        f = img.astype(np.float32) / 255.0
        if not bool(self.var["show_orig"].get()):
            # exposure / gamma / contrast
            ev = float(self.var["exposure"].get())
            f *= 2.0 ** ev

            gamma = max(0.01, float(self.var["gamma"].get()))
            f = np.power(np.clip(f, 0, 1), 1.0 / gamma)

            alpha = float(self.var["contrast"].get())
            f = np.clip((f - 0.5) * alpha + 0.5, 0, 1)

            # WB temp/tint
            temp = float(self.var["temp"].get()) / 100.0
            tint = float(self.var["tint"].get()) / 100.0
            r_gain = 1.0 + 0.6 * max(0.0, temp)
            b_gain = 1.0 + 0.6 * max(0.0, -temp)
            g_gain = 1.0 + 0.6 * max(0.0, -tint)
            m_gain = 1.0 + 0.6 * max(0.0, tint)
            b, g, r = cv2.split(f)
            r = np.clip(r * r_gain * m_gain, 0, 1)
            g = np.clip(g * g_gain, 0, 1)
            b = np.clip(b * b_gain * m_gain, 0, 1)
            f = cv2.merge([b, g, r])

            # HSV domain tweaks
            hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            sat = float(self.var["sat"].get())
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)

            # vibrance
            vib = float(self.var["vibrance"].get())
            if abs(vib - 1.0) > 1e-3:
                S = hsv[:, :, 1] / 255.0
                hsv[:, :, 1] = np.clip((S + (vib - 1.0) * (1.0 - S)) * 255.0, 0, 255)

            # highlights / shadows
            V = hsv[:, :, 2] / 255.0
            h_adj = float(self.var["high"].get()) / 100.0
            s_adj = float(self.var["shadow"].get()) / 100.0
            if abs(s_adj) > 1e-3:
                mask = (V < 0.5).astype(np.float32)
                V = np.clip(V + s_adj * (0.5 - V) * 2.0 * mask, 0, 1)
            if abs(h_adj) > 1e-3:
                mask = (V >= 0.5).astype(np.float32)
                V = np.clip(V + h_adj * (0.5 - V) * 2.0 * mask, 0, 1)
            hsv[:, :, 2] = V * 255.0
            f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

            # clarity
            strength = float(self.var["clarity"].get())
            if strength > 0.001:
                blurred = cv2.GaussianBlur(f, (0, 0), sigmaX=2.0)
                f = np.clip(f + (f - blurred) * (strength * 1.2), 0, 1)

            # denoise
            den = float(self.var["denoise"].get())
            if den > 0.01:
                tmp8 = (f * 255).astype(np.uint8)
                tmp8 = cv2.fastNlMeansDenoisingColored(tmp8, None, h=den, hColor=den,
                                                    templateWindowSize=7, searchWindowSize=21)
                f = tmp8.astype(np.float32) / 255.0

            # sharpen
            sh = float(self.var["sharpen"].get())
            if sh > 0.001:
                blurred = cv2.GaussianBlur(f, (0, 0), sigmaX=1.0)
                f = np.clip(f + (f - blurred) * sh, 0, 1)

            # grain
            grain = float(self.var["grain"].get())
            if grain > 0.5:
                noise = np.random.normal(0, grain / 255.0, f.shape).astype(np.float32)
                f = np.clip(f + noise, 0, 1)

            # vignette
            vig = float(self.var["vignette"].get())
            if vig > 0.1:
                h, w = f.shape[:2]
                y, x = np.ogrid[:h, :w]
                cy, cx = h / 2.0, w / 2.0
                r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                r = r / r.max()
                mask = 1.0 - (vig / 80.0) * (r ** 2)
                mask = np.clip(mask, 0.2, 1.0).astype(np.float32)
                f *= mask[..., None]

            out = (np.clip(f, 0, 1) * 255).astype(np.uint8)

            # HSL 分色
            out = self._apply_hsl(out)

            # 去雾
            deh = float(self.var["dehaze"].get())
            if deh > 0.01:
                out = self._simple_dehaze(out, deh)
        else:
            out = img

        return out


    def refresh_preview(self):
        try:
            small = resize_max_edge(self.bgr_src, self.preview_size)
            out = self.apply_edits(small)
            if bool(self.var["side_by_side"].get()):
                rgb1 = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                rgb2 = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                # pad heights
                h1,w1 = rgb1.shape[:2]; h2,w2 = rgb2.shape[:2]
                H = max(h1,h2)
                pad1 = np.zeros((H-h1, w1, 3), dtype=rgb1.dtype) if H>h1 else None
                pad2 = np.zeros((H-h2, w2, 3), dtype=rgb2.dtype) if H>h2 else None
                if pad1 is not None: rgb1 = np.vstack([rgb1,pad1])
                if pad2 is not None: rgb2 = np.vstack([rgb2,pad2])
                rgb = np.hstack([rgb1, rgb2])
            else:
                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            self._preview_tk = ImageTk.PhotoImage(im)
            self.lbl.configure(image=self._preview_tk)
        except Exception as e:
            self.lbl.configure(text=str(e))

    def reset_params(self):
        for k,v in self.params.items():
            if isinstance(self.var[k], tk.BooleanVar):
                self.var[k].set(False if k in ("show_orig","side_by_side") else v if isinstance(v,bool) else False)
            elif isinstance(self.var[k], tk.StringVar):
                self.var[k].set("Red" if k=="hsl_band" else str(v))
            else:
                self.var[k].set(v)
        self._push_undo()
        self.refresh_preview()

    def preset_iso_up(self):
        self.var["exposure"].set(0.6); self.var["gamma"].set(1.05); self.var["contrast"].set(1.1)
        self.var["sat"].set(1.05); self.var["clarity"].set(0.4); self.var["denoise"].set(2.0)
        self.var["grain"].set(10.0); self.var["high"].set(-10.0); self.var["shadow"].set(15.0)
        self.var["sharpen"].set(0.3)
        self._push_undo(); self.refresh_preview()

    def preset_iso_down(self):
        self.var["exposure"].set(-0.3); self.var["gamma"].set(0.95); self.var["contrast"].set(0.95)
        self.var["sat"].set(0.95); self.var["clarity"].set(0.1); self.var["denoise"].set(10.0)
        self.var["grain"].set(0.0); self.var["high"].set(10.0); self.var["shadow"].set(-10.0)
        self.var["sharpen"].set(0.0)
        self._push_undo(); self.refresh_preview()

    def save_as(self):
        path = filedialog.asksaveasfilename(
            title=t(self.app,"edit_saveas"),
            defaultextension=".jpg",
            filetypes=[("JPEG","*.jpg;*.jpeg"), ("PNG","*.png"), ("TIFF","*.tif;*.tiff")]
        )
        if not path: return
        try:
            full = self.apply_edits(self.bgr_src)
            ext = Path(path).suffix.lower()
            if ext in [".jpg",".jpeg"]:
                cv2.imwrite(path, full, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            else:
                cv2.imwrite(path, full)
            messagebox.showinfo("Saved", f"Saved to:\\n{path}")
            try:
                bgr = full
                gray = cv2.cvtColor(resize_max_edge(full, 1200), cv2.COLOR_BGR2GRAY)
                sharp_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                exif = {}
                self.app._upsert_db_final(path, bgr, sharp_val, exif,
                                          ai_pairs=None, ai_top_label=None, ai_conf=None, ai_provider=None)
                self.app.refresh_filters()
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Save As", str(e))

    # -------------- 预设 --------------
    def _collect_params(self):
        snap = {k:(v.get() if hasattr(v,'get') else v) for k,v in self.var.items()}
        # 仅保存可调参数，不保存 show flags
        keep = ["exposure","gamma","contrast","sat","temp","tint","clarity","denoise","grain",
                "high","shadow","vibrance","vignette","sharpen","rotate","dehaze",
                "hsl_band","hsl_hue","hsl_sat","hsl_luma"]
        return {k:snap[k] for k in keep}

    def save_preset(self):
        preset = self._collect_params()
        path = filedialog.asksaveasfilename(title="保存预设为 JSON", defaultextension=".json",
                                            filetypes=[("JSON",".json")])
        if not path: return
        with open(path,"w",encoding="utf-8") as f:
            json.dump(preset, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("预设", f"已保存：\\n{path}")

    def load_preset(self):
        path = filedialog.askopenfilename(title="加载预设 JSON", filetypes=[("JSON",".json")])
        if not path: return
        try:
            preset = json.load(open(path, "r", encoding="utf-8"))
            for k,v in preset.items():
                if k in self.var:
                    if isinstance(self.var[k], tk.StringVar): self.var[k].set(str(v))
                    else: self.var[k].set(float(v))
            self._push_undo()
            self.refresh_preview()
        except Exception as e:
            messagebox.showerror("预设", str(e))

# ---- main ----
if __name__ == "__main__":
    App().mainloop()
