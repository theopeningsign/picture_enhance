"""
인터랙티브 SAM 편집 창
- 박스 드래그: SAM 자동 영역 선택
- 브러시/지우기: 마스크 직접 드로잉
- 적용: 16:9 자동 크롭 + 컬러/흑백 합성 후 메인 앱에 전달
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageTk, ImageDraw

if TYPE_CHECKING:
    from app import App
    from sam_manager import SAMManager

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    import Tkinter as tk  # type: ignore

# 16:9 크롭 기본 패딩 비율
DEFAULT_PADDING = 0.35
# 캔버스 최대 크기
CANVAS_MAX_W = 860
CANVAS_MAX_H = 600
# 마스크 오버레이 — 형광 라임 그린 (밝은 낮 사진에서도 잘 보임)
OVERLAY_COLOR = (57, 255, 20)
OVERLAY_ALPHA = 130
# 포인트 반지름 (캔버스 픽셀)
POINT_RADIUS = 7
# 크롭 핸들 반경 (캔버스 픽셀)
HANDLE_R = 7


def compute_crop_rect_16_9(
    mask: np.ndarray, img_w: int, img_h: int, padding: float = DEFAULT_PADDING
) -> tuple[int, int, int, int]:
    """
    마스크 bounding box를 기준으로 16:9 자동 크롭 사각형 계산.
    Returns (x1, y1, x2, y2) in original image coordinates.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, img_w, img_h)

    y1m, y2m = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1m, x2m = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    cx = (x1m + x2m) / 2.0
    cy = (y1m + y2m) / 2.0
    sw = x2m - x1m
    sh = y2m - y1m

    # padding 적용
    half_w = sw * (1.0 + padding) / 2.0
    half_h = sh * (1.0 + padding) / 2.0

    # 16:9 비율 맞추기 (넓은 쪽 기준 확장)
    target_ratio = 16.0 / 9.0
    if half_w / max(half_h, 1) < target_ratio:
        half_w = half_h * target_ratio
    else:
        half_h = half_w / target_ratio

    x1c = max(0, int(cx - half_w))
    y1c = max(0, int(cy - half_h))
    x2c = min(img_w, int(cx + half_w))
    y2c = min(img_h, int(cy + half_h))

    return (x1c, y1c, x2c, y2c)


def apply_postprocess(
    pil_img: Image.Image,
    mask_full: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    wm_positions=None,  # list of (rel_x, rel_y) 0~1, None이면 기본 배치
    wm_size: float = 0.20,  # 로고 너비 (이미지 너비 대비)
) -> Image.Image:
    """
    크롭 + 컬러/흑백 합성.
    간판(mask=True) → 컬러, 배경 → 흑백 + 80% 밝기
    """
    x1, y1, x2, y2 = crop_rect
    cropped = pil_img.crop((x1, y1, x2, y2))

    # 마스크를 크롭 영역에 맞게 슬라이싱
    mask_crop = mask_full[y1:y2, x1:x2]

    # 배경: 흑백 + 20% 어둡게 (원래대로)
    gray = cropped.convert("L").convert("RGB")
    gray = ImageEnhance.Brightness(gray).enhance(0.8)

    # 간판(컬러): 채도 + 대비 + 선명도 강화 → 색상이 팍 튀어나오게
    color_img = cropped.convert("RGB")
    color_img = ImageEnhance.Color(color_img).enhance(1.5)      # 채도 50% 업
    color_img = ImageEnhance.Contrast(color_img).enhance(1.2)   # 대비 20% 업
    color_img = ImageEnhance.Sharpness(color_img).enhance(1.3)  # 선명도 30% 업

    color_arr = np.array(color_img)
    gray_arr = np.array(gray)

    # mask 크기 불일치 방지
    h, w = color_arr.shape[:2]
    if mask_crop.shape[:2] != (h, w):
        mask_img = Image.fromarray(mask_crop.astype(np.uint8) * 255)
        mask_img = mask_img.resize((w, h), Image.NEAREST)
        mask_crop = np.array(mask_img) > 128

    result_arr = np.where(mask_crop[:, :, None], color_arr, gray_arr)
    result = Image.fromarray(result_arr.astype(np.uint8), "RGB")

    # 워터마크: 중앙 하단 단일 배치 (기본), 드래그로 위치 조절 가능
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        W, H = result.width, result.height

        _lr = Image.open(logo_path).convert("RGBA")
        n_pos = len(wm_positions) if wm_positions else 1
        logo_tile_w = max(int(W * wm_size), 40)
        logo_tile_h = int(_lr.height * (logo_tile_w / _lr.width))

        # supersampling: 2배 크기로 렌더 후 축소 → 선명한 엣지
        scale2 = 2
        _lr2 = _lr.resize((logo_tile_w * scale2, logo_tile_h * scale2), Image.LANCZOS)
        _la2 = _lr2.split()[3]

        # 다크 아웃라인: 알파를 팽창+블러 → 검정 레이어로 깔아서 흰 로고 테두리 강조
        from PIL import ImageFilter as _IFw
        _outline_alpha = _la2.filter(_IFw.MaxFilter(size=5))
        _outline_alpha = _outline_alpha.filter(_IFw.GaussianBlur(radius=3))
        _dark = Image.new("RGBA", _lr2.size, (0, 0, 0, 255))
        _dark.putalpha(_outline_alpha.point(lambda x: int(x * 0.55)))

        # 흰색 로고
        _wl2 = Image.new("RGBA", _lr2.size, (255, 255, 255, 255))
        _wl2.putalpha(_la2.point(lambda x: int(x * 0.92)))

        # 다크 아웃라인 위에 흰 로고 합성
        _combined = Image.alpha_composite(_dark, _wl2)

        # 단일 배치는 기울기 없이 정방향, 여러 개면 30도
        angle = 0 if n_pos == 1 else 30
        _rotated2 = _combined.rotate(angle, expand=True, resample=Image.BICUBIC)
        txt_tile = _rotated2.resize(
            (_rotated2.width // scale2, _rotated2.height // scale2), Image.LANCZOS)
        tile_w, tile_h = txt_tile.size

        if wm_positions is not None:
            abs_positions = [(int(rx * W), int(ry * H)) for rx, ry in wm_positions]
        else:
            abs_positions = [(W // 2, int(H * 0.88))]  # 기본: 중앙 하단

        tile_layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
        for cx, cy in abs_positions:
            px = cx - tile_w // 2
            py = cy - tile_h // 2
            tile_layer.paste(txt_tile, (px, py), mask=txt_tile)

        result_rgba = result.convert("RGBA")
        result_rgba = Image.alpha_composite(result_rgba, tile_layer)
        result = result_rgba.convert("RGB")

    return result


class EditWindow(tk.Toplevel):
    """SAM 인터랙티브 편집 창"""

    def __init__(self, parent_app: "App", image_path: Path, sam_manager: "SAMManager"):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.image_path = image_path
        self.sam_manager = sam_manager

        # 원본 이미지 로드 (EXIF orientation 자동 보정)
        try:
            _raw = Image.open(image_path)
            # EXIF orientation 적용 (폰 세로 사진 90도 회전 문제 방지)
            try:
                from PIL import ImageOps
                _raw = ImageOps.exif_transpose(_raw)
            except Exception:
                pass
            self.orig_img = _raw.convert("RGB")
        except Exception as e:
            messagebox.showerror("오류", f"이미지를 열 수 없습니다:\n{e}", parent=self)
            self.destroy()
            return

        self.orig_w, self.orig_h = self.orig_img.size
        self.img_np = np.array(self.orig_img)  # RGB numpy

        # ── 기본 상태 ──
        self.mask: Optional[np.ndarray] = None           # (H, W) bool
        self.mode_var = tk.StringVar(value="box")        # "box" | "brush" | "erase"
        self.brush_size = tk.IntVar(value=20)
        self.show_crop_preview = tk.BooleanVar(value=False)
        self._sam_running = False
        self._last_brush_xy: Optional[tuple[int, int]] = None
        self._box_start: Optional[tuple[int, int]] = None
        self._box_rect_id = None

        # ── 브러시 커서 ──
        self._mouse_cx: Optional[int] = None
        self._mouse_cy: Optional[int] = None

        # ── Shift+드래그 직선 모드 ──
        self._line_mode = False
        self._line_start_canvas: Optional[tuple[int, int]] = None
        self._line_preview_id = None

        # ── 전용 직선 모드 ──
        self._line_erase: bool = False

        # ── 크롭 핸들 드래그 ──
        self._custom_crop: Optional[tuple[int, int, int, int]] = None
        self._dragging_handle: Optional[str] = None   # 'nw'|'ne'|'sw'|'se'|'move'
        self._drag_start_canvas: Optional[tuple[int, int]] = None
        self._drag_start_crop: Optional[tuple] = None

        # ── 워터마크 미리보기 ──
        self.show_wm_preview = tk.BooleanVar(value=False)
        self._wm_positions: list = [
            (0.5, 0.88),  # 중앙 하단
        ]
        self._wm_size: float = 0.20               # 로고 너비 (이미지 너비 대비 비율)
        self._dragging_wm_idx: Optional[int] = None
        self._wm_drag_start_canvas: Optional[tuple] = None
        self._wm_drag_start_pos: Optional[tuple] = None
        self._wm_resizing: bool = False
        self._wm_resize_start_canvas: Optional[tuple] = None
        self._wm_resize_start_size: float = 0.20
        self._wm_tile_cache: Optional[Image.Image] = None
        self._wm_tile_cache_size: float = -1.0    # 캐시된 size 값

        # ── 되돌리기 히스토리 (최대 20단계) ──
        self._mask_history: list = []

        # 화면 해상도 기준으로 캔버스 크기 동적 계산
        # (툴바 130px + 우측패널 260px 제외, 여백 40px)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        canvas_max_w = screen_w - 260 - 40
        canvas_max_h = screen_h - 130 - 60

        # 스케일 계산
        scale_x = canvas_max_w / self.orig_w
        scale_y = canvas_max_h / self.orig_h
        self.scale = min(scale_x, scale_y, 1.0)
        self.canvas_w = int(self.orig_w * self.scale)
        self.canvas_h = int(self.orig_h * self.scale)

        # 표시용 이미지
        self.display_img = self.orig_img.resize(
            (self.canvas_w, self.canvas_h), Image.LANCZOS
        )

        self._build_ui()

        win_w = max(self.canvas_w + 260, 920)
        win_h = self.canvas_h + 130
        self.resizable(True, True)
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(920, 480)
        self.title(f"SAM 편집 — {image_path.name}")
        self.lift()
        self.focus_force()

        self.after(50, self._render)

        if sam_manager.is_loaded():
            threading.Thread(target=self._set_sam_image, daemon=True).start()
        else:
            self._update_status("SAM 미로딩 — 브러시만 사용 가능")

        self.grab_set()

    # ─────────────────────────── UI 빌드 ────────────────────────────

    def _build_ui(self):
        # ── 툴바 ──
        toolbar = tk.Frame(self, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)

        tk.Label(toolbar, text="모드:").pack(side=tk.LEFT, padx=(4, 2))
        for label, val in [("📦 박스", "box"), ("🖌 칠하기", "brush"), ("🧹 지우기", "erase"), ("📏 직선", "line")]:
            tk.Radiobutton(
                toolbar, text=label, variable=self.mode_var, value=val,
                indicatoron=False, width=7, pady=2,
                command=self._update_brush_cursor,
            ).pack(side=tk.LEFT, padx=1)
        tk.Label(toolbar, text="  (직선: 우클릭=지우기)", fg="gray", font=("", 8)).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(toolbar, text="브러시 크기:").pack(side=tk.LEFT, padx=(8, 0))
        tk.Scale(
            toolbar, variable=self.brush_size,
            from_=5, to=80, orient=tk.HORIZONTAL, length=90, showvalue=True
        ).pack(side=tk.LEFT)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Checkbutton(
            toolbar, text="크롭 미리보기", variable=self.show_crop_preview,
            command=self._render
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            toolbar, text="워터마크 미리보기", variable=self.show_wm_preview,
            command=self._render
        ).pack(side=tk.LEFT)

        tk.Button(toolbar, text="초기화", command=self._on_reset, width=6).pack(
            side=tk.RIGHT, padx=4
        )
        tk.Button(toolbar, text="↩ 되돌리기", command=self._on_undo, width=9).pack(
            side=tk.RIGHT, padx=2
        )

        # ── 본문 (캔버스 + 사이드패널) ──
        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=4)

        # 캔버스
        self.canvas = tk.Canvas(
            body, width=self.canvas_w, height=self.canvas_h,
            cursor="crosshair", bd=0, highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>",        self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>",        self._on_right_press)
        self.canvas.bind("<B3-Motion>",       self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_release)
        # 브러시 커서 이벤트
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>",  self._on_mouse_leave)
        self.brush_size.trace_add("write", lambda *_: self._update_brush_cursor())
        # 되돌리기 단축키
        self.bind("<Control-z>", self._on_undo)

        # 사이드패널
        side = tk.Frame(body, width=160)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0))
        side.pack_propagate(False)

        tk.Label(side, text="사용법", font=("", 9, "bold")).pack(anchor=tk.W)
        tk.Label(side, text="① 박스로 간판 드래그", justify=tk.LEFT).pack(anchor=tk.W)
        tk.Label(side, text="② 칠하기/지우기로\n  세부 조정", justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        tk.Label(side, text="③ 경계 정제로\n  엣지 자동 스냅", justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        self.btn_refine = tk.Button(
            side, text="✨ 경계 정제", command=self._on_refine_edges,
            bg="#4CAF50", fg="white", width=12, font=("", 9, "bold")
        )
        self.btn_refine.pack(anchor=tk.W, pady=(0, 4))

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── 적용 / 취소 버튼 (우측 패널) ──
        tk.Button(
            side, text="✔ 적용 (16:9 크롭 저장)", command=self._on_apply,
            bg="#2196F3", fg="white", font=("", 9, "bold"), width=18
        ).pack(anchor=tk.W, pady=(0, 4))
        tk.Button(
            side, text="✖ 취소", command=self.destroy, width=18
        ).pack(anchor=tk.W)

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        self.lbl_status = tk.Label(side, text="박스 모드로 간판을 드래그하세요", wraplength=140,
                                   justify=tk.LEFT, fg="gray")
        self.lbl_status.pack(anchor=tk.W)

        self.lbl_device = tk.Label(
            side,
            text=self.sam_manager.get_status(),
            fg="blue", font=("", 8)
        )
        self.lbl_device.pack(anchor=tk.W, pady=(4, 0))

    # ─────────────────────────── 렌더링 ─────────────────────────────

    def _render(self):
        """캔버스 전체 재렌더링"""
        composite = self.display_img.copy().convert("RGBA")

        # 마스크 오버레이 (numpy 배열 방식 — 빠름)
        if self.mask is not None:
            mask_small = self._resize_mask_to_canvas(self.mask)
            overlay_arr = np.zeros((self.canvas_h, self.canvas_w, 4), dtype=np.uint8)
            overlay_arr[mask_small] = [*OVERLAY_COLOR, OVERLAY_ALPHA]
            overlay = Image.fromarray(overlay_arr, "RGBA")
            composite = Image.alpha_composite(composite, overlay)

            # 크롭 미리보기
            if self.show_crop_preview.get():
                crop = self._get_crop_rect()
                if crop:
                    sx1 = int(crop[0] * self.scale)
                    sy1 = int(crop[1] * self.scale)
                    sx2 = int(crop[2] * self.scale)
                    sy2 = int(crop[3] * self.scale)
                    draw2 = ImageDraw.Draw(composite)
                    # 노란 테두리
                    for t in range(3):
                        draw2.rectangle(
                            [sx1 + t, sy1 + t, sx2 - t, sy2 - t],
                            outline=(255, 200, 0, 220)
                        )
                    # 코너 핸들 (흰 원 + 검정 테두리)
                    for hx, hy in [(sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)]:
                        draw2.ellipse(
                            [hx - HANDLE_R, hy - HANDLE_R, hx + HANDLE_R, hy + HANDLE_R],
                            fill=(255, 255, 255, 230), outline=(0, 0, 0, 255)
                        )

        # 워터마크 미리보기
        if self.show_wm_preview.get():
            crop = self._get_crop_rect() or (0, 0, self.orig_w, self.orig_h)
            x1c, y1c, x2c, y2c = crop
            cw, ch = x2c - x1c, y2c - y1c
            wm_tile = self._get_wm_tile()
            draw2 = ImageDraw.Draw(composite)
            for i, (rx, ry) in enumerate(self._wm_positions):
                # crop 좌표 → canvas 좌표
                cx = int((x1c + rx * cw) * self.scale)
                cy = int((y1c + ry * ch) * self.scale)
                tw_c = th_c = 0
                if wm_tile is not None:
                    tw_c = int(wm_tile.width * self.scale)
                    th_c = int(wm_tile.height * self.scale)
                    if tw_c > 0 and th_c > 0:
                        t_small = wm_tile.resize((tw_c, th_c), Image.LANCZOS)
                        composite.paste(t_small, (cx - tw_c // 2, cy - th_c // 2), mask=t_small)
                # 이동 핸들 (주황 원, 중앙)
                r = HANDLE_R + 3
                draw2.ellipse([cx - r, cy - r, cx + r, cy + r],
                              fill=(255, 140, 0, 220), outline=(255, 255, 255, 255))
                # 리사이즈 핸들 (초록 사각형, 우하단)
                if tw_c > 0 and th_c > 0:
                    rhx = cx + tw_c // 2
                    rhy = cy + th_c // 2
                    rs = HANDLE_R + 2
                    draw2.rectangle([rhx - rs, rhy - rs, rhx + rs, rhy + rs],
                                    fill=(50, 220, 50, 220), outline=(255, 255, 255, 255))

                # 드래그 중 중앙 도움선: 크롭 범위 정중앙에 가까우면 십자선 표시
                if self._dragging_wm_idx == i:
                    crop_cx = int((x1c + cw * 0.5) * self.scale)
                    crop_cy = int((y1c + ch * 0.5) * self.scale)
                    snap_px = int(cw * self.scale * 0.06)  # 6% 이내면 가이드 표시
                    # 세로 중앙선 (수직)
                    if abs(cx - crop_cx) < snap_px:
                        draw2.line([(crop_cx, 0), (crop_cx, self.canvas_h)],
                                   fill=(255, 220, 0, 180), width=1)
                    # 가로 중앙선 (수평)
                    if abs(cy - crop_cy) < snap_px:
                        draw2.line([(0, crop_cy), (self.canvas_w, crop_cy)],
                                   fill=(255, 220, 0, 180), width=1)

        self._tk_img = ImageTk.PhotoImage(composite.convert("RGB"))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        # 브러시 커서 복원 (delete("all") 후)
        self._update_brush_cursor()

    def _resize_mask_to_canvas(self, mask: np.ndarray) -> np.ndarray:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((self.canvas_w, self.canvas_h), Image.NEAREST)
        return np.array(mask_img) > 128

    # ─────────────────────────── 좌표 변환 ──────────────────────────

    def _canvas_to_img(self, cx: int, cy: int) -> tuple[int, int]:
        ix = int(cx / self.scale)
        iy = int(cy / self.scale)
        return (
            max(0, min(self.orig_w - 1, ix)),
            max(0, min(self.orig_h - 1, iy)),
        )

    def _img_to_canvas(self, ix: int, iy: int) -> tuple[int, int]:
        return int(ix * self.scale), int(iy * self.scale)

    # ─────────────────────────── 되돌리기 ──────────────────────────

    def _push_history(self):
        """현재 마스크 상태를 히스토리에 저장 (최대 20단계)"""
        self._mask_history.append(self.mask.copy() if self.mask is not None else None)
        if len(self._mask_history) > 20:
            self._mask_history.pop(0)

    def _on_undo(self, event=None):
        """한 단계 되돌리기 (Ctrl+Z 또는 버튼) — 키 반복 방지"""
        import time
        now = time.time()
        if hasattr(self, '_last_undo_time') and now - self._last_undo_time < 0.3:
            return  # 300ms 이내 중복 호출 무시 (키 반복 방지)
        self._last_undo_time = now
        if not self._mask_history:
            self._update_status("되돌릴 내용 없음")
            return
        self.mask = self._mask_history.pop()
        self._render()
        remaining = len(self._mask_history)
        self._update_status(f"되돌리기 완료 (잔여 {remaining}단계)")

    # ─────────────────────────── 브러시 커서 ────────────────────────

    def _on_mouse_move(self, event):
        self._mouse_cx = event.x
        self._mouse_cy = event.y
        self._update_brush_cursor()

    def _on_mouse_leave(self, event):
        self._mouse_cx = None
        self._mouse_cy = None
        self.canvas.delete("brush_cursor")

    def _update_brush_cursor(self):
        self.canvas.delete("brush_cursor")
        mode = self.mode_var.get()
        if mode not in ("brush", "erase", "line") or self._mouse_cx is None:
            return
        r = self.brush_size.get()
        x, y = self._mouse_cx, self._mouse_cy
        color = "#ff4444" if mode == "erase" else "#44ff44"
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline=color, width=1, tags="brush_cursor"
        )

    # ─────────────────────────── 크롭 헬퍼 ─────────────────────────

    def _get_crop_rect(self) -> Optional[tuple[int, int, int, int]]:
        """custom_crop이 있으면 반환, 없으면 마스크 기반 자동 계산"""
        if self._custom_crop is not None:
            return self._custom_crop
        if self.mask is not None and self.mask.any():
            return compute_crop_rect_16_9(self.mask, self.orig_w, self.orig_h)
        return None

    def _get_wm_tile(self) -> Optional[Image.Image]:
        """워터마크 미리보기용 로고 타일 (size 변경시 캐시 갱신)"""
        if self._wm_tile_cache is not None and self._wm_tile_cache_size == self._wm_size:
            return self._wm_tile_cache
        logo_path = Path(__file__).parent / "logo.png"
        if not logo_path.exists():
            return None
        _lr = Image.open(logo_path).convert("RGBA")
        n_pos = len(self._wm_positions)
        logo_tile_w = max(int(self.orig_w * self._wm_size), 40)
        logo_tile_h = int(_lr.height * (logo_tile_w / _lr.width))
        _lr2 = _lr.resize((logo_tile_w * 2, logo_tile_h * 2), Image.LANCZOS)
        _la2 = _lr2.split()[3]
        from PIL import ImageFilter as _IFw2
        _outline_a = _la2.filter(_IFw2.MaxFilter(size=5)).filter(_IFw2.GaussianBlur(radius=3))
        _dark = Image.new("RGBA", _lr2.size, (0, 0, 0, 255))
        _dark.putalpha(_outline_a.point(lambda x: int(x * 0.55)))
        _wl2 = Image.new("RGBA", _lr2.size, (255, 255, 255, 255))
        _wl2.putalpha(_la2.point(lambda x: int(x * 0.92)))
        _combined = Image.alpha_composite(_dark, _wl2)
        angle = 0 if n_pos == 1 else 30
        rotated = _combined.rotate(angle, expand=True, resample=Image.BICUBIC)
        tile = rotated.resize((rotated.width // 2, rotated.height // 2), Image.LANCZOS)
        self._wm_tile_cache = tile
        self._wm_tile_cache_size = self._wm_size
        return tile

    def _hit_test_crop(self, cx: int, cy: int, crop: tuple) -> Optional[str]:
        """캔버스 좌표 (cx, cy)가 핸들/내부 중 어디 있는지 반환"""
        ix1, iy1, ix2, iy2 = crop
        sx1, sy1 = self._img_to_canvas(ix1, iy1)
        sx2, sy2 = self._img_to_canvas(ix2, iy2)
        # 코너 핸들 우선
        for name, (hx, hy) in [
            ("nw", (sx1, sy1)), ("ne", (sx2, sy1)),
            ("sw", (sx1, sy2)), ("se", (sx2, sy2)),
        ]:
            if abs(cx - hx) <= HANDLE_R + 3 and abs(cy - hy) <= HANDLE_R + 3:
                return name
        # 내부 → 이동
        if sx1 < cx < sx2 and sy1 < cy < sy2:
            return "move"
        return None

    # ─────────────────────────── 이벤트 핸들러 ──────────────────────

    def _on_press(self, event):
        # ① 워터마크 핸들 hit test
        if self.show_wm_preview.get():
            crop = self._get_crop_rect() or (0, 0, self.orig_w, self.orig_h)
            x1c, y1c, x2c, y2c = crop
            cw, ch = x2c - x1c, y2c - y1c
            wm_tile = self._get_wm_tile()
            for i, (rx, ry) in enumerate(self._wm_positions):
                cx = int((x1c + rx * cw) * self.scale)
                cy = int((y1c + ry * ch) * self.scale)
                # 리사이즈 핸들 (우하단) 먼저 체크
                if wm_tile is not None:
                    tw_c = int(wm_tile.width * self.scale)
                    th_c = int(wm_tile.height * self.scale)
                    rhx = cx + tw_c // 2
                    rhy = cy + th_c // 2
                    if abs(event.x - rhx) <= HANDLE_R + 5 and abs(event.y - rhy) <= HANDLE_R + 5:
                        self._wm_resizing = True
                        self._wm_resize_start_canvas = (event.x, event.y)
                        self._wm_resize_start_size = self._wm_size
                        return
                # 이동 핸들 (중앙)
                if abs(event.x - cx) <= HANDLE_R + 6 and abs(event.y - cy) <= HANDLE_R + 6:
                    self._dragging_wm_idx = i
                    self._wm_drag_start_canvas = (event.x, event.y)
                    self._wm_drag_start_pos = self._wm_positions[i]
                    return

        # ② 크롭 핸들 hit test
        if self.show_crop_preview.get():
            crop = self._get_crop_rect()
            if crop:
                if self._custom_crop is None:
                    self._custom_crop = crop
                handle = self._hit_test_crop(event.x, event.y, crop)
                if handle:
                    self._dragging_handle = handle
                    self._drag_start_canvas = (event.x, event.y)
                    self._drag_start_crop = self._custom_crop
                    return  # 브러시/박스 차단

        mode = self.mode_var.get()

        # ② 박스 모드
        if mode == "box":
            self._box_start = (event.x, event.y)
            return

        # ③ 전용 직선 모드
        if mode == "line":
            self._push_history()
            self._line_start_canvas = (event.x, event.y)
            self._line_erase = False
            return

        # ④ Shift+드래그 직선 모드 진입
        if event.state & 0x0001:
            self._push_history()
            self._line_mode = True
            self._line_start_canvas = (event.x, event.y)
            return

        # ⑤ 일반 브러시/지우기
        self._line_mode = False
        erase = (mode == "erase")
        self._push_history()
        self._brush_paint(event, erase=erase)

    def _on_drag(self, event):
        # ① 워터마크 리사이즈 드래그
        if self._wm_resizing and self._wm_resize_start_canvas is not None:
            dx = event.x - self._wm_resize_start_canvas[0]
            dy = event.y - self._wm_resize_start_canvas[1]
            delta_rel = (dx + dy) / 2 / (self.orig_w * self.scale)
            new_size = max(0.05, min(0.60, self._wm_resize_start_size + delta_rel))
            self._wm_size = new_size
            self._wm_tile_cache = None  # 캐시 무효화
            self._render()
            return

        # ① 워터마크 이동 드래그
        if self._dragging_wm_idx is not None and self._wm_drag_start_canvas is not None:
            crop = self._get_crop_rect() or (0, 0, self.orig_w, self.orig_h)
            x1c, y1c, x2c, y2c = crop
            cw, ch = x2c - x1c, y2c - y1c
            orx, ory = self._wm_drag_start_pos
            dx_rel = (event.x - self._wm_drag_start_canvas[0]) / (cw * self.scale)
            dy_rel = (event.y - self._wm_drag_start_canvas[1]) / (ch * self.scale)
            new_rx = max(0.0, min(1.0, orx + dx_rel))
            new_ry = max(0.0, min(1.0, ory + dy_rel))
            pos = list(self._wm_positions)
            pos[self._dragging_wm_idx] = (new_rx, new_ry)
            self._wm_positions = pos
            self._render()
            return

        # ② 크롭 핸들 드래그
        if self._dragging_handle is not None and self._custom_crop is not None:
            h = self._dragging_handle
            RATIO = 16.0 / 9.0

            if h == "move":
                dx = int((event.x - self._drag_start_canvas[0]) / self.scale)
                dy = int((event.y - self._drag_start_canvas[1]) / self.scale)
                ox1, oy1, ox2, oy2 = self._drag_start_crop
                w, hh = ox2 - ox1, oy2 - oy1
                nx1 = max(0, min(ox1 + dx, self.orig_w - w))
                ny1 = max(0, min(oy1 + dy, self.orig_h - hh))
                self._custom_crop = (nx1, ny1, nx1 + w, ny1 + hh)
            else:
                # 코너 드래그: 16:9 비율 유지 (가로 기준)
                ix, iy = self._canvas_to_img(event.x, event.y)
                x1, y1, x2, y2 = self._custom_crop
                if h == "se":
                    w = max(ix - x1, 40)
                    x2, y2 = x1 + w, y1 + int(w / RATIO)
                elif h == "sw":
                    w = max(x2 - ix, 40)
                    x1, y2 = x2 - w, y1 + int(w / RATIO)
                elif h == "ne":
                    w = max(ix - x1, 40)
                    x2, y1 = x1 + w, y2 - int(w / RATIO)
                elif h == "nw":
                    w = max(x2 - ix, 40)
                    x1, y1 = x2 - w, y2 - int(w / RATIO)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.orig_w, x2), min(self.orig_h, y2)
                self._custom_crop = (x1, y1, x2, y2)

            self._render()
            return

        mode = self.mode_var.get()

        # ② 전용 직선 모드 — 미리보기 선
        if mode == "line" and self._line_start_canvas:
            if self._line_preview_id:
                self.canvas.delete(self._line_preview_id)
            sx, sy = self._line_start_canvas
            color = "#ff4444" if self._line_erase else "#44ff44"
            lw = max(int(self.brush_size.get() * self.scale * 2), 2)
            self._line_preview_id = self.canvas.create_line(
                sx, sy, event.x, event.y,
                fill=color, width=lw, tags="line_preview"
            )
            return

        # ③ Shift 직선 모드 — 캔버스 미리보기 선
        if self._line_mode and self._line_start_canvas:
            if self._line_preview_id:
                self.canvas.delete(self._line_preview_id)
            sx, sy = self._line_start_canvas
            color = "#ff4444" if mode == "erase" else "#44ff44"
            self._line_preview_id = self.canvas.create_line(
                sx, sy, event.x, event.y,
                fill=color, width=2, tags="line_preview"
            )
            return

        # ④ 박스 미리보기
        if mode == "box":
            if self._box_start is None:
                return
            sx, sy = self._box_start
            if self._box_rect_id:
                self.canvas.delete(self._box_rect_id)
            self._box_rect_id = self.canvas.create_rectangle(
                sx, sy, event.x, event.y,
                outline="yellow", width=2, dash=(6, 3)
            )

        # ④ 브러시/지우기 페인트
        elif mode in ("brush", "erase"):
            self._brush_paint(event, erase=(mode == "erase"))

    def _on_release(self, event):
        # ① 워터마크 리사이즈 종료
        if self._wm_resizing:
            self._wm_resizing = False
            self._wm_resize_start_canvas = None
            return

        # ① 워터마크 이동 드래그 종료
        if self._dragging_wm_idx is not None:
            self._dragging_wm_idx = None
            self._wm_drag_start_canvas = None
            self._wm_drag_start_pos = None
            return

        # ② 크롭 핸들 드래그 종료
        if self._dragging_handle is not None:
            self._dragging_handle = None
            self._drag_start_canvas = None
            return

        mode = self.mode_var.get()

        # ② 전용 직선 모드 커밋
        if mode == "line" and self._line_start_canvas:
            self._commit_line(event.x, event.y, erase=self._line_erase)
            return

        # ③ Shift 직선 커밋
        if self._line_mode and self._line_start_canvas:
            if self._line_preview_id:
                self.canvas.delete(self._line_preview_id)
                self._line_preview_id = None

            erase = (mode == "erase")
            ix1, iy1 = self._canvas_to_img(*self._line_start_canvas)
            ix2, iy2 = self._canvas_to_img(event.x, event.y)

            # 드래그 거리로 클릭/드래그 구분
            ddx = event.x - self._line_start_canvas[0]
            ddy = event.y - self._line_start_canvas[1]
            is_drag = (ddx * ddx + ddy * ddy) > 25  # 5px 이상이면 드래그

            if self.mask is None:
                self.mask = np.zeros((self.orig_h, self.orig_w), dtype=bool)

            r_img = max(int(self.brush_size.get() / self.scale), 3)

            if is_drag or self._last_brush_xy is None:
                # 드래그: press → release 직선
                start_ix, start_iy = ix1, iy1
            else:
                # 클릭: 마지막 브러시 위치 → 현재 (기존 Shift+클릭 동작 유지)
                start_ix, start_iy = self._last_brush_xy

            dist = int(np.hypot(ix2 - start_ix, iy2 - start_iy))
            steps = max(dist // max(r_img // 2, 1), 1)
            for i in range(steps + 1):
                t = i / steps
                self._apply_brush_at(
                    int(start_ix + (ix2 - start_ix) * t),
                    int(start_iy + (iy2 - start_iy) * t),
                    r_img, not erase
                )
            self._last_brush_xy = (ix2, iy2)
            self._line_mode = False
            self._line_start_canvas = None
            self._render()
            return

        # ③ 박스 모드 — SAM 예측
        if mode == "box":
            if self._box_start is None:
                return
            sx, sy = self._box_start
            ex, ey = event.x, event.y
            self._box_start = None
            if self._box_rect_id:
                self.canvas.delete(self._box_rect_id)
                self._box_rect_id = None
            if abs(ex - sx) > 10 and abs(ey - sy) > 10:
                x1, y1 = min(sx, ex), min(sy, ey)
                x2, y2 = max(sx, ex), max(sy, ey)
                ix1, iy1 = self._canvas_to_img(x1, y1)
                ix2, iy2 = self._canvas_to_img(x2, y2)
                self._run_sam_box_async((ix1, iy1, ix2, iy2))

        # ④ 브러시 완료
        elif mode in ("brush", "erase"):
            self._reset_brush_pos()

    def _on_right_press(self, event):
        mode = self.mode_var.get()
        if mode == "line":
            self._line_start_canvas = (event.x, event.y)
            self._line_erase = True
        elif mode in ("brush", "erase"):
            self._brush_paint(event, erase=True)

    def _on_right_drag(self, event):
        mode = self.mode_var.get()
        if mode == "line" and self._line_start_canvas:
            if self._line_preview_id:
                self.canvas.delete(self._line_preview_id)
            sx, sy = self._line_start_canvas
            lw = max(int(self.brush_size.get() * self.scale * 2), 2)
            self._line_preview_id = self.canvas.create_line(
                sx, sy, event.x, event.y,
                fill="#ff4444", width=lw, tags="line_preview"
            )
        elif mode in ("brush", "erase"):
            self._brush_paint(event, erase=True)

    def _on_right_release(self, event):
        mode = self.mode_var.get()
        if mode == "line" and self._line_start_canvas:
            self._commit_line(event.x, event.y, erase=True)
        else:
            self._reset_brush_pos()

    def _commit_line(self, ex: int, ey: int, erase: bool):
        """직선 커밋 — 시작점~끝점을 브러시 크기로 채우거나 지움"""
        if self._line_preview_id:
            self.canvas.delete(self._line_preview_id)
            self._line_preview_id = None
        ix1, iy1 = self._canvas_to_img(*self._line_start_canvas)
        ix2, iy2 = self._canvas_to_img(ex, ey)
        self._line_start_canvas = None
        if self.mask is None:
            self.mask = np.zeros((self.orig_h, self.orig_w), dtype=bool)
        r_img = max(int(self.brush_size.get() / self.scale), 3)
        dist = int(np.hypot(ix2 - ix1, iy2 - iy1))
        steps = max(dist // max(r_img // 2, 1), 1)
        for i in range(steps + 1):
            t = i / steps
            self._apply_brush_at(
                int(ix1 + (ix2 - ix1) * t),
                int(iy1 + (iy2 - iy1) * t),
                r_img, not erase
            )
        self._render()

    def _reset_brush_pos(self):
        self._last_brush_xy = None

    def _apply_brush_at(self, ix: int, iy: int, r: int, value: bool):
        """로컬 패치만 업데이트하는 빠른 원형 브러시"""
        x1, x2 = max(0, ix - r), min(self.orig_w, ix + r + 1)
        y1, y2 = max(0, iy - r), min(self.orig_h, iy + r + 1)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle = (xx - ix) ** 2 + (yy - iy) ** 2 <= r ** 2
        self.mask[y1:y2, x1:x2][circle] = value

    def _brush_paint(self, event, erase: bool = False):
        if self.mode_var.get() == "erase":
            erase = True
        ix, iy = self._canvas_to_img(event.x, event.y)
        if self.mask is None:
            self.mask = np.zeros((self.orig_h, self.orig_w), dtype=bool)

        r_img = max(int(self.brush_size.get() / self.scale), 3)
        value = not erase

        if self._last_brush_xy is not None:
            px, py = self._last_brush_xy
            dist = int(np.hypot(ix - px, iy - py))
            step = max(r_img // 2, 1)
            if dist > step:
                steps = dist // step
                for i in range(steps + 1):
                    t = i / steps
                    self._apply_brush_at(int(px + (ix - px) * t), int(py + (iy - py) * t), r_img, value)
            else:
                self._apply_brush_at(ix, iy, r_img, value)
        else:
            self._apply_brush_at(ix, iy, r_img, value)

        self._last_brush_xy = (ix, iy)
        self._render()

    # ─────────────────────────── SAM 예측 ───────────────────────────

    def _run_sam_box_async(self, box: tuple):
        if self._sam_running:
            return
        self._push_history()
        self._sam_running = True
        self._update_status("박스 예측 중...")
        threading.Thread(target=self._run_sam_box_predict, args=(box,), daemon=True).start()

    def _run_sam_box_predict(self, box: tuple):
        try:
            mask = self.sam_manager.predict_box(box)
            if mask is not None:
                self.mask = mask
                self._custom_crop = None  # SAM 새 예측 시 custom crop 초기화
            self.after(0, self._render)
            self._update_status("완료. 브러시로 세부 조정하세요.")
        except Exception as e:
            self._update_status(f"예측 실패: {e}")
        finally:
            self._sam_running = False

    def _on_refine_edges(self):
        """GrabCut으로 마스크 경계를 이미지 엣지에 자동 스냅"""
        if self.mask is None or not self.mask.any():
            messagebox.showwarning("경계 정제", "먼저 영역을 선택하세요.", parent=self)
            return
        self._push_history()
        self.btn_refine.config(state=tk.DISABLED, text="정제 중...")
        self._update_status("경계 정제 중...")
        threading.Thread(target=self._run_grabcut, daemon=True).start()

    def _run_grabcut(self):
        try:
            import cv2
            img_bgr = cv2.cvtColor(self.img_np, cv2.COLOR_RGB2BGR)
            mask_u8 = self.mask.astype(np.uint8) * 255

            # 경계 띠 폭 (픽셀) — 이 범위만 GrabCut이 수정
            band = 15
            kernel = np.ones((band * 2 + 1, band * 2 + 1), np.uint8)
            eroded  = cv2.erode(mask_u8,  kernel, iterations=1) > 128
            dilated = cv2.dilate(mask_u8, kernel, iterations=1) > 128

            gc_mask = np.full(self.mask.shape, cv2.GC_BGD, dtype=np.uint8)
            gc_mask[dilated]   = cv2.GC_PR_BGD
            gc_mask[self.mask] = cv2.GC_PR_FGD
            gc_mask[eroded]    = cv2.GC_FGD

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

            refined = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
            self.mask = refined
            self.after(0, self._render)
            self._update_status("경계 정제 완료!")
        except ImportError:
            self._update_status("opencv-python 미설치\npip install opencv-python")
        except Exception as e:
            self._update_status(f"정제 실패: {e}")
        finally:
            self.after(0, lambda: self.btn_refine.config(state=tk.NORMAL, text="✨ 경계 정제"))

    def _set_sam_image(self):
        try:
            self.sam_manager.set_image(self.img_np)
            self._update_status("준비됨. 박스 모드로 간판을 드래그하세요.")
        except Exception as e:
            self._update_status(f"SAM 이미지 설정 실패:\n{e}")

    def _run_sam_async(self):
        if self._sam_running:
            return
        self._sam_running = True
        self._update_status("예측 중...")
        threading.Thread(target=self._run_sam_predict, daemon=True).start()

    def _run_sam_predict(self):
        try:
            mask = self.sam_manager.predict(self.pos_points, self.neg_points)
            if mask is not None:
                self.mask = mask
            self.after(0, self._render)
            pts = len(self.pos_points) + len(self.neg_points)
            self._update_status(f"마스크 생성 완료 (포인트 {pts}개)")
        except Exception as e:
            self._update_status(f"예측 실패: {e}")
        finally:
            self._sam_running = False

    def _update_status(self, msg: str):
        self.after(0, lambda: self.lbl_status.config(text=msg))

    # ─────────────────────────── 초기화 / 적용 ──────────────────────

    def _on_reset(self):
        self.mask = None
        self._custom_crop = None
        self._render()
        self._update_status("초기화됨. 박스 모드로 간판을 드래그하세요.")

    def _on_apply(self):
        if self.mask is None or not self.mask.any():
            messagebox.showwarning(
                "마스크 없음",
                "간판 영역을 먼저 선택하세요.\n박스 모드로 드래그하거나 브러시로 직접 칠해주세요.",
                parent=self
            )
            return

        try:
            crop_rect = self._get_crop_rect()
            if crop_rect is None:
                crop_rect = (0, 0, self.orig_w, self.orig_h)
            processed = apply_postprocess(self.orig_img, self.mask, crop_rect,
                                          wm_positions=self._wm_positions,
                                          wm_size=self._wm_size)

            # 임시 파일 저장
            temp_path = self.image_path.parent / f"{self.image_path.stem}_sam_temp.jpg"
            processed.save(temp_path, "JPEG", quality=95, optimize=True)

            self.parent_app._receive_sam_result(self.image_path, temp_path)
            self.destroy()

        except Exception as e:
            messagebox.showerror("처리 오류", traceback.format_exc(), parent=self)
