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
# 다각형 모드에서 '점 찍기'와 '끌어서 이동'을 가르는 거리 (캔버스 픽셀)
POLY_DRAG_THRESHOLD = 4


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
    daytime_mode: bool = False,  # True=채도 강화 OFF (주간/밝은 사진용)
    apply_wm: bool = True,  # False면 워터마크 굽지 않음
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

    # 간판(컬러): 채도 + 대비 + 선명도 강화
    color_img = cropped.convert("RGB")
    if daytime_mode:
        # 주간/밝은 사진: 채도 강화 없이 원본 컬러 그대로 (파란빛 방지)
        pass
    else:
        # 야간/어두운 사진: 채도 50% + 대비 20% + 선명도 30% 강화
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
    if apply_wm and logo_path.exists():
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

    def __init__(self, parent_app: "App", image_path: Path, sam_manager: "SAMManager",
                 prev_state: Optional[dict] = None):
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

        # ── 다각형 영역 선택 ──
        # 찍는 중인 꼭지점들. 확대/이동해도 점이 사진에 붙어 따라가야 하므로
        # 캔버스 좌표가 아니라 '원본 이미지 좌표'로 보관한다.
        # '영역 선택'을 누르면 내부를 마스크에 더한 뒤 비운다
        # → 다음 다각형은 이전 점들과 무관하게 새로 시작.
        self._poly_points: list = []
        # 좌클릭을 누른 자리. 손을 뗄 때까지 점 찍기를 보류했다가,
        # 일정 거리 이상 끌면 '이동', 제자리에서 떼면 '점 찍기'로 판정한다.
        self._poly_pending: Optional[tuple] = None
        self._poly_dragged: bool = False
        # 누른 지점이 크롭 사각형 안이면 그 크롭을 담아둔다.
        # 끌면 사진 이동이 아니라 이 크롭을 옮긴다.
        self._poly_pending_crop: Optional[tuple] = None

        # ── 크롭 핸들 드래그 ──
        self._custom_crop: Optional[tuple[int, int, int, int]] = None
        self._dragging_handle: Optional[str] = None   # 'nw'|'ne'|'sw'|'se'|'move'
        self._drag_start_canvas: Optional[tuple[int, int]] = None
        self._drag_start_crop: Optional[tuple] = None

        # ── 워터마크 미리보기 ──
        self.show_wm_preview = tk.BooleanVar(value=False)
        self.apply_wm = tk.BooleanVar(value=True)  # 워터마크 굽기 여부 (저장 시 적용)
        self._wm_positions: list = [
            (0.5, 0.88),  # 중앙 하단
        ]
        self._wm_size: float = 0.20               # 로고 너비 (이미지 너비 대비 비율)
        self._dragging_wm_idx: Optional[int] = None
        self._wm_drag_start_canvas: Optional[tuple] = None
        self.daytime_mode = tk.BooleanVar(value=False)  # 주간/밝은 사진 → 채도 강화 OFF
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

        # 표시용 이미지 (zoom = 1.0 일 때 쓰는 캐시)
        self.display_img = self.orig_img.resize(
            (self.canvas_w, self.canvas_h), Image.LANCZOS
        )

        # ── 휠 확대/축소 ──
        # self.scale : 창에 맞춘 기본 배율 (고정)
        # self.zoom  : 그 위에 곱해지는 확대율 (1.0 = 전체 보기)
        # _off_x/_off_y : 화면에 보이는 영역의 좌상단 (원본 이미지 좌표)
        self.zoom: float = 1.0
        self._off_x: float = 0.0
        self._off_y: float = 0.0
        self._base_cache = None        # 확대된 배경 이미지 캐시
        self._base_cache_key = None
        self._panning = False
        self._pan_start = None

        # ── 이전에 '적용'했던 편집 내용 복원 ──
        # 같은 사진을 다시 열었을 때 백지에서 시작하지 않게 한다.
        if prev_state:
            self._restore_state(prev_state)

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

    # ─────────────────────── 편집 상태 저장/복원 ────────────────────

    def _capture_state(self) -> dict:
        """'적용' 시점의 편집 내용을 통째로 담아둔다 (다시 열 때 복원용)"""
        return {
            "mask": self.mask.copy() if self.mask is not None else None,
            "custom_crop": self._custom_crop,
            "wm_positions": list(self._wm_positions),
            "wm_size": self._wm_size,
            "daytime": self.daytime_mode.get(),
            "apply_wm": self.apply_wm.get(),
            "crop_preview": self.show_crop_preview.get(),
            "wm_preview": self.show_wm_preview.get(),
        }

    def _restore_state(self, st: dict) -> None:
        """저장해둔 편집 내용을 되살린다. 사진 크기가 다르면 마스크는 버린다."""
        m = st.get("mask")
        if isinstance(m, np.ndarray) and m.shape == (self.orig_h, self.orig_w):
            self.mask = m.copy()
        crop = st.get("custom_crop")
        if crop:
            self._custom_crop = tuple(crop)
        if st.get("wm_positions"):
            self._wm_positions = list(st["wm_positions"])
        if st.get("wm_size"):
            self._wm_size = float(st["wm_size"])
        self.daytime_mode.set(bool(st.get("daytime", False)))
        self.apply_wm.set(bool(st.get("apply_wm", True)))
        self.show_crop_preview.set(bool(st.get("crop_preview", False)))
        self.show_wm_preview.set(bool(st.get("wm_preview", False)))

    # ─────────────────────────── UI 빌드 ────────────────────────────

    def _build_ui(self):
        # ── 툴바 ──
        toolbar = tk.Frame(self, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)

        tk.Label(toolbar, text="모드:").pack(side=tk.LEFT, padx=(4, 2))
        for label, val in [("📦 박스", "box"), ("🖌 칠하기", "brush"), ("🧹 지우기", "erase"),
                           ("📏 직선", "line"), ("⬠ 다각형", "poly")]:
            tk.Radiobutton(
                toolbar, text=label, variable=self.mode_var, value=val,
                indicatoron=False, width=7, pady=2,
                command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=1)
        tk.Label(toolbar,
                 text="  (휠=확대 / 다각형: 클릭=점, 끌기=이동, 우클릭·Enter=영역 선택)",
                 fg="gray", font=("", 8)).pack(side=tk.LEFT, padx=(6, 0))
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
        tk.Checkbutton(
            toolbar, text="워터마크 삽입", variable=self.apply_wm,
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
        # 휠 확대 / 가운데버튼 드래그 이동
        self.canvas.bind("<MouseWheel>",      self._on_wheel)
        self.canvas.bind("<Button-2>",        self._on_pan_press)
        self.canvas.bind("<B2-Motion>",       self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)
        self.brush_size.trace_add("write", lambda *_: self._update_brush_cursor())
        # 되돌리기 단축키
        self.bind("<Control-z>", self._on_undo)
        # 다각형 단축키
        self.bind("<Return>",    self._commit_polygon)
        self.bind("<KP_Enter>",  self._commit_polygon)
        self.bind("<BackSpace>", self._undo_polygon_point)
        self.bind("<Escape>",    self._cancel_polygon)
        # 확대 초기화
        self.bind("<Control-Key-0>", self._reset_zoom)

        # 사이드패널
        side = tk.Frame(body, width=160)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0))
        side.pack_propagate(False)

        tk.Label(side, text="사용법", font=("", 9, "bold")).pack(anchor=tk.W)
        tk.Label(side, text="① 박스로 간판 드래그", justify=tk.LEFT).pack(anchor=tk.W)
        tk.Label(side, text="② 다각형은 꼭지점을\n  찍고 '해당 영역 선택'", justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        tk.Label(side, text="② 칠하기/지우기로\n  세부 조정", justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        tk.Label(side, text="③ 경계 정제로\n  엣지 자동 스냅", justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── 다각형 영역 선택 ──
        self.btn_poly_apply = tk.Button(
            side, text="⬠ 해당 영역 선택", command=self._commit_polygon,
            bg="#FF9800", fg="white", width=16, font=("", 9, "bold")
        )
        self.btn_poly_apply.pack(anchor=tk.W, pady=(0, 2))
        self.btn_poly_undo = tk.Button(
            side, text="↺ 마지막 점 취소", command=self._undo_polygon_point, width=16
        )
        self.btn_poly_undo.pack(anchor=tk.W, pady=(0, 6))

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        self.btn_refine = tk.Button(
            side, text="✨ 경계 정제", command=self._on_refine_edges,
            bg="#4CAF50", fg="white", width=12, font=("", 9, "bold")
        )
        self.btn_refine.pack(anchor=tk.W, pady=(0, 4))

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── 주간/밝은 사진 모드 ──
        tk.Checkbutton(
            side, text="☀ 주간/밝은 사진\n(채도 강화 끄기)",
            variable=self.daytime_mode,
            justify=tk.LEFT, anchor=tk.W,
            font=("", 9)
        ).pack(anchor=tk.W, pady=(0, 6))

        ttk.Separator(side, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

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
        composite = self._base_image().copy().convert("RGBA")

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
                    sx1, sy1 = self._img_to_canvas(crop[0], crop[1])
                    sx2, sy2 = self._img_to_canvas(crop[2], crop[3])
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

        # 워터마크 미리보기 (삽입이 켜져 있을 때만)
        if self.show_wm_preview.get() and self.apply_wm.get():
            crop = self._get_crop_rect() or (0, 0, self.orig_w, self.orig_h)
            x1c, y1c, x2c, y2c = crop
            cw, ch = x2c - x1c, y2c - y1c
            wm_tile = self._get_wm_tile()
            draw2 = ImageDraw.Draw(composite)
            for i, (rx, ry) in enumerate(self._wm_positions):
                # crop 좌표 → canvas 좌표
                cx, cy = self._img_to_canvas(x1c + rx * cw, y1c + ry * ch)
                tw_c = th_c = 0
                if wm_tile is not None:
                    tw_c = int(wm_tile.width * self._eff_scale())
                    th_c = int(wm_tile.height * self._eff_scale())
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
                    crop_cx, crop_cy = self._img_to_canvas(x1c + cw * 0.5, y1c + ch * 0.5)
                    snap_px = int(cw * self._eff_scale() * 0.06)  # 6% 이내면 가이드 표시
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

        # 브러시 커서 / 다각형 오버레이 복원 (delete("all") 후)
        self._update_brush_cursor()
        self._draw_polygon_overlay()

    def _resize_mask_to_canvas(self, mask: np.ndarray) -> np.ndarray:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize(
            (self.canvas_w, self.canvas_h), Image.NEAREST, box=self._visible_rect()
        )
        return np.array(mask_img) > 128

    # ─────────────────────────── 좌표 변환 ──────────────────────────

    def _eff_scale(self) -> float:
        """화면 1px = 원본 몇 px 인지 결정하는 실효 배율"""
        return self.scale * self.zoom

    def _visible_rect(self) -> tuple:
        """현재 화면에 보이는 원본 이미지 영역 (x1, y1, x2, y2) — float"""
        if self.zoom <= 1.0001:
            return (0.0, 0.0, float(self.orig_w), float(self.orig_h))
        s = self._eff_scale()
        return (self._off_x, self._off_y,
                self._off_x + self.canvas_w / s,
                self._off_y + self.canvas_h / s)

    def _clamp_view(self):
        """보이는 영역이 이미지 밖으로 나가지 않게 보정"""
        if self.zoom <= 1.0001:
            self._off_x = self._off_y = 0.0
            return
        s = self._eff_scale()
        vw = self.canvas_w / s
        vh = self.canvas_h / s
        self._off_x = max(0.0, min(self.orig_w - vw, self._off_x))
        self._off_y = max(0.0, min(self.orig_h - vh, self._off_y))

    def _canvas_to_img(self, cx: int, cy: int) -> tuple[int, int]:
        s = self._eff_scale()
        ix = int(self._off_x + cx / s)
        iy = int(self._off_y + cy / s)
        return (
            max(0, min(self.orig_w - 1, ix)),
            max(0, min(self.orig_h - 1, iy)),
        )

    def _img_to_canvas(self, ix, iy) -> tuple[int, int]:
        s = self._eff_scale()
        return int((ix - self._off_x) * s), int((iy - self._off_y) * s)

    # ─────────────────────────── 확대 / 이동 ───────────────────────

    def _base_image(self) -> Image.Image:
        """보이는 영역을 캔버스 크기로 렌더한 배경 (zoom/offset 바뀔 때만 재계산)"""
        if self.zoom <= 1.0001:
            return self.display_img
        key = (round(self.zoom, 4), round(self._off_x, 2), round(self._off_y, 2))
        if self._base_cache is not None and self._base_cache_key == key:
            return self._base_cache
        # box에 float를 넘기면 서브픽셀까지 정확히 잘라준다 (점 찍기 정밀도 유지)
        base = self.orig_img.resize(
            (self.canvas_w, self.canvas_h), Image.BILINEAR, box=self._visible_rect()
        )
        self._base_cache = base
        self._base_cache_key = key
        return base

    def _invalidate_view(self):
        self._base_cache = None
        self._base_cache_key = None

    def _on_wheel(self, event):
        """휠 = 커서 아래 지점을 붙들어 둔 채 확대/축소"""
        s_old = self._eff_scale()
        ix = self._off_x + event.x / s_old
        iy = self._off_y + event.y / s_old

        step = 1.25 if event.delta > 0 else 1 / 1.25
        new_zoom = max(1.0, min(16.0, self.zoom * step))
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        self.zoom = new_zoom

        # 커서 아래 있던 원본 좌표가 그대로 커서 아래 남도록 offset 재계산
        s_new = self._eff_scale()
        self._off_x = ix - event.x / s_new
        self._off_y = iy - event.y / s_new
        self._clamp_view()
        self._invalidate_view()
        self._render()
        if self.zoom > 1.0001:
            self._update_status(f"확대 {self.zoom * 100:.0f}%  "
                                "(가운데버튼 드래그=이동, Ctrl+0=전체)")
        else:
            self._update_status("전체 보기")

    def _reset_zoom(self, event=None):
        if self.zoom <= 1.0001:
            return
        self.zoom = 1.0
        self._off_x = self._off_y = 0.0
        self._invalidate_view()
        self._render()
        self._update_status("전체 보기로 복귀")

    def _begin_pan(self, x: int, y: int):
        """(x, y)를 잡은 지점으로 삼아 화면 이동 시작"""
        if self.zoom <= 1.0001:
            return
        self._panning = True
        self._pan_start = (x, y, self._off_x, self._off_y)
        self.canvas.config(cursor="fleur")

    def _on_pan_press(self, event):
        self._begin_pan(event.x, event.y)

    def _on_pan_drag(self, event):
        if not self._panning or self._pan_start is None:
            return
        sx, sy, ox, oy = self._pan_start
        s = self._eff_scale()
        self._off_x = ox - (event.x - sx) / s
        self._off_y = oy - (event.y - sy) / s
        self._clamp_view()
        self._invalidate_view()
        self._render()

    def _on_pan_release(self, event):
        self._panning = False
        self._pan_start = None
        self.canvas.config(cursor="crosshair")

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

    # ─────────────────────────── 다각형 영역 ────────────────────────

    def _on_mode_change(self):
        """모드 전환 시 커서/오버레이 정리"""
        mode = self.mode_var.get()
        if mode != "poly" and self._poly_points:
            # 다각형 모드를 벗어나면 찍던 점은 버린다
            self._poly_points = []
        self._update_brush_cursor()
        self._draw_polygon_overlay()
        if mode == "poly":
            self._update_status("꼭지점을 클릭해서 찍고, 우클릭 또는 Enter로 영역 선택.")

    def _draw_polygon_overlay(self):
        """찍는 중인 다각형을 캔버스에 표시 (아직 마스크가 아닌 임시 표시)"""
        self.canvas.delete("poly_overlay")
        if self.mode_var.get() != "poly" or not self._poly_points:
            return
        # 원본 좌표 → 현재 화면 좌표로 변환해서 그린다
        pts = [self._img_to_canvas(ix, iy) for ix, iy in self._poly_points]
        for i in range(len(pts) - 1):
            self.canvas.create_line(*pts[i], *pts[i + 1],
                                    fill="#ffd400", width=2, tags="poly_overlay")
        # 닫히는 변은 점선으로 미리 보여준다
        if len(pts) >= 3:
            self.canvas.create_line(*pts[-1], *pts[0],
                                    fill="#ffd400", width=2, dash=(5, 3), tags="poly_overlay")
        for i, (x, y) in enumerate(pts):
            r = 5
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=("#ff6a00" if i == 0 else "#ffffff"),   # 첫 점은 주황
                outline="#000000", width=1, tags="poly_overlay"
            )
        x0, y0 = pts[0]
        self.canvas.create_text(
            x0 + 10, y0 - 12, text=f"{len(pts)}점", anchor=tk.W,
            fill="#ffd400", font=("", 9, "bold"), tags="poly_overlay"
        )

    def _commit_polygon(self, event=None):
        """
        찍어둔 꼭지점 내부를 마스크에 더한다.
        커밋 후 점 목록을 비우므로, 다음 다각형은 이전 점들과 완전히 독립적이다.
        """
        if not self._poly_points:
            if self.mode_var.get() == "poly":
                self._update_status("먼저 꼭지점을 클릭해서 찍어주세요.")
            return
        if len(self._poly_points) < 3:
            self._update_status(f"꼭지점이 {len(self._poly_points)}개뿐입니다. 3개 이상 필요합니다.")
            return

        self._push_history()
        if self.mask is None:
            self.mask = np.zeros((self.orig_h, self.orig_w), dtype=bool)

        # 이미 원본 좌표로 들고 있으므로 그대로 채운다
        img_pts = list(self._poly_points)
        poly_img = Image.new("L", (self.orig_w, self.orig_h), 0)
        ImageDraw.Draw(poly_img).polygon(img_pts, fill=255)

        # OR 합성 → 앞서 만들어둔 영역은 건드리지 않고 새 영역만 추가
        self.mask = self.mask | (np.array(poly_img) > 127)

        n = len(self._poly_points)
        self._poly_points = []
        self._render()
        self._update_status(f"영역 추가 완료 (꼭지점 {n}개). 다른 영역을 이어서 찍으세요.")

    def _undo_polygon_point(self, event=None):
        """마지막으로 찍은 꼭지점 하나만 취소 (마스크는 건드리지 않음)"""
        if not self._poly_points:
            return
        self._poly_points.pop()
        self._draw_polygon_overlay()
        self._update_status(f"꼭지점 {len(self._poly_points)}개")

    def _cancel_polygon(self, event=None):
        """찍던 다각형 전체 취소"""
        if not self._poly_points:
            return
        self._poly_points = []
        self._draw_polygon_overlay()
        self._update_status("찍던 꼭지점을 취소했습니다.")

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

    def _crop_move_to(self, ex: int, ey: int):
        """크롭 사각형을 누른 지점 기준으로 평행 이동 (크기 유지)"""
        dx = int((ex - self._drag_start_canvas[0]) / self._eff_scale())
        dy = int((ey - self._drag_start_canvas[1]) / self._eff_scale())
        ox1, oy1, ox2, oy2 = self._drag_start_crop
        cw, chh = ox2 - ox1, oy2 - oy1
        nx1 = max(0, min(ox1 + dx, self.orig_w - cw))
        ny1 = max(0, min(oy1 + dy, self.orig_h - chh))
        self._custom_crop = (nx1, ny1, nx1 + cw, ny1 + chh)

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
        if self.show_wm_preview.get() and self.apply_wm.get():
            crop = self._get_crop_rect() or (0, 0, self.orig_w, self.orig_h)
            x1c, y1c, x2c, y2c = crop
            cw, ch = x2c - x1c, y2c - y1c
            wm_tile = self._get_wm_tile()
            for i, (rx, ry) in enumerate(self._wm_positions):
                cx, cy = self._img_to_canvas(x1c + rx * cw, y1c + ry * ch)
                # 리사이즈 핸들 (우하단) 먼저 체크
                if wm_tile is not None:
                    tw_c = int(wm_tile.width * self._eff_scale())
                    th_c = int(wm_tile.height * self._eff_scale())
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
                handle = self._hit_test_crop(event.x, event.y, crop)
                # 다각형 모드에서 크롭 안쪽(move)을 누른 경우는 바로 판단하지 않는다.
                # 제자리에서 떼면 꼭지점 찍기, 끌면 크롭 이동.
                if handle == "move" and self.mode_var.get() == "poly":
                    self._poly_pending = (event.x, event.y)
                    self._poly_dragged = False
                    self._poly_pending_crop = crop
                    return
                if handle:
                    # 핸들을 실제로 잡았을 때만 크롭을 수동 고정으로 전환
                    # (그냥 클릭했다고 고정되면 이후 마스크가 커져도 크롭이 안 따라감)
                    if self._custom_crop is None:
                        self._custom_crop = crop
                    self._dragging_handle = handle
                    self._drag_start_canvas = (event.x, event.y)
                    self._drag_start_crop = self._custom_crop
                    return  # 브러시/박스 차단

        mode = self.mode_var.get()

        # ② 다각형 모드 — 꼭지점 추가
        if mode == "poly":
            # 여기서 바로 찍지 않고 손 뗄 때까지 보류한다.
            # 끌면 화면 이동, 제자리에서 떼면 점 찍기.
            self._poly_pending = (event.x, event.y)
            self._poly_dragged = False
            self._poly_pending_crop = None
            return

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
            delta_rel = (dx + dy) / 2 / (self.orig_w * self._eff_scale())
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
            dx_rel = (event.x - self._wm_drag_start_canvas[0]) / (cw * self._eff_scale())
            dy_rel = (event.y - self._wm_drag_start_canvas[1]) / (ch * self._eff_scale())
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
                self._crop_move_to(event.x, event.y)
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

        # ② 다각형 모드 — 일정 거리 이상 끌면 화면 이동으로 전환
        if mode == "poly" and self._poly_pending is not None:
            sx, sy = self._poly_pending
            crossed = (abs(event.x - sx) > POLY_DRAG_THRESHOLD
                       or abs(event.y - sy) > POLY_DRAG_THRESHOLD)
            if not self._poly_dragged and crossed:
                if self._poly_pending_crop is not None:
                    # 크롭 안쪽을 끌었다 → 사진이 아니라 크롭 사각형을 옮긴다
                    self._poly_dragged = True
                    if self._custom_crop is None:
                        self._custom_crop = self._poly_pending_crop
                    self._dragging_handle = "move"
                    self._drag_start_canvas = (sx, sy)
                    self._drag_start_crop = self._custom_crop
                elif self.zoom > 1.0001:
                    # 크롭 밖이고 확대 상태 → 사진 이동
                    self._poly_dragged = True
                    self._begin_pan(sx, sy)
            if self._poly_dragged:
                if self._dragging_handle == "move":
                    self._crop_move_to(event.x, event.y)
                    self._render()
                elif self._panning:
                    self._on_pan_drag(event)
            return

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
            # 다각형 모드에서 크롭을 끈 경우도 여기로 오므로 보류 상태를 함께 정리
            self._poly_pending = None
            self._poly_dragged = False
            self._poly_pending_crop = None
            return

        mode = self.mode_var.get()

        # ② 다각형 모드 — 끌었으면 이동으로 끝, 제자리면 점 찍기
        if mode == "poly":
            pending, dragged = self._poly_pending, self._poly_dragged
            self._poly_pending = None
            self._poly_dragged = False
            self._poly_pending_crop = None
            if self._panning:
                self._on_pan_release(event)
            if pending is not None and not dragged:
                # 원본 좌표로 저장 → 확대/축소/이동해도 점이 사진에 붙어 따라간다
                self._poly_points.append(self._canvas_to_img(*pending))
                self._draw_polygon_overlay()
                self._update_status(
                    f"꼭지점 {len(self._poly_points)}개. "
                    "우클릭 또는 Enter로 영역 선택."
                )
            return

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

            r_img = max(int(self.brush_size.get() / self._eff_scale()), 3)

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
                # Ctrl 누른 상태면 기존 마스크에 추가, 아니면 교체
                add_mode = bool(event.state & 0x0004)
                self._run_sam_box_async((ix1, iy1, ix2, iy2), add_mode=add_mode)

        # ④ 브러시 완료
        elif mode in ("brush", "erase"):
            self._reset_brush_pos()

    def _on_right_press(self, event):
        mode = self.mode_var.get()
        if mode == "poly":
            self._commit_polygon()
        elif mode == "line":
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
        r_img = max(int(self.brush_size.get() / self._eff_scale()), 3)
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

        r_img = max(int(self.brush_size.get() / self._eff_scale()), 3)
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

    def _run_sam_box_async(self, box: tuple, add_mode: bool = False):
        if self._sam_running:
            return
        self._push_history()
        self._sam_running = True
        self._update_status("박스 예측 중..." + (" (추가)" if add_mode else ""))
        threading.Thread(target=self._run_sam_box_predict, args=(box, add_mode), daemon=True).start()

    def _run_sam_box_predict(self, box: tuple, add_mode: bool = False):
        try:
            mask = self.sam_manager.predict_box(box)
            if mask is not None:
                if add_mode and self.mask is not None:
                    # Ctrl+박스: 기존 마스크에 새 마스크 OR 합성
                    self.mask = self.mask | mask
                else:
                    self.mask = mask
                    self._custom_crop = None  # 일반 박스: custom crop 초기화
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
            if self.mask is not None and self.mask.any():
                self._update_status("이전 편집 내용을 불러왔습니다. 이어서 수정하세요.")
            else:
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
        self._poly_points = []
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
                                          wm_size=self._wm_size,
                                          daytime_mode=self.daytime_mode.get(),
                                          apply_wm=self.apply_wm.get())

            # 임시 파일 저장
            temp_path = self.image_path.parent / f"{self.image_path.stem}_sam_temp.jpg"
            processed.save(temp_path, "JPEG", quality=95, optimize=True)

            # 편집 내용을 함께 넘겨서, 다시 열었을 때 이어서 작업할 수 있게 한다
            self.parent_app._receive_sam_result(self.image_path, temp_path,
                                                self._capture_state())
            self.destroy()

        except Exception as e:
            messagebox.showerror("처리 오류", traceback.format_exc(), parent=self)
