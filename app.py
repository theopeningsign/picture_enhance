import io
import os
import sys
import queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from PIL import Image, ImageOps, ImageFile, ImageEnhance, ImageFilter, ImageStat

# SAM 통합 (설치되지 않은 환경에서도 기본 압축 기능은 동작)
try:
    from sam_manager import SAMManager
    from edit_window import EditWindow
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
# 손상된 이미지 파일도 처리하도록 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:
    HEIF_AVAILABLE = False
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


BYTES_IN_MB = 1024 * 1024

# Pillow 버전 호환 LANCZOS 상수
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 9
except Exception:
    RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class ResultItem:
    src_path: Path
    status: str  # 'pass' | 'compressed' | 'error'
    before_bytes: int
    after_bytes: int
    out_path: Optional[Path]
    message: str = ''


@dataclass(frozen=True)
class CompressionProfile:
    label: str
    target_bytes: int
    hard_limit_bytes: int
    pass_threshold_bytes: Optional[int]
    quality_low: int
    quality_high: int
    max_downscale_attempts: int
    downscale_ratio: float
    min_dimension: int


# 초고해상도 이미지 사전 리사이즈 설정
# 긴 변이 너무 큰 경우(예: 3000px 이상), 먼저 웹용 사이즈로 한 번 줄인 뒤
# 나머지 기존 압축 로직(품질 탐색 + 소폭 리사이즈)을 적용한다.
# 포트폴리오 웹용 기준: 4000×3000 같은 이미지도 과하므로, 3000px 이상이면 사전 리사이즈
PRE_DOWNSCALE_LONG_SIDE_THRESHOLD = 3000  # 이 값 이상이면 사전 리사이즈 대상
# 포트폴리오 웹용: 가로 2000~2400px 정도면 충분히 선명하고 용량도 적당
PRE_DOWNSCALE_TARGET_LONG_SIDE = 2400     # 긴 변을 이 정도로 맞춤 (필요 시 조정 가능)


PROFILE = CompressionProfile(
    label="강력 압축",
    target_bytes=int(1.8 * BYTES_IN_MB),
    hard_limit_bytes=2_100_000,
    pass_threshold_bytes=None,
    quality_low=40,
    quality_high=90,
    max_downscale_attempts=4,
    downscale_ratio=0.92,
    min_dimension=1600,
)


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < BYTES_IN_MB:
        return f"{n / 1024:.1f} KB"
    return f"{n / BYTES_IN_MB:.1f} MB"


def has_alpha(img: Image.Image) -> bool:
    mode = img.mode
    if mode in ("RGBA", "LA"):
        return True
    if mode == "P":
        return "transparency" in img.info
    return False


def flatten_to_rgb(img: Image.Image) -> Image.Image:
    # 손상된 이미지 파일 처리를 위해 완전히 로드
    try:
        img.load()
    except Exception:
        pass  # 로드 실패해도 시도는 계속
    
    # EXIF 회전 보정 시도 (손상된 파일일 수 있으므로 예외 처리)
    def safe_exif_transpose(im: Image.Image) -> Image.Image:
        try:
            return ImageOps.exif_transpose(im)
        except Exception:
            # EXIF 처리 실패 시 원본 반환
            return im
    
    if not has_alpha(img):
        # EXIF 회전 보정만 적용
        rgb_img = img.convert("RGB") if img.mode != "RGB" else img
        return safe_exif_transpose(rgb_img)
    # 흰 배경에 합성
    base = Image.new("RGB", img.size, (255, 255, 255))
    rgba_img = safe_exif_transpose(img.convert("RGBA"))
    base.paste(rgba_img, mask=rgba_img.split()[-1])
    return base


def _is_high_key_scene(img: Image.Image) -> bool:
    """
    (보류용) 전체적으로 매우 밝고(하얀 벽/간판 등), 명암 차이가 크지 않은 고키(high-key) 장면인지 판별.
    현재는 외부/내부 모드를 명시적으로 선택하므로 사용하지 않지만,
    추후 자동 모드 감지를 위해 남겨둔다.
    """
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean_l = stat.mean[0]
    std_l = stat.stddev[0]
    return mean_l >= 210 and std_l < 20


def apply_watermark_subtle(img: Image.Image) -> Image.Image:
    """
    내부 사진용 미세 워터마크.
    흰색 반투명(불투명도 40%), 너비 8%, 하단 중앙 배치.
    밝은 인테리어 사진에 자연스럽게 녹아드는 스타일.
    """
    logo_path = Path(__file__).parent / "logo.png"
    if not logo_path.exists():
        return img

    logo = Image.open(logo_path).convert("RGBA")

    # 긴 변 기준 10.4% → 가로/세로 무관하게 일정 크기
    base = max(img.width, img.height)
    logo_w = max(int(base * 0.104), 52)
    ratio = logo_w / logo.width
    logo_h = int(logo.height * ratio)
    logo = logo.resize((logo_w, logo_h), RESAMPLE_LANCZOS)

    # 흰색 로고 생성
    alpha = logo.split()[3]
    white_logo = Image.new("RGBA", logo.size, (255, 255, 255, 255))
    white_logo.putalpha(alpha)

    # 불투명도 40%로 낮추기 (point()로 numpy 없이 처리)
    alpha_faded = alpha.point(lambda x: int(x * 0.40))
    white_logo.putalpha(alpha_faded)

    # 하단 중앙 배치 — 기존 ly에서 logo_h만큼 위로 올림
    margin = max(int(img.height * 0.025), 10)
    lx = (img.width - logo_w) // 2
    ly = img.height - logo_h - margin - logo_h

    result = img.convert("RGBA")
    result.paste(white_logo, (lx, ly), mask=white_logo)
    return result.convert("RGB")


def apply_portfolio_filter(img: Image.Image, mode: str = "outdoor") -> Image.Image:
    """
    포트폴리오 웹용 보정 필터.

    mode:
      - "outdoor" : 외부(야간/거리 포함)용, 비교적 강한 보정
      - "indoor"  : 내부(포트폴리오/간판/인테리어)용, 약한 보정 + 웜톤
    """
    if mode == "indoor":
        # 내부(포트폴리오) 모드:
        # - 이미 충분히 밝은 인테리어/간판 사진을 전제로,
        # - 밝기는 거의 유지하고, 엣지/텍스트 선명도와 미세한 웜톤만 부여

        # 1) 기본 톤: 밝기/대비는 과하지 않게만 살짝 정리
        img = ImageEnhance.Brightness(img).enhance(1.02)
        img = ImageEnhance.Contrast(img).enhance(1.05)

        # 2) 선명도: 웹용 기준으로 조금 또렷하게
        img = ImageEnhance.Sharpness(img).enhance(1.25)

        # 3) 고급 웜톤: 채널별 미세 조정 (화이트는 유지하면서 아주 살짝 따뜻하게)
        r, g, b = img.split()
        r = r.point(lambda i: i * 1.015)  # 빨간기 +1.5%
        b = b.point(lambda i: i * 0.985)  # 파란기 -1.5%
        img = Image.merge("RGB", (r, g, b))

        return img

    # 기본값: 외부(야간/거리) 모드 - 이전에 쓰던 상대적으로 강한 보정
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Color(img).enhance(1.05)
    img = ImageEnhance.Contrast(img).enhance(1.03)
    img = ImageEnhance.Sharpness(img).enhance(1.05)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
    return img


def jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
    )
    return buf.getvalue()


def search_quality_for_target(
    img: Image.Image,
    target_bytes: int,
    quality_low: int,
    quality_high: int,
    max_iter: int = 8,
) -> Tuple[int, bytes]:
    low, high = quality_low, quality_high
    best_q = high
    best_data = jpeg_bytes(img, best_q)

    # 목표 이하 중에서 가장 큰(품질 높은) 결과를 선택
    for _ in range(max_iter):
        mid = (low + high) // 2
        data = jpeg_bytes(img, mid)
        if len(data) <= target_bytes:
            best_q = mid
            best_data = data
            low = mid + 1  # 더 높은 품질 시도
        else:
            high = mid - 1

    # 경계 확인 (low/hi 근처 품질도 확인)
    for q in [high, low]:
        if quality_low <= q <= quality_high:
            d = jpeg_bytes(img, q)
            if len(d) <= target_bytes and len(d) > len(best_data):
                best_q, best_data = q, d

    return best_q, best_data


def compress_file_to_target(path: Path, profile: CompressionProfile, apply_enhance: bool = False, filter_mode: str = "outdoor") -> ResultItem:
    try:
        before = path.stat().st_size
        if profile.pass_threshold_bytes is not None and before <= profile.pass_threshold_bytes:
            # 용량이 작아서 압축은 건너뛰는 조건이지만,
            # 보정 통합 실행(apply_enhance=True)인 경우에는 최소한 보정은 수행한다.
            if apply_enhance:
                return enhance_only_file(path, mode=filter_mode)
            return ResultItem(
                src_path=path,
                status="pass",
                before_bytes=before,
                after_bytes=before,
                out_path=None,
                message="통과",
            )

        # 손상된 이미지 파일 처리 강화
        try:
            with Image.open(path) as im:
                # 이미지 완전히 로드 시도 (손상된 파일 처리)
                try:
                    im.load()
                except Exception as load_err:
                    # 로드 실패해도 시도는 계속
                    pass
                img = flatten_to_rgb(im)
        except Exception as open_err:
            # 파일 열기 자체가 실패한 경우
            raise Exception(f"이미지 파일 열기 실패: {open_err}")

        # 0단계: 초고해상도 이미지 사전 리사이즈
        # 긴 변이 너무 큰 경우(예: 5400×8200 등), 먼저 웹용 사이즈(예: 2600px)로 축소
        width, height = img.size
        long_side = max(width, height)
        if long_side >= PRE_DOWNSCALE_LONG_SIDE_THRESHOLD:
            # 목표 긴 변은 최소 축소 한계(profile.min_dimension)보다 작지 않도록 보정
            target_long = max(PRE_DOWNSCALE_TARGET_LONG_SIDE, profile.min_dimension)
            scale = target_long / float(long_side)
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            img = img.resize((new_w, new_h), RESAMPLE_LANCZOS)
            width, height = img.size

        if apply_enhance:
            img = apply_portfolio_filter(img, mode=filter_mode)

        # 1차: 품질 이진 탐색으로 타깃 근사
        final_pil = img   # 워터마크 삽입을 위해 최종 PIL 이미지 추적
        q, data = search_quality_for_target(img, profile.target_bytes, profile.quality_low, profile.quality_high)

        # 2차: 여전히 초과면 단계적 다운스케일 + 품질 재탐색
        width, height = img.size
        attempt = 0
        while (
            len(data) > profile.hard_limit_bytes
            and attempt < profile.max_downscale_attempts
            and width > profile.min_dimension
            and height > profile.min_dimension
        ):
            attempt += 1
            width = int(width * profile.downscale_ratio)
            height = int(height * profile.downscale_ratio)
            resized = img.resize((max(1, width), max(1, height)), RESAMPLE_LANCZOS)
            q, data = search_quality_for_target(resized, profile.target_bytes, profile.quality_low, profile.quality_high)
            final_pil = resized   # 루프를 돌 때마다 최신 PIL로 갱신

        if len(data) >= before:
            # 압축 결과가 원본보다 크거나 이득이 없을 때도,
            # 보정 통합 실행인 경우에는 보정-only 결과를 남긴다.
            if apply_enhance:
                return enhance_only_file(path, mode=filter_mode)
            return ResultItem(
                src_path=path,
                status="pass",
                before_bytes=before,
                after_bytes=before,
                out_path=None,
                message="원본이 더 작음 → 건너뜀",
            )

        # 내부 모드: 최종 이미지에 미세 워터마크 삽입 후 재인코딩
        if filter_mode == "indoor":
            wm = apply_watermark_subtle(final_pil)
            buf = io.BytesIO()
            wm.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
            data = buf.getvalue()

        suffix = "_compressed_enhanced" if apply_enhance else "_compressed"
        out_path = path.with_name(f"{path.stem}{suffix}.jpg")
        with open(out_path, "wb") as f:
            f.write(data)

        after = out_path.stat().st_size
        return ResultItem(src_path=path, status="compressed", before_bytes=before, after_bytes=after, out_path=out_path, message=f"Q={q}")

    except Exception as e:
        tb = traceback.format_exc()
        return ResultItem(src_path=path, status="error", before_bytes=0, after_bytes=0, out_path=None, message=f"{e}\n{tb}")


def enhance_only_file(path: Path, jpeg_quality: int = 90, mode: str = "outdoor") -> ResultItem:
    """리사이즈/압축 없이 보정만 적용. 저장: *_enhanced.jpg"""
    try:
        before = path.stat().st_size
        with Image.open(path) as im:
            try:
                im.load()
            except Exception:
                pass
            img = flatten_to_rgb(im)
        img = apply_portfolio_filter(img, mode=mode)
        # 내부 모드: 보정 후 미세 워터마크 삽입
        if mode == "indoor":
            img = apply_watermark_subtle(img)
        out_path = path.with_name(f"{path.stem}_enhanced.jpg")
        data = jpeg_bytes(img, jpeg_quality)
        with open(out_path, "wb") as f:
            f.write(data)
        after = out_path.stat().st_size
        return ResultItem(
            src_path=path,
            status="enhanced",
            before_bytes=before,
            after_bytes=after,
            out_path=out_path,
            message="보정 완료",
        )
    except Exception as e:
        tb = traceback.format_exc()
        return ResultItem(src_path=path, status="error", before_bytes=0, after_bytes=0, out_path=None, message=f"{e}\n{tb}")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("이미지 압축 & SAM 포트폴리오 최적화")
        self.geometry("900x640")
        self.minsize(800, 480)

        self.profile = PROFILE
        self._build_ui()

        self.files: List[Path] = []
        self.results: List[ResultItem] = []
        self.q: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None

        # SAM 관련
        self.sam_manager: Optional["SAMManager"] = SAMManager() if SAM_AVAILABLE else None
        self._sam_temps: Dict[Path, Path] = {}   # original Path → temp Path
        self._file_index: Dict[Path, int] = {}   # process Path → 트리뷰 행 인덱스

        if self.sam_manager is not None:
            threading.Thread(target=self._init_sam_model, daemon=True).start()

        self.after(100, self._poll_queue)

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 8}

        top = ttk.Frame(self)
        top.pack(fill=tk.X, **pad)

        self.btn_select = ttk.Button(top, text="이미지 선택", command=self.on_select)
        self.btn_select.pack(side=tk.LEFT)

        self.btn_start = ttk.Button(top, text="압축 실행", command=self.on_start, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=6)

        self.btn_enhance_only = ttk.Button(top, text="보정만 실행", command=self.on_enhance_only, state=tk.DISABLED)
        self.btn_enhance_only.pack(side=tk.LEFT, padx=6)

        self.btn_open = ttk.Button(top, text="폴더 열기", command=self.on_open_folder, state=tk.DISABLED)
        self.btn_open.pack(side=tk.LEFT, padx=6)

        # 기본값: 압축 시 보정 적용 ON
        self.apply_enhance_var = tk.BooleanVar(value=True)
        self.chk_enhance = ttk.Checkbutton(top, text="압축 시 보정 적용", variable=self.apply_enhance_var)
        self.chk_enhance.pack(side=tk.LEFT, padx=(12, 0))

        # 보정 모드 선택: 외부(야간/거리) / 내부(포트폴리오)
        self.filter_mode_var = tk.StringVar(value="outdoor")
        mode_frame = ttk.Frame(top)
        mode_frame.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(mode_frame, text="보정 모드:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="외부", value="outdoor", variable=self.filter_mode_var).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="내부", value="indoor", variable=self.filter_mode_var).pack(side=tk.LEFT)

        self.lbl_target = ttk.Label(top, text=self._profile_summary())
        self.lbl_target.pack(side=tk.RIGHT)

        sam_text = "SAM: 로딩 중..." if SAM_AVAILABLE else "SAM: 미설치"
        sam_color = "blue" if SAM_AVAILABLE else "gray"
        self.lbl_sam = ttk.Label(top, text=sam_text, foreground=sam_color)
        self.lbl_sam.pack(side=tk.RIGHT, padx=(0, 8))

        # SAM 장치 선택 (자동 / CPU / CUDA)
        if SAM_AVAILABLE:
            sam_dev_frame = ttk.Frame(top)
            sam_dev_frame.pack(side=tk.RIGHT, padx=(0, 4))
            ttk.Label(sam_dev_frame, text="SAM 장치:").pack(side=tk.LEFT)
            self.sam_device_var = tk.StringVar(value="자동")
            sam_combo = ttk.Combobox(
                sam_dev_frame, textvariable=self.sam_device_var,
                values=["자동", "CPU", "CUDA"], state="readonly", width=5
            )
            sam_combo.pack(side=tk.LEFT, padx=(2, 0))
            ttk.Button(
                sam_dev_frame, text="재로드", width=5,
                command=self._on_reload_sam
            ).pack(side=tk.LEFT, padx=(2, 0))

        cols = ("name", "before", "after", "status", "out")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=16)
        self.tree.heading("name", text="파일명")
        self.tree.heading("before", text="전 용량")
        self.tree.heading("after", text="후 용량")
        self.tree.heading("status", text="상태")
        self.tree.heading("out", text="출력 경로")
        self.tree.column("name", width=240)
        self.tree.column("before", width=90, anchor=tk.E)
        self.tree.column("after", width=90, anchor=tk.E)
        self.tree.column("status", width=100, anchor=tk.CENTER)
        self.tree.column("out", width=300)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10)
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, **pad)

        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(fill=tk.X, expand=True)

        self.lbl_summary = ttk.Label(bottom, text="대기 중")
        self.lbl_summary.pack(anchor=tk.W)

        # 로그 영역
        log_frame = ttk.LabelFrame(self, text="로그")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        self.log_text = ScrolledText(log_frame, height=14, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        # Pillow 버전 정보 출력
        try:
            import PIL
            self._log(f"앱 시작 - Pillow {getattr(PIL, '__version__', 'unknown')}")
        except Exception:
            self._log("앱 시작")
        if HEIF_AVAILABLE:
            self._log("HEIC/HEIF 지원 활성화 (JPEG로 자동 변환)")
        else:
            self._log("HEIC/HEIF 지원 비활성화 - pillow-heif 설치 필요")

        # 전역 예외 훅: GUI에 출력
        def excepthook(exctype, value, tb):
            msg = "\n".join([str(value), "".join(traceback.format_exception(exctype, value, tb))])
            self._log(f"[Unhandled] {msg}")
            # 콘솔에도 출력 유지
            sys.__excepthook__(exctype, value, tb)
        sys.excepthook = excepthook

    def _log(self, text: str) -> None:
        try:
            self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
        except Exception:
            pass
        try:
            print(text)
        except Exception:
            pass

    def on_select(self) -> None:
        extensions = "*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tiff;*.tif"
        if HEIF_AVAILABLE:
            extensions += ";*.heic;*.heif"
        paths = filedialog.askopenfilenames(
            title="이미지 파일 선택",
            filetypes=[
                ("이미지 파일", extensions),
                ("모든 파일", "*.*"),
            ],
        )
        if not paths:
            return
        self.files = [Path(p) for p in paths]
        self._refresh_list_initial()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_enhance_only.config(state=tk.NORMAL)
        self.btn_open.config(state=tk.DISABLED)
        self.lbl_summary.config(text=f"선택됨: {len(self.files)}개")
        self.progress.configure(value=0, maximum=max(1, len(self.files)))
        self._log(f"선택: {len(self.files)}개 파일")

    def _refresh_list_initial(self) -> None:
        for i in self.tree.get_children():
            self.tree.delete(i)
        self._sam_temps.clear()
        self._file_index.clear()
        for idx, p in enumerate(self.files):
            try:
                before = format_bytes(p.stat().st_size)
            except Exception:
                before = "-"
            self.tree.insert("", tk.END, iid=str(idx), values=(p.name, before, "-", "대기", str(p.parent)))
            self._file_index[p] = idx

    def on_start(self) -> None:
        if not self.files:
            return
        profile = self._current_profile()
        self.btn_select.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_enhance_only.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.results = []
        self.progress.configure(value=0, maximum=max(1, len(self.files)))
        self.lbl_summary.config(text="처리 중...")
        apply_enhance = self.apply_enhance_var.get()
        mode = self.filter_mode_var.get()
        mode_text = "외부" if mode == "outdoor" else "내부"
        self._log(f"처리 시작 - {profile.label}" + (" + 보정" if apply_enhance else "") + f" (모드: {mode_text})")

        # SAM 임시 파일 적용: original → temp 매핑
        process_list = [self._sam_temps.get(p, p) for p in self.files]
        for proc_p, idx in [(process_list[i], i) for i in range(len(process_list))]:
            self._file_index[proc_p] = idx

        def run(active_profile: CompressionProfile = profile, do_enhance: bool = apply_enhance,
                filter_mode: str = mode, plist: List[Path] = process_list):
            total_before = 0
            total_after = 0
            for idx, p in enumerate(plist, start=1):
                res = compress_file_to_target(p, active_profile, apply_enhance=do_enhance, filter_mode=filter_mode)
                self.results.append(res)
                if res.before_bytes:
                    total_before += res.before_bytes
                if res.after_bytes:
                    total_after += res.after_bytes
                self.q.put(("progress", idx, res))
            self.q.put(("done", total_before, total_after))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def on_enhance_only(self) -> None:
        if not self.files:
            return
        self.btn_select.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_enhance_only.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.results = []
        self.progress.configure(value=0, maximum=max(1, len(self.files)))
        self.lbl_summary.config(text="보정 중...")
        mode = self.filter_mode_var.get()
        mode_text = "외부" if mode == "outdoor" else "내부"
        self._log(f"보정만 실행 (모드: {mode_text})")

        # SAM 임시 파일 적용
        process_list_e = [self._sam_temps.get(p, p) for p in self.files]
        for proc_p, idx in [(process_list_e[i], i) for i in range(len(process_list_e))]:
            self._file_index[proc_p] = idx

        def run_enhance(filter_mode: str = mode, plist: List[Path] = process_list_e):
            total_before = 0
            total_after = 0
            for idx, p in enumerate(plist, start=1):
                res = enhance_only_file(p, mode=filter_mode)
                self.results.append(res)
                if res.before_bytes:
                    total_before += res.before_bytes
                if res.after_bytes:
                    total_after += res.after_bytes
                self.q.put(("progress", idx, res))
            self.q.put(("done", total_before, total_after))

        self.worker = threading.Thread(target=run_enhance, daemon=True)
        self.worker.start()

    def _profile_summary(self) -> str:
        approx_mb = self.profile.target_bytes / BYTES_IN_MB
        heif_text = " + HEIC→JPG" if HEIF_AVAILABLE else ""
        return f"{self.profile.label}: ~{approx_mb:.1f}MB 목표{heif_text}"

    def _current_profile(self) -> CompressionProfile:
        return self.profile

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    idx, res = msg[1], msg[2]
                    self._update_row(res)
                    self.progress.configure(value=idx)
                    if res.status == "error":
                        self._log(f"[에러] {res.src_path.name}: {res.message}")
                    elif res.status == "compressed":
                        self._log(
                            f"[완료] {res.src_path.name}: {format_bytes(res.before_bytes)} → {format_bytes(res.after_bytes)}"
                        )
                    elif res.status == "enhanced":
                        self._log(f"[보정 완료] {res.src_path.name}: {format_bytes(res.before_bytes)} → {format_bytes(res.after_bytes)}")
                    elif res.status == "pass":
                        self._log(f"[통과] {res.src_path.name}: {format_bytes(res.before_bytes)}")
                elif kind == "done":
                    total_before, total_after = msg[1], msg[2]
                    saved = max(0, total_before - total_after)
                    self.lbl_summary.config(
                        text=f"총 {format_bytes(total_before)} → {format_bytes(total_after)} (절감: {format_bytes(saved)})"
                    )
                    self.btn_select.config(state=tk.NORMAL)
                    self.btn_start.config(state=tk.NORMAL)
                    self.btn_enhance_only.config(state=tk.NORMAL)
                    self.btn_open.config(state=tk.NORMAL)
                    # SAM 임시 파일 정리
                    for temp_p in list(self._sam_temps.values()):
                        try:
                            if temp_p.exists():
                                temp_p.unlink()
                        except Exception:
                            pass
                    self._sam_temps.clear()
                    self._log("처리 완료")
        except queue.Empty:
            pass
        finally:
            self.after(80, self._poll_queue)

    def _update_row(self, res: ResultItem) -> None:
        idx = self._file_index.get(res.src_path)
        if idx is None:
            # fallback: 파일명으로 검색
            for iid in self.tree.get_children():
                v = self.tree.item(iid, "values")
                if v and (v[0] == res.src_path.name or v[0] == f"[SAM] {res.src_path.stem.replace('_sam_temp', '')}.{res.src_path.suffix.lstrip('.')}"):
                    idx = int(iid)
                    break
        if idx is None:
            return
        iid = str(idx)
        vals = self.tree.item(iid, "values")
        if not vals:
            return
        before = format_bytes(res.before_bytes) if res.before_bytes else vals[1]
        after = format_bytes(res.after_bytes) if res.after_bytes else "-"
        status = {
            "pass": "통과",
            "compressed": "압축 완료",
            "enhanced": "보정 완료",
            "error": "에러",
        }.get(res.status, res.status)
        out = str(res.out_path) if res.out_path else vals[4]
        self.tree.item(iid, values=(vals[0], before, after, status, out))

    # ──────────────────── SAM 관련 메서드 ────────────────────────

    def _get_force_device(self) -> Optional[str]:
        """UI 선택값을 force_device 파라미터로 변환"""
        if not SAM_AVAILABLE or not hasattr(self, "sam_device_var"):
            return None
        val = self.sam_device_var.get()
        if val == "CPU":
            return "cpu"
        if val == "CUDA":
            return "cuda"
        return None  # 자동

    def _init_sam_model(self) -> None:
        """백그라운드 스레드에서 SAM 모델 로드"""
        if self.sam_manager is None:
            return
        self.after(0, lambda: self.lbl_sam.config(text="SAM: 로딩 중...", foreground="blue"))
        try:
            def on_progress(msg: str):
                self.after(0, lambda m=msg: self.lbl_sam.config(text=f"SAM: {m}"))
            status = self.sam_manager.load_model(
                on_progress=on_progress,
                force_device=self._get_force_device(),
            )
            self.after(0, lambda s=status: self.lbl_sam.config(
                text=s, foreground="green" if "cuda" in s.lower() else "darkorange"
            ))
            self._log(f"[SAM] {status}")
        except Exception as e:
            msg = f"SAM 로드 실패: {e}"
            self.after(0, lambda m=msg: self.lbl_sam.config(text=m[:50], foreground="red"))
            self._log(f"[SAM] {e}")

    def _on_reload_sam(self) -> None:
        """SAM 재로드 버튼 클릭"""
        if self.sam_manager is None:
            return
        threading.Thread(target=self._init_sam_model, daemon=True).start()

    def _on_tree_double_click(self, event) -> None:
        """트리뷰 더블클릭 → SAM 편집 창 열기"""
        if not SAM_AVAILABLE:
            messagebox.showinfo("SAM 미설치",
                "SAM 기능을 사용하려면 다음 패키지를 설치하세요:\n"
                "pip install mobile-sam torch torchvision opencv-python\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git")
            return
        if self.sam_manager is None or not self.sam_manager.is_loaded():
            messagebox.showinfo("SAM 로딩 중", "모델 로딩이 완료되지 않았습니다. 잠시 후 다시 시도하세요.")
            return

        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        try:
            idx = int(iid)
        except ValueError:
            return
        if idx >= len(self.files):
            return

        original_path = self.files[idx]
        EditWindow(self, original_path, self.sam_manager)

    def _receive_sam_result(self, original_path: Path, temp_path: Path) -> None:
        """SAM 편집 완료 → 메인 화면 트리뷰 업데이트"""
        idx = self._file_index.get(original_path)
        if idx is None:
            self._log(f"[SAM] 파일 인덱스를 찾을 수 없음: {original_path.name}")
            return

        # 기존 temp 파일이 있으면 삭제
        old_temp = self._sam_temps.get(original_path)
        if old_temp and old_temp.exists():
            try:
                old_temp.unlink()
            except Exception:
                pass

        self._sam_temps[original_path] = temp_path
        self._file_index[temp_path] = idx

        # 트리뷰 행 업데이트
        iid = str(idx)
        vals = self.tree.item(iid, "values")
        if vals:
            try:
                size_str = format_bytes(temp_path.stat().st_size)
            except Exception:
                size_str = vals[1]
            self.tree.item(iid, values=(
                f"[SAM] {original_path.name}",
                size_str, "-", "SAM 적용", vals[4]
            ))

        self._log(f"[SAM] {original_path.name} → SAM 처리 완료. '압축 실행'을 누르세요.")

    # ─────────────────────────────────────────────────────────────

    def on_open_folder(self) -> None:
        # 출력이 원본과 같은 폴더이므로 첫 파일의 폴더를 연다
        if not self.files:
            return
        folder = str(self.files[0].parent)
        try:
            if os.name == "nt":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":  # type: ignore[name-defined]
                os.system(f"open '{folder}'")
            else:
                os.system(f"xdg-open '{folder}'")
        except Exception as e:
            messagebox.showerror("오류", f"폴더 열기 실패: {e}")


def main() -> None:
    app = App()
    try:
        # Windows에서 DPI 스케일 선호
        app.tk.call('tk', 'scaling', 1.2)
    except Exception:
        pass
    app.mainloop()


if __name__ == "__main__":
    main()


