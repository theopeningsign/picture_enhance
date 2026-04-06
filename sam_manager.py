"""
SAM 모델 관리 모듈
- MobileSAM (mobile_sam.pt) 우선 사용, 없으면 SAM ViT-H (sam_vit_h_4b8939.pth) 사용
- CUDA OOM 발생 시 CPU fallback 자동 처리
"""
from __future__ import annotations

import gc
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# 모델 파일 경로
MODELS_DIR = Path(__file__).parent / "models"
MOBILE_SAM_PATH = MODELS_DIR / "mobile_sam.pt"
VIT_H_SAM_PATH = MODELS_DIR / "sam_vit_h_4b8939.pth"


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _try_import_mobile_sam():
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        return sam_model_registry, SamPredictor
    except ImportError:
        return None, None


def _try_import_segment_anything():
    try:
        from segment_anything import sam_model_registry, SamPredictor
        return sam_model_registry, SamPredictor
    except ImportError:
        return None, None


class SAMManager:
    """SAM 모델 로딩 및 예측 관리 클래스"""

    def __init__(self):
        self.sam = None
        self.predictor = None
        self.device: str = "cpu"
        self.model_name: str = "없음"
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def get_status(self) -> str:
        """UI 표시용 상태 문자열 반환"""
        if not self._loaded:
            return "SAM: 미로딩"
        return f"SAM: {self.model_name} ({self.device.upper()})"

    def load_model(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
        force_device: Optional[str] = None,  # None=자동, "cpu"=CPU 강제, "cuda"=CUDA 강제
    ) -> str:
        """
        사용 가능한 모델을 자동 감지하여 로드.
        force_device: None=자동(CUDA 우선), "cpu"=CPU 강제, "cuda"=CUDA 강제
        Returns: 상태 메시지
        Raises: RuntimeError if no model available
        """
        # 기존 모델 언로드
        if self._loaded:
            self.unload()

        torch = _try_import_torch()
        if torch is None:
            raise RuntimeError("torch가 설치되지 않았습니다. pip install torch torchvision")

        # CUDA 강제 요청인데 사용 불가 시 에러
        if force_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 드라이버와 CUDA 버전을 확인하세요.")

        self._force_device = force_device  # _load()에서 참조

        def _log(msg: str):
            if on_progress:
                on_progress(msg)

        # MobileSAM 우선 시도
        if MOBILE_SAM_PATH.exists():
            registry, Predictor = _try_import_mobile_sam()
            if registry is not None:
                _log("MobileSAM 로딩 중...")
                self._load(torch, registry, Predictor, "vit_t", MOBILE_SAM_PATH, "MobileSAM")
                return self.get_status()
            else:
                _log("mobile-sam 패키지 없음. pip install mobile-sam 필요. SAM ViT-H로 전환.")

        # SAM ViT-H fallback
        if VIT_H_SAM_PATH.exists():
            registry, Predictor = _try_import_segment_anything()
            if registry is not None:
                _log("SAM ViT-H 로딩 중... (큰 모델, 수초 소요)")
                self._load(torch, registry, Predictor, "vit_h", VIT_H_SAM_PATH, "SAM ViT-H")
                return self.get_status()
            else:
                raise RuntimeError(
                    "segment-anything 패키지 없음.\n"
                    "pip install git+https://github.com/facebookresearch/segment-anything.git"
                )

        raise RuntimeError(
            f"모델 파일을 찾을 수 없습니다.\n"
            f"  MobileSAM: {MOBILE_SAM_PATH}\n"
            f"  SAM ViT-H: {VIT_H_SAM_PATH}"
        )

    def _load(self, torch, registry, Predictor, model_type: str, model_path: Path, name: str):
        """모델 로드. force_device에 따라 장치 결정, OOM 시 CPU fallback."""
        sam = registry[model_type](checkpoint=str(model_path))
        force = getattr(self, "_force_device", None)

        if force == "cpu":
            sam.to("cpu")
            self.device = "cpu"
        elif force == "cuda":
            sam.to("cuda")  # CUDA 불가면 load_model에서 이미 에러 처리됨
            self.device = "cuda"
        else:
            # 자동: CUDA 우선, OOM 시 CPU fallback
            if torch.cuda.is_available():
                try:
                    sam.to("cuda")
                    self.device = "cuda"
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        sam.to("cpu")
                        self.device = "cpu"
                    else:
                        raise
            else:
                sam.to("cpu")
                self.device = "cpu"

        sam.eval()
        self.sam = sam
        self.predictor = Predictor(sam)
        self.model_name = name
        self._loaded = True

    def set_image(self, image_rgb: np.ndarray):
        """예측 전 이미지 설정 (RGB numpy uint8)"""
        if not self._loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        self.predictor.set_image(image_rgb)

    def predict(
        self,
        pos_points: list[tuple[int, int]],
        neg_points: list[tuple[int, int]],
    ) -> Optional[np.ndarray]:
        """
        포인트 기반 마스크 예측.
        pos_points: 포함 포인트 리스트 [(x, y), ...]
        neg_points: 제외 포인트 리스트 [(x, y), ...]
        Returns: bool mask (H, W) or None
        """
        if not self._loaded:
            return None
        if not pos_points and not neg_points:
            return None

        all_points = list(pos_points) + list(neg_points)
        all_labels = [1] * len(pos_points) + [0] * len(neg_points)

        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(all_labels, dtype=np.int32)

        try:
            # 점 1개: 3개 후보 중 최선 선택 / 2개 이상: 단일 마스크(안정적)
            multi = (len(all_points) == 1)
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multi,
            )
            best_idx = int(np.argmax(scores))
            return masks[best_idx].astype(bool)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch = _try_import_torch()
                if torch:
                    torch.cuda.empty_cache()
            raise

    def predict_box(
        self,
        box: tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        박스 기반 마스크 예측 (포인트보다 훨씬 안정적).
        box: (x1, y1, x2, y2) in image coordinates
        Returns: bool mask (H, W) or None
        """
        if not self._loaded:
            return None

        box_arr = np.array(list(box), dtype=np.float32)

        try:
            masks, scores, _ = self.predictor.predict(
                box=box_arr,
                multimask_output=False,
            )
            return masks[0].astype(bool)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch = _try_import_torch()
                if torch:
                    torch.cuda.empty_cache()
            raise

    def unload(self):
        """모델 언로드 및 VRAM 해제"""
        self.predictor = None
        self.sam = None
        self._loaded = False
        torch = _try_import_torch()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
