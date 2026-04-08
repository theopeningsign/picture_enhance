"""
PyInstaller 빌드 스크립트
실행: python make_exe.py
결과: dist/PicCompress/ 폴더 → PicCompress.exe 더블클릭으로 실행
"""
import os
import sys
import subprocess
from pathlib import Path


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except Exception:
        print("[info] PyInstaller 미설치 → 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def run_build() -> None:
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        ensure_pyinstaller()

    # ── 포함할 데이터 파일 ──
    # (원본경로, 번들 내 대상 폴더)
    sep = ";" if sys.platform == "win32" else ":"
    datas = [
        f"logo.png{sep}.",
        f"models/mobile_sam.pt{sep}models",
    ]

    hidden_imports = [
        # PIL
        "PIL._tkinter_finder",
        "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont", "PIL.ImageFilter",
        "PIL.ImageEnhance", "PIL.ImageOps", "PIL.ImageChops",
        "PIL.JpegImagePlugin", "PIL.PngImagePlugin", "PIL.WebPImagePlugin",
        "PIL.BmpImagePlugin", "PIL.TiffImagePlugin", "PIL.GifImagePlugin",
        # torch / torchvision
        "torch", "torchvision", "torchvision.transforms",
        "torch.nn", "torch.nn.functional",
        # SAM
        "segment_anything", "segment_anything.modeling",
        "mobile_sam", "mobile_sam.modeling",
        # numpy / cv2
        "numpy", "cv2",
        # tkinter
        "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
        # 기타
        "threading", "io", "pathlib", "json", "logging",
    ]

    collect_all = [
        "torch", "torchvision", "segment_anything", "mobile_sam",
        "PIL", "cv2", "numpy",
    ]

    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--name", "PicCompress",
        "--onedir",          # 폴더 형태 (torch 때문에 onefile보다 훨씬 빠름)
        "--windowed",        # 콘솔 창 숨기기
        "--icon", "logo.png" if (project_dir / "logo.png").exists() else "NONE",
    ]

    for d in datas:
        args += ["--add-data", d]

    for h in hidden_imports:
        args += ["--hidden-import", h]

    for c in collect_all:
        args += ["--collect-all", c]

    args.append("app.py")

    print("[info] 빌드 시작... (torch 포함이라 5~15분 소요될 수 있어요)")
    print(f"[info] 명령어: {' '.join(args[:6])} ...")
    result = subprocess.run(args)

    if result.returncode == 0:
        out = project_dir / "dist" / "PicCompress"
        print(f"\n[done] 빌드 완료!")
        print(f"[done] 실행 파일: {out / 'PicCompress.exe'}")
        print(f"[done] 폴더째 복사해서 배포하세요: {out}")
    else:
        print(f"\n[error] 빌드 실패 (exit {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    run_build()
