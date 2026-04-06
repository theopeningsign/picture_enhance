import os
import sys
import subprocess
from pathlib import Path


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except Exception:
        print("[info] PyInstaller 미설치 → 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])  # noqa: S603,S607


def run_build() -> None:
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    spec_path = project_dir / "make_5M_pic.spec"
    target = project_dir / "dist" / "make_5M_pic"

    ensure_pyinstaller()

    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
    ]

    if spec_path.exists():
        args.append(str(spec_path))
    else:
        # 스펙이 없다면 직접 옵션을 지정해 단일 파일로 빌드
        args += [
            "--onefile",
            "--windowed",
            "--name",
            "make_5M_pic",
            "--hidden-import=PIL.Image",
            "--hidden-import=PIL.JpegImagePlugin",
            "--hidden-import=PIL.PngImagePlugin",
            "--hidden-import=PIL.WebPImagePlugin",
            "--hidden-import=PIL.BmpImagePlugin",
            "--hidden-import=PIL.TiffImagePlugin",
            "--hidden-import=PIL._tkinter_finder",
            "--hidden-import=pillow_heif",
            "app.py",
        ]

    print("[info] 빌드 시작...")
    subprocess.check_call(args)  # noqa: S603
    print(f"[done] 빌드 완료 → {target if target.exists() else project_dir/'dist'}")


if __name__ == "__main__":
    try:
        run_build()
    except subprocess.CalledProcessError as e:
        print(f"[error] 빌드 실패 (exit {e.returncode}). 콘솔 로그를 확인하세요.")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)






