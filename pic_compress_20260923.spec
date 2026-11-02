# -*- mode: python ; coding: utf-8 -*-
#
# 2026-09-23 빌드 (다각형 영역 선택 / 휠 확대 / 재편집 복원 포함)
#
# 기존 PicCompress.spec 과 내용은 같고 산출물 이름만 다르다.
# → dist/pic_compress_20260923/ 로 나가므로 dist/PicCompress/ 는 건드리지 않는다.
from PyInstaller.utils.hooks import collect_all

datas = [('logo.png', '.'), ('models/mobile_sam.pt', 'models')]
binaries = []
hiddenimports = ['PIL._tkinter_finder', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont', 'PIL.ImageFilter', 'PIL.ImageEnhance', 'PIL.ImageOps', 'PIL.ImageChops', 'PIL.JpegImagePlugin', 'PIL.PngImagePlugin', 'PIL.WebPImagePlugin', 'PIL.BmpImagePlugin', 'PIL.TiffImagePlugin', 'PIL.GifImagePlugin', 'torch', 'torchvision', 'torchvision.transforms', 'torch.nn', 'torch.nn.functional', 'segment_anything', 'segment_anything.modeling', 'mobile_sam', 'mobile_sam.modeling', 'numpy', 'cv2', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'threading', 'io', 'pathlib', 'json', 'logging']
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('segment_anything')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mobile_sam')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('PIL')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pic_compress_20260923',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pic_compress_20260923',
)
