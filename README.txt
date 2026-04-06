이미지 5MB 압축기 (tkinter + Pillow)

- 5MB 이하 파일은 통과, 초과 파일은 약 4.9MB 목표로 자동 압축
- 여러 장 일괄 처리, 전/후 용량 표시, 같은 폴더에 _compressed.jpg 저장
- GUI: tkinter, 빌드: PyInstaller

사용법 (개발 환경)
1) 가상환경 생성 및 실행
   python -m venv .venv
   .venv\Scripts\activate
2) 의존성 설치
   pip install -r requirements.txt
3) 실행
   python app.py

빌드 (Windows, EXE)
   pip install pyinstaller
   pyinstaller --noconfirm --onefile --windowed --name PicCompressor app.py
   (생성물: dist/PicCompressor.exe)

동작 방식
1) 5MB 초과 여부 확인
2) PNG/WebP 등 알파 채널은 흰색 배경으로 합성 후 RGB로 변환
3) JPEG 품질 이진 탐색으로 목표(약 4.9MB)에 맞춤
4) 여전히 5MB 초과 시 이미지 크기를 10%씩 축소하며 재탐색
5) 결과는 원본 폴더에 파일명_compressed.jpg로 저장

주의
- 원본 파일은 변경하지 않습니다. 5MB 이하 파일은 파일 생성 없이 "통과" 처리합니다.
- EXIF 회전은 자동 보정됩니다.








