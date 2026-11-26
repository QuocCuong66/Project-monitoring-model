"""Đóng gói MÃ NGUỒN dự án thành dist/Fall_and_Stroke_Warning_System_Source_v<version>.zip.

Gồm toàn bộ file được git quản lý (theo trạng thái hiện tại của thư mục làm việc,
kể cả file mới chưa commit nhưng không bị .gitignore) + các mô hình ONNX bắt buộc
để chạy được ngay bằng 1_Cai_Dat_Tu_Dong.bat / 2_Chay_He_Thong.bat.
"""

import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DIST_DIR = ROOT_DIR / "dist"
PACKAGE_NAME = "Fall_and_Stroke_Warning_System"

# Mô hình bị .gitignore nhưng pipeline cần để chạy
REQUIRED_MODELS = [
    "models/pose/rtmo-s.onnx",
    "models/checkpoints/best_st_transformer.onnx",
    "models/checkpoints/best_pose_autoencoder.onnx",
]

# Không đưa vào gói mã nguồn
EXCLUDED_PREFIXES = (".claude/", ".gemini/", "dist/", "build/")
EXCLUDED_SUFFIXES = (".zip",)


def project_version() -> str:
    with open(ROOT_DIR / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def list_source_files() -> list[Path]:
    """Danh sách file mã nguồn theo git (tracked + untracked không bị ignore, bỏ file đã xóa)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT_DIR, capture_output=True, check=True,
    ).stdout.decode("utf-8")
    files: list[Path] = []
    for rel in filter(None, out.split("\0")):
        if rel.startswith(EXCLUDED_PREFIXES) or rel.endswith(EXCLUDED_SUFFIXES):
            continue
        path = ROOT_DIR / rel
        if path.is_file():  # bỏ qua file đã bị xóa khỏi thư mục làm việc
            files.append(path)
    return files


def package_source() -> Path:
    version = project_version()
    DIST_DIR.mkdir(exist_ok=True)
    output_zip = DIST_DIR / f"{PACKAGE_NAME}_Source_v{version}.zip"

    files = list_source_files()
    for rel in REQUIRED_MODELS:
        model = ROOT_DIR / rel
        if not model.exists():
            sys.exit(f"[LỖI] Thiếu mô hình bắt buộc: {rel}")
        files.append(model)

    print("=" * 70)
    print("    FALL AND STROKE WARNING SYSTEM — ĐÓNG GÓI MÃ NGUỒN")
    print("=" * 70)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for path in sorted(set(files)):
            rel_path = path.relative_to(ROOT_DIR).as_posix()
            zipf.write(path, arcname=f"{PACKAGE_NAME}_Source/{rel_path}")

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f" [THÀNH CÔNG] {output_zip.relative_to(ROOT_DIR)}  ({len(set(files))} file, {size_mb:.1f} MB)")
    print(" Người nhận giải nén, chạy 1_Cai_Dat_Tu_Dong.bat rồi 2_Chay_He_Thong.bat.")
    print("=" * 70)
    return output_zip


if __name__ == "__main__":
    package_source()
