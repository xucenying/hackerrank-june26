"""Image loading and base64 encoding for the VLM pipeline. Stdlib only (pathlib + base64)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

# Matches the separator used for image_paths, risk_flags, and supporting_image_ids.
LIST_SEPARATOR = ";"

MEDIA_TYPE_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _sniff_media_type(data: bytes) -> str | None:
    """Detect the real image format from its magic bytes.

    File extensions in this dataset are not always trustworthy (e.g. some
    ".jpg" files are actually WebP), and the Anthropic API validates the
    declared media_type against the actual bytes -- a mismatch is a 400.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@dataclass
class LoadedImage:
    """One successfully loaded and base64-encoded image."""

    image_id: str  # filename without extension, e.g. "img_1"
    path: Path
    media_type: str  # e.g. "image/jpeg"
    base64_data: str


@dataclass
class ImageLoadResult:
    """Result of loading every path in a claim's image_paths field."""

    valid_images: list[LoadedImage]
    invalid_paths: list[str]  # raw path strings that were missing, unreadable, or unsupported


def image_id_from_path(path: Path) -> str:
    return path.stem


def split_image_paths(image_paths: str) -> list[str]:
    return [p.strip() for p in image_paths.split(LIST_SEPARATOR) if p.strip()]


def load_image(path: Path) -> LoadedImage:
    """Load and base64-encode a single image.

    Raises FileNotFoundError, OSError, or ValueError (unrecognized image format)
    if the image can't be read. Callers treat any of these as an invalid path.
    """
    data = path.read_bytes()
    media_type = _sniff_media_type(data) or MEDIA_TYPE_BY_SUFFIX.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(f"Unrecognized image format: {path}")
    encoded = base64.b64encode(data).decode("ascii")
    return LoadedImage(
        image_id=image_id_from_path(path),
        path=path,
        media_type=media_type,
        base64_data=encoded,
    )


def load_claim_images(image_paths: str, base_dir: Path) -> ImageLoadResult:
    """Load every image referenced by a claim's semicolon-separated image_paths field.

    Paths in image_paths are relative to the dataset directory (e.g.
    "images/test/case_001/img_1.jpg"), so each is resolved against base_dir.
    """
    valid_images: list[LoadedImage] = []
    invalid_paths: list[str] = []
    for raw_path in split_image_paths(image_paths):
        full_path = base_dir / raw_path
        try:
            valid_images.append(load_image(full_path))
        except (FileNotFoundError, OSError, ValueError):
            invalid_paths.append(raw_path)
    return ImageLoadResult(valid_images=valid_images, invalid_paths=invalid_paths)
