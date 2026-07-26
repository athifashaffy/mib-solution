"""PDF text extraction: spans, visibility filtering, and OCR fallback.

Documents in this challenge may contain hidden text spans embedded as a
prompt-injection defense against downstream LLM adjudication (e.g. white-on-
white text instructing a reader to "ignore visible evidence"). This module
separates each page's text into `visible` and `hidden` buckets using a fixed,
deterministic visibility rule so callers never have to rely on rendered
color at adjudication time - hidden spans are surfaced (for logging/risk
flags) but never treated as trustworthy page content.

Some pages carry no extractable text layer at all (scanned/rasterized
images). Those are detected as `image_only` and OCR'd via local tesseract as
a fallback so the pipeline can still recover a best-effort text signal.
"""
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz

# A span is hidden iff its rendered luminance exceeds this threshold (i.e. it
# is at or near white and would be invisible against a white page background).
LUMINANCE_HIDDEN_THRESHOLD = 240

# A span is hidden iff its font size is below this point size (unreadably
# small to a human reader).
MIN_VISIBLE_FONT_SIZE = 3

# A page whose visible text layer has fewer than this many non-whitespace
# characters is a candidate for being image_only (subject to also having at
# least one embedded image).
IMAGE_ONLY_VISIBLE_CHAR_THRESHOLD = 100

# OCR is retried with a different tesseract page-segmentation mode when the
# first pass yields fewer than this many non-whitespace characters.
OCR_MIN_CHARS = 50

OCR_DPI = 200
OCR_TIMEOUT_SECONDS = 20


@dataclass
class PageText:
    visible: str
    hidden: str
    ocr: str | None
    image_only: bool


@dataclass
class DocText:
    pages: list[PageText] = field(default_factory=list)


def _luminance(color_int: int) -> float:
    """Perceptual luminance of a packed 0xRRGGBB integer color, per
    PyMuPDF's span["color"] convention (0 - 16777215)."""
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return 0.299 * r + 0.587 * g + 0.114 * b


def _is_span_hidden(span: dict, page_rect: "fitz.Rect") -> bool:
    """A span is hidden iff its color is near-white, its font is too small
    to read, or its bbox lies fully outside the page cropbox."""
    color = span.get("color", 0)
    if _luminance(color) > LUMINANCE_HIDDEN_THRESHOLD:
        return True

    size = span.get("size", 0)
    if size < MIN_VISIBLE_FONT_SIZE:
        return True

    bbox = fitz.Rect(span["bbox"])
    if (bbox & page_rect).is_empty:
        return True

    return False


def _split_page_spans(page: "fitz.Page") -> tuple[str, str]:
    """Walk page.get_text('dict') blocks/lines/spans in reading order and
    split span text into visible vs hidden line-joined strings.

    Spans within a line are concatenated as-is (no separator inserted,
    matching how PyMuPDF's own plain get_text() abuts adjacent spans);
    lines are joined with "\\n" so downstream label/value parsing (which
    expects each field on its own line) keeps working.
    """
    page_rect = page.cropbox
    visible_lines: list[str] = []
    hidden_lines: list[str] = []

    # PyMuPDF's get_text() clips to the page rect by default and silently
    # drops spans placed fully outside it - which would make the "bbox fully
    # outside cropbox" hidden-check unreachable. Extract with a generously
    # expanded clip so any off-page span is still handed to us, and let
    # _is_span_hidden() be the one place that decides visibility.
    huge_clip = page_rect + (-100000, -100000, 100000, 100000)
    page_dict = page.get_text("dict", clip=huge_clip)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            # Non-text (e.g. image) block; nothing to extract here.
            continue
        for line in block.get("lines", []):
            visible_parts: list[str] = []
            hidden_parts: list[str] = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                if _is_span_hidden(span, page_rect):
                    hidden_parts.append(text)
                else:
                    visible_parts.append(text)
            if visible_parts:
                visible_lines.append("".join(visible_parts))
            if hidden_parts:
                hidden_lines.append("".join(hidden_parts))

    return "\n".join(visible_lines), "\n".join(hidden_lines)


def _run_tesseract(png_path: str, psm: str) -> str:
    try:
        result = subprocess.run(
            ["tesseract", png_path, "stdout", "--psm", psm, "-l", "eng"],
            capture_output=True,
            text=True,
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout or ""


def _ocr_page(page: "fitz.Page") -> str:
    """Render the page at OCR_DPI grayscale and run local tesseract on it.

    Tries --psm 6 first; if the result has fewer than OCR_MIN_CHARS
    non-whitespace characters, retries with --psm 3 and keeps whichever
    result is longer. Any subprocess failure/timeout yields "" for that
    attempt rather than raising.
    """
    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csGRAY)

    fd = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    png_path = fd.name
    fd.close()
    try:
        pix.save(png_path)
        text = _run_tesseract(png_path, "6")
        if len(text.strip()) < OCR_MIN_CHARS:
            retry_text = _run_tesseract(png_path, "3")
            if len(retry_text) > len(text):
                text = retry_text
        return text
    finally:
        Path(png_path).unlink(missing_ok=True)


def extract_document(pdf_path) -> DocText:
    """Open pdf_path and extract per-page visible/hidden text, flagging and
    OCR'ing any image-only pages. Raises if the PDF can't be opened/parsed;
    callers are responsible for fallback handling of corrupt inputs."""
    pages: list[PageText] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            visible_text, hidden_text = _split_page_spans(page)
            has_images = bool(page.get_images())
            image_only = (
                len(visible_text.strip()) < IMAGE_ONLY_VISIBLE_CHAR_THRESHOLD
                and has_images
            )
            ocr_text = _ocr_page(page) if image_only else None
            pages.append(
                PageText(
                    visible=visible_text,
                    hidden=hidden_text,
                    ocr=ocr_text,
                    image_only=image_only,
                )
            )
    finally:
        doc.close()
    return DocText(pages=pages)
