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

from .parse import (
    has_fee_value,
    has_finding_sentence,
    has_observed_flags_line,
    looks_like_biometric,
    looks_like_fee,
    looks_like_note,
)

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
# Retry DPI for pages that look like adjudicator notes whose Finding
# sentence didn't survive the first OCR pass. Notes carry the single most
# decision-relevant sentence in a packet, and the runtime budget has ~28x
# headroom, so the extra render is cheap where it matters.
OCR_NOTE_RETRY_DPI = 300
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


def _composited_luminance(color_int: int, alpha: int) -> float:
    """Perceptual luminance of a packed 0xRRGGBB color after alpha-
    compositing it against an opaque white page background.

    PyMuPDF reports span["alpha"] in [0, 255] (0 == fully transparent, e.g.
    text drawn with the invisible PDF render mode `3 Tr`). Fully transparent
    text renders as pure white regardless of its nominal color, so it must
    be treated as hidden the same as literal white text.
    """
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    factor = alpha / 255.0
    cr = factor * r + (1 - factor) * 255
    cg = factor * g + (1 - factor) * 255
    cb = factor * b + (1 - factor) * 255
    return 0.299 * cr + 0.587 * cg + 0.114 * cb


def _is_span_hidden(span: dict, page_rect: "fitz.Rect") -> bool:
    """A span is hidden iff its alpha-composited (against a white page
    background) color is near-white, its font is too small to read, or its
    bbox lies fully outside the page cropbox."""
    color = span.get("color", 0)
    alpha = span.get("alpha", 255)
    if _composited_luminance(color, alpha) > LUMINANCE_HIDDEN_THRESHOLD:
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


def _non_whitespace_count(text: str) -> int:
    """Count of non-whitespace characters in text."""
    return sum(1 for c in text if not c.isspace())


def _needs_ocr_retry(text: str) -> bool:
    """True iff the first-pass OCR text has fewer than OCR_MIN_CHARS
    non-whitespace characters, warranting a retry with a different
    tesseract page-segmentation mode."""
    return _non_whitespace_count(text) < OCR_MIN_CHARS


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


def _ocr_at_dpi(page: "fitz.Page", dpi: int) -> str:
    """Render the page at `dpi` grayscale and run local tesseract on it.

    Tries --psm 6 first; if the result has fewer than OCR_MIN_CHARS
    non-whitespace characters, retries with --psm 3 and keeps whichever
    result is longer. Any subprocess failure/timeout yields "" for that
    attempt rather than raising.
    """
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)

    fd = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    png_path = fd.name
    fd.close()
    try:
        pix.save(png_path)
        text = _run_tesseract(png_path, "6")
        if _needs_ocr_retry(text):
            retry_text = _run_tesseract(png_path, "3")
            if len(retry_text) > len(text):
                text = retry_text
        return text
    finally:
        Path(png_path).unlink(missing_ok=True)


# (looks-like hint, recovered-the-goal check) pairs for the targeted
# higher-DPI retry: a page matching a hint whose goal content didn't
# survive the first OCR pass gets one 300-dpi retry, adopted only when the
# retry actually recovers that content — so a retry can never make a page
# worse. Notes carry the decision sentence, fee receipts the fee status,
# biometric slips the observed risk flags.
_RETRY_GOALS = (
    (looks_like_note, has_finding_sentence),
    (looks_like_fee, has_fee_value),
    (looks_like_biometric, has_observed_flags_line),
)


def _ocr_page(page: "fitz.Page") -> str:
    """First-pass OCR at OCR_DPI, plus at most one targeted higher-DPI
    retry when the page looks like a doc type whose key content didn't
    survive the first pass (see _RETRY_GOALS)."""
    text = _ocr_at_dpi(page, OCR_DPI)
    for looks_like, recovered in _RETRY_GOALS:
        if looks_like(text) and not recovered(text):
            retry_text = _ocr_at_dpi(page, OCR_NOTE_RETRY_DPI)
            if recovered(retry_text):
                return retry_text
            break
    return text


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
