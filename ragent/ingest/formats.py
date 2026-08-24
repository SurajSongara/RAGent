"""What did the user actually upload, and which pipeline path does it take?

Extensions lie. Users rename `.doc` to `.pdf`, browsers save `.xlsx` as
`.xls`, and scanners emit `.tif` files that are really JPEGs inside. Detection
here leads with magic bytes and falls back to the extension only when the
content is genuinely ambiguous (plain text has no signature).

The important output is not the MIME type, it is the **family**, because family
decides the route through the ingest DAG:

    PDF      parse natively, OCR only the pages that need it
    IMAGE    no text layer exists by definition, so always OCR
    OFFICE   convert to PDF first, then follow the PDF path
    WEB      render to PDF first, then follow the PDF path
    FLOW     no pages and no geometry; text-offset provenance instead

That last family is the one with real consequences. A `.md` or `.csv` file has
no pages and no coordinates, so there is no pixel region to highlight. Rather
than fabricate a layout for it, FLOW documents carry character-offset
provenance and the viewer highlights a text range. Two honest provenance modes
beat one that quietly invents bounding boxes.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "FormatFamily",
    "ProvenanceMode",
    "DetectedFormat",
    "UnsupportedFormatError",
    "detect_format",
    "SUPPORTED_EXTENSIONS",
]


class FormatFamily(StrEnum):
    PDF = "pdf"
    IMAGE = "image"
    OFFICE = "office"
    WEB = "web"
    FLOW = "flow"


class ProvenanceMode(StrEnum):
    """How a citation from this document resolves back to its source."""

    #: Rendered page plus a normalised bounding box. The viewer draws a highlight.
    PAGED = "paged"
    #: Character offsets into the extracted text. The viewer highlights a range.
    FLOW = "flow"


class UnsupportedFormatError(ValueError):
    """Raised for content we cannot route. The document is quarantined, not retried."""


@dataclass(frozen=True, slots=True)
class DetectedFormat:
    mime: str
    family: FormatFamily
    extension: str
    label: str

    @property
    def provenance(self) -> ProvenanceMode:
        return ProvenanceMode.FLOW if self.family is FormatFamily.FLOW else ProvenanceMode.PAGED

    @property
    def needs_conversion(self) -> bool:
        """True when the file must be rendered to PDF before it can be parsed."""
        return self.family in (FormatFamily.OFFICE, FormatFamily.WEB)

    @property
    def always_ocr(self) -> bool:
        """Images have no text layer at all, so the confidence gate is skipped."""
        return self.family is FormatFamily.IMAGE


def _f(mime: str, family: FormatFamily, ext: str, label: str) -> DetectedFormat:
    return DetectedFormat(mime=mime, family=family, extension=ext, label=label)


# --------------------------------------------------------------- signatures
# Ordered longest-prefix first so a more specific signature wins.

_MAGIC: list[tuple[bytes, DetectedFormat]] = [
    (b"%PDF-", _f("application/pdf", FormatFamily.PDF, "pdf", "PDF")),
    (b"\x89PNG\r\n\x1a\n", _f("image/png", FormatFamily.IMAGE, "png", "PNG image")),
    (b"\xff\xd8\xff", _f("image/jpeg", FormatFamily.IMAGE, "jpg", "JPEG image")),
    (b"GIF87a", _f("image/gif", FormatFamily.IMAGE, "gif", "GIF image")),
    (b"GIF89a", _f("image/gif", FormatFamily.IMAGE, "gif", "GIF image")),
    (b"II*\x00", _f("image/tiff", FormatFamily.IMAGE, "tiff", "TIFF image")),
    (b"MM\x00*", _f("image/tiff", FormatFamily.IMAGE, "tiff", "TIFF image")),
    (b"BM", _f("image/bmp", FormatFamily.IMAGE, "bmp", "BMP image")),
    (b"{\\rtf", _f("application/rtf", FormatFamily.OFFICE, "rtf", "Rich Text")),
]

# Legacy Microsoft formats all share one OLE2 container signature, so the
# extension is the only thing that separates .doc from .xls from .ppt.
_OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_OLE2_BY_EXT: dict[str, DetectedFormat] = {
    "doc": _f("application/msword", FormatFamily.OFFICE, "doc", "Word 97-2003"),
    "xls": _f("application/vnd.ms-excel", FormatFamily.OFFICE, "xls", "Excel 97-2003"),
    "ppt": _f("application/vnd.ms-powerpoint", FormatFamily.OFFICE, "ppt", "PowerPoint 97-2003"),
}

# OOXML and ODF are both ZIP containers; the internal layout tells them apart.
_ZIP_ENTRY_MARKERS: list[tuple[str, DetectedFormat]] = [
    (
        "word/",
        _f(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            FormatFamily.OFFICE,
            "docx",
            "Word document",
        ),
    ),
    (
        "xl/",
        _f(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            FormatFamily.OFFICE,
            "xlsx",
            "Excel workbook",
        ),
    ),
    (
        "ppt/",
        _f(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            FormatFamily.OFFICE,
            "pptx",
            "PowerPoint deck",
        ),
    ),
]

_ODF_MIMETYPES: dict[str, DetectedFormat] = {
    "application/vnd.oasis.opendocument.text": _f(
        "application/vnd.oasis.opendocument.text", FormatFamily.OFFICE, "odt", "OpenDocument Text"
    ),
    "application/vnd.oasis.opendocument.spreadsheet": _f(
        "application/vnd.oasis.opendocument.spreadsheet",
        FormatFamily.OFFICE,
        "ods",
        "OpenDocument Sheet",
    ),
    "application/vnd.oasis.opendocument.presentation": _f(
        "application/vnd.oasis.opendocument.presentation",
        FormatFamily.OFFICE,
        "odp",
        "OpenDocument Slides",
    ),
}

# Text-ish formats have no signature worth trusting, so these resolve by
# extension after the binary checks have all declined.
_BY_EXTENSION: dict[str, DetectedFormat] = {
    "pdf": _MAGIC[0][1],
    "png": _f("image/png", FormatFamily.IMAGE, "png", "PNG image"),
    "jpg": _f("image/jpeg", FormatFamily.IMAGE, "jpg", "JPEG image"),
    "jpeg": _f("image/jpeg", FormatFamily.IMAGE, "jpg", "JPEG image"),
    "tif": _f("image/tiff", FormatFamily.IMAGE, "tiff", "TIFF image"),
    "tiff": _f("image/tiff", FormatFamily.IMAGE, "tiff", "TIFF image"),
    "webp": _f("image/webp", FormatFamily.IMAGE, "webp", "WebP image"),
    "bmp": _f("image/bmp", FormatFamily.IMAGE, "bmp", "BMP image"),
    "html": _f("text/html", FormatFamily.WEB, "html", "HTML page"),
    "htm": _f("text/html", FormatFamily.WEB, "html", "HTML page"),
    "md": _f("text/markdown", FormatFamily.FLOW, "md", "Markdown"),
    "markdown": _f("text/markdown", FormatFamily.FLOW, "md", "Markdown"),
    "txt": _f("text/plain", FormatFamily.FLOW, "txt", "Plain text"),
    "text": _f("text/plain", FormatFamily.FLOW, "txt", "Plain text"),
    "log": _f("text/plain", FormatFamily.FLOW, "txt", "Plain text"),
    "csv": _f("text/csv", FormatFamily.FLOW, "csv", "CSV"),
    "tsv": _f("text/tab-separated-values", FormatFamily.FLOW, "tsv", "TSV"),
    "json": _f("application/json", FormatFamily.FLOW, "json", "JSON"),
    "xml": _f("application/xml", FormatFamily.FLOW, "xml", "XML"),
    **_OLE2_BY_EXT,
    "docx": _ZIP_ENTRY_MARKERS[0][1],
    "xlsx": _ZIP_ENTRY_MARKERS[1][1],
    "pptx": _ZIP_ENTRY_MARKERS[2][1],
    "rtf": _f("application/rtf", FormatFamily.OFFICE, "rtf", "Rich Text"),
}

# ODF types resolve from the container's declared mimetype rather than the
# extension table, so they are added explicitly.
SUPPORTED_EXTENSIONS = frozenset(_BY_EXTENSION) | {"odt", "ods", "odp"}


def _extension_of(filename: str) -> str:
    _, _, ext = filename.rpartition(".")
    return ext.lower() if ext and ext != filename else ""


def _sniff_zip(data: bytes, extension: str) -> DetectedFormat | None:
    """Look inside a ZIP container to tell OOXML and ODF apart."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            # ODF states its type in an uncompressed `mimetype` member.
            if "mimetype" in names:
                declared = zf.read("mimetype").decode("ascii", "ignore").strip()
                if declared in _ODF_MIMETYPES:
                    return _ODF_MIMETYPES[declared]
            for marker, fmt in _ZIP_ENTRY_MARKERS:
                if any(name.startswith(marker) for name in names):
                    return fmt
    except (zipfile.BadZipFile, KeyError, OSError):
        return None
    # A ZIP we cannot classify is not an error yet; the extension may still say.
    return _BY_EXTENSION.get(extension)


def _looks_like_html(head: bytes) -> bool:
    probe = head[:1024].lstrip().lower()
    return probe.startswith((b"<!doctype html", b"<html", b"<?xml-stylesheet", b"<head"))


def _looks_like_text(data: bytes) -> bool:
    """Decodable as UTF-8 and free of control bytes that only appear in binaries."""
    sample = data[:8192]
    if b"\x00" in sample:
        return False
    try:
        decoded = sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    # Tab, newline and carriage return are the only control codes text should use.
    control = sum(1 for ch in decoded if ord(ch) < 32 and ch not in "\t\n\r")
    return control / max(len(decoded), 1) < 0.02


def detect_format(data: bytes, filename: str = "") -> DetectedFormat:
    """Identify uploaded content. Magic bytes win; the extension only breaks ties.

    Raises:
        UnsupportedFormatError: content we have no route for. Callers should
            quarantine the document rather than retry it — a retry cannot change
            what the bytes are.
    """
    if not data:
        raise UnsupportedFormatError("empty file")

    extension = _extension_of(filename)

    for signature, fmt in _MAGIC:
        if data.startswith(signature):
            # A TIFF signature with a .jpg name is still a TIFF; trust the bytes.
            return fmt

    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return _BY_EXTENSION["webp"]

    if data.startswith(b"PK\x03\x04"):
        sniffed = _sniff_zip(data, extension)
        if sniffed is not None:
            return sniffed
        raise UnsupportedFormatError("ZIP archive that is not a recognised document")

    if data.startswith(_OLE2):
        legacy = _OLE2_BY_EXT.get(extension)
        if legacy is not None:
            return legacy
        # Container is real but the extension does not say which application.
        raise UnsupportedFormatError(
            "legacy Microsoft container with an unrecognised extension "
            f"{extension!r}; expected one of {sorted(_OLE2_BY_EXT)}"
        )

    if _looks_like_html(data):
        return _BY_EXTENSION["html"]

    if _looks_like_text(data):
        # Extension distinguishes csv from md from txt; they route identically
        # but the parse stage treats them differently.
        by_ext = _BY_EXTENSION.get(extension)
        if by_ext is not None and by_ext.family is FormatFamily.FLOW:
            return by_ext
        return _BY_EXTENSION["txt"]

    raise UnsupportedFormatError(f"unrecognised content{f' for .{extension}' if extension else ''}")
