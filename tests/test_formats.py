"""Format detection.

Extensions lie, so every test here that pits bytes against a filename asserts
that the bytes win. Misrouting is expensive in a specific way: a `.docx` sent
down the PDF path fails deep inside the parser with an unhelpful error, while a
PDF sent to the converter silently re-renders and loses its text layer.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from ragent.ingest.formats import (
    FormatFamily,
    ProvenanceMode,
    UnsupportedFormatError,
    detect_format,
)

PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj"
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
TIFF_LE = b"II*\x00\x08\x00\x00\x00"
TIFF_BE = b"MM\x00*\x00\x00\x00\x08"
GIF = b"GIF89a\x01\x00\x01\x00"
BMP = b"BM\x36\x00\x00\x00\x00\x00"
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 "
OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 16
RTF = b"{\\rtf1\\ansi\\deff0"


def ooxml(prefix: str) -> bytes:
    """Minimal OOXML container: a ZIP whose internal layout names the app."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr(f"{prefix}/document.xml", "<w:document/>")
    return buf.getvalue()


def odf(mimetype: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", mimetype)
        zf.writestr("content.xml", "<office/>")
    return buf.getvalue()


class TestPdf:
    def test_detects_pdf(self) -> None:
        fmt = detect_format(PDF, "filing.pdf")
        assert fmt.family is FormatFamily.PDF
        assert fmt.mime == "application/pdf"
        assert fmt.provenance is ProvenanceMode.PAGED
        assert fmt.needs_conversion is False
        assert fmt.always_ocr is False

    def test_detects_pdf_despite_wrong_extension(self) -> None:
        assert detect_format(PDF, "actually-a-pdf.docx").family is FormatFamily.PDF


class TestImages:
    @pytest.mark.parametrize(
        ("data", "ext"),
        [
            (PNG, "png"),
            (JPEG, "jpg"),
            (TIFF_LE, "tiff"),
            (TIFF_BE, "tiff"),
            (GIF, "gif"),
            (BMP, "bmp"),
            (WEBP, "webp"),
        ],
    )
    def test_detects_image_types(self, data: bytes, ext: str) -> None:
        fmt = detect_format(data, f"scan.{ext}")
        assert fmt.family is FormatFamily.IMAGE
        assert fmt.extension == ext

    def test_images_always_ocr(self) -> None:
        """An image has no text layer, so the confidence gate has nothing to score."""
        assert detect_format(PNG, "scan.png").always_ocr is True

    def test_images_are_paged(self) -> None:
        assert detect_format(JPEG, "x.jpg").provenance is ProvenanceMode.PAGED

    def test_scanner_lies_about_the_extension(self) -> None:
        """Scanners emit .tif files that are really JPEGs. Trust the bytes."""
        assert detect_format(JPEG, "page001.tif").mime == "image/jpeg"


class TestOffice:
    @pytest.mark.parametrize(("prefix", "ext"), [("word", "docx"), ("xl", "xlsx"), ("ppt", "pptx")])
    def test_ooxml_by_internal_layout(self, prefix: str, ext: str) -> None:
        """All OOXML files are ZIPs; only the internal paths distinguish them."""
        fmt = detect_format(ooxml(prefix), f"report.{ext}")
        assert fmt.family is FormatFamily.OFFICE
        assert fmt.extension == ext

    def test_ooxml_ignores_a_misleading_extension(self) -> None:
        assert detect_format(ooxml("xl"), "report.docx").extension == "xlsx"

    @pytest.mark.parametrize(
        ("mimetype", "ext"),
        [
            ("application/vnd.oasis.opendocument.text", "odt"),
            ("application/vnd.oasis.opendocument.spreadsheet", "ods"),
            ("application/vnd.oasis.opendocument.presentation", "odp"),
        ],
    )
    def test_odf_by_declared_mimetype(self, mimetype: str, ext: str) -> None:
        fmt = detect_format(odf(mimetype), f"doc.{ext}")
        assert fmt.family is FormatFamily.OFFICE
        assert fmt.extension == ext

    @pytest.mark.parametrize("ext", ["doc", "xls", "ppt"])
    def test_legacy_ole2_needs_the_extension(self, ext: str) -> None:
        """One container signature covers .doc/.xls/.ppt; nothing else separates them."""
        assert detect_format(OLE2, f"old.{ext}").extension == ext

    def test_legacy_ole2_without_a_usable_extension_is_rejected(self) -> None:
        with pytest.raises(UnsupportedFormatError, match="legacy Microsoft"):
            detect_format(OLE2, "mystery.bin")

    def test_rtf(self) -> None:
        assert detect_format(RTF, "letter.rtf").family is FormatFamily.OFFICE

    def test_office_needs_conversion(self) -> None:
        assert detect_format(ooxml("word"), "a.docx").needs_conversion is True

    def test_plain_zip_is_rejected(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("notes.txt", "hello")
        with pytest.raises(UnsupportedFormatError, match="ZIP archive"):
            detect_format(buf.getvalue(), "bundle.zip")


class TestWeb:
    @pytest.mark.parametrize(
        "body",
        [
            b"<!DOCTYPE html><html><body>hi</body></html>",
            b"<html lang='en'><head></head></html>",
            b"\n\n  <!doctype HTML>\n<html>",
        ],
    )
    def test_detects_html(self, body: bytes) -> None:
        fmt = detect_format(body, "page.html")
        assert fmt.family is FormatFamily.WEB
        assert fmt.needs_conversion is True
        assert fmt.provenance is ProvenanceMode.PAGED


class TestFlow:
    @pytest.mark.parametrize(
        ("ext", "mime"),
        [
            ("md", "text/markdown"),
            ("txt", "text/plain"),
            ("csv", "text/csv"),
            ("json", "application/json"),
        ],
    )
    def test_text_types_resolve_by_extension(self, ext: str, mime: str) -> None:
        fmt = detect_format(b"alpha,beta\n1,2\n", f"data.{ext}")
        assert fmt.family is FormatFamily.FLOW
        assert fmt.mime == mime

    def test_flow_uses_character_provenance(self) -> None:
        """No pages means no bbox to highlight; offsets are the honest answer."""
        fmt = detect_format(b"# Heading\n\nSome prose.\n", "notes.md")
        assert fmt.provenance is ProvenanceMode.FLOW
        assert fmt.needs_conversion is False

    def test_unknown_extension_falls_back_to_plain_text(self) -> None:
        assert detect_format(b"just some words here", "README").mime == "text/plain"

    def test_utf8_text_is_accepted(self) -> None:
        assert detect_format("naïve café — ünicode".encode(), "a.txt").family is FormatFamily.FLOW


class TestRejection:
    def test_empty_file(self) -> None:
        with pytest.raises(UnsupportedFormatError, match="empty file"):
            detect_format(b"", "empty.pdf")

    def test_binary_garbage(self) -> None:
        with pytest.raises(UnsupportedFormatError, match="unrecognised content"):
            detect_format(b"\x00\x01\x02\x03\xff\xfe" * 40, "mystery.dat")

    def test_null_bytes_disqualify_text(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            detect_format(b"looks like text\x00but is not", "a.txt")
