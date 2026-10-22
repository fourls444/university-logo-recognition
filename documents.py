"""Bounded local decoding for image, PDF and DOCX uploads."""
import io
import math
import warnings
import zipfile
from pathlib import PurePosixPath
from PIL import Image, ImageOps

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_PDF_PAGES = 100
MAX_PDF_EDGE = 2400
MAX_DOCX_EXPANDED_BYTES = 50 * 1024 * 1024
MAX_DOCX_ENTRIES = 1000
MAX_DOCX_IMAGES = 100
MAX_DOCX_DECODED_PIXELS = 20_000_000


def _validate(data):
    if not isinstance(data, bytes) or not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('ไฟล์ว่างหรือใหญ่เกิน 20 MB')


def load_image(data):
    _validate(data)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError('ภาพมีขนาดใหญ่เกิน 20 ล้านพิกเซล')
                image.load()
                return ImageOps.exif_transpose(image).copy()
    except Exception as exc:
        raise ValueError('อ่านภาพไม่ได้ หรือภาพมีขนาดใหญ่เกินกำหนด') from exc


def _open_pdf(data):
    _validate(data)
    if not data.lstrip().startswith(b'%PDF-'):
        raise ValueError('ไฟล์ PDF ไม่ถูกต้อง')
    import pypdfium2 as pdfium
    try:
        pdf = pdfium.PdfDocument(data)
        if not 0 < len(pdf) <= MAX_PDF_PAGES:
            pdf.close()
            raise ValueError('PDF ต้องมี 1–100 หน้า')
        return pdf
    except Exception as exc:
        raise ValueError('อ่าน PDF ไม่ได้ หรือจำนวนหน้าเกินกำหนด') from exc


def pdf_page_count(data):
    pdf = _open_pdf(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_pdf_page(data, page_index=0):
    pdf = _open_pdf(data)
    try:
        if not isinstance(page_index, int) or not 0 <= page_index < len(pdf):
            raise ValueError('หมายเลขหน้าไม่ถูกต้อง')
        page = pdf[page_index]
        try:
            width, height = page.get_size()
            if not all(math.isfinite(v) and v > 0 for v in (width, height)):
                raise ValueError('ขนาดหน้า PDF ไม่ถูกต้อง')
            scale = min(2.0, MAX_PDF_EDGE / max(width, height))
            bitmap = page.render(scale=scale)
            try:
                return bitmap.to_pil().copy()
            finally:
                bitmap.close()
        finally:
            page.close()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('แสดงหน้า PDF ไม่ได้') from exc
    finally:
        pdf.close()


def extract_docx_images(data):
    _validate(data)
    images = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if '[Content_Types].xml' not in names or 'word/document.xml' not in names:
                raise ValueError('ไม่ใช่ไฟล์ DOCX')
            if len(entries) > MAX_DOCX_ENTRIES or sum(e.file_size for e in entries) > MAX_DOCX_EXPANDED_BYTES:
                raise ValueError('ข้อมูลภายใน DOCX ใหญ่เกินกำหนด')
            if any(e.flag_bits & 1 or e.file_size / max(e.compress_size, 1) > 200 for e in entries):
                raise ValueError('DOCX เข้ารหัสหรือมีอัตราบีบอัดสูงเกินกำหนด')
            media = [e for e in entries if e.filename.startswith('word/media/') and PurePosixPath(e.filename).suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp'}]
            if len(media) > MAX_DOCX_IMAGES:
                raise ValueError('DOCX มีรูปภาพเกิน 100 รูป')
            decoded_pixels = 0
            for entry in media:
                payload = archive.read(entry)
                with Image.open(io.BytesIO(payload)) as header:
                    decoded_pixels += header.width * header.height
                if decoded_pixels > MAX_DOCX_DECODED_PIXELS:
                    raise ValueError('ภาพรวมใน DOCX มีขนาดเกิน 20 ล้านพิกเซล')
                images.append((PurePosixPath(entry.filename).name, load_image(payload)))
        return images
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('อ่าน DOCX ไม่ได้') from exc
