import io
import zipfile
import unittest
from PIL import Image
import documents as d


def png():
    out = io.BytesIO()
    Image.new('RGBA', (20, 10), (0, 80, 0, 128)).save(out, 'PNG')
    return out.getvalue()


def docx(entries):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', '<Types/>')
        z.writestr('word/document.xml', '<document/>')
        for name, content in entries:
            z.writestr(name, content)
    return out.getvalue()


class DocumentTests(unittest.TestCase):
    def test_reject_empty_and_oversize(self):
        for data in [b'', b'x' * (d.MAX_UPLOAD_BYTES + 1)]:
            with self.assertRaises(ValueError): d.load_image(data)

    def test_image_decode_and_malformed(self):
        self.assertEqual(d.load_image(png()).size, (20, 10))
        with self.assertRaises(ValueError): d.load_image(b'bad image')

    def test_docx_images_without_word(self):
        images = d.extract_docx_images(docx([('word/media/image1.png', png())]))
        self.assertEqual(images[0][0], 'image1.png')
        self.assertEqual(images[0][1].size, (20, 10))

    def test_docx_malformed_and_bomb(self):
        for data in [b'bad zip', docx([('word/media/bomb.png', b'0' * 2_000_000)])]:
            with self.assertRaises(ValueError): d.extract_docx_images(data)

    def test_docx_aggregate_limit(self):
        from unittest.mock import patch
        with patch.object(d, 'MAX_DOCX_EXPANDED_BYTES', 100):
            with self.assertRaises(ValueError):
                d.extract_docx_images(docx([('word/media/image.png', png())]))

    def test_docx_decoded_pixel_budget(self):
        from unittest.mock import patch
        payload = docx([('word/media/a.png', png()), ('word/media/b.png', png())])
        with patch.object(d, 'MAX_DOCX_DECODED_PIXELS', 300, create=True):
            with self.assertRaises(ValueError):
                d.extract_docx_images(payload)

    def test_pdf_page_and_bounds(self):
        out = io.BytesIO()
        Image.new('RGB', (100, 80), 'white').save(out, 'PDF')
        data = out.getvalue()
        self.assertEqual(d.pdf_page_count(data), 1)
        self.assertEqual(d.render_pdf_page(data, 0).size, (200, 160))
        with self.assertRaises(ValueError): d.render_pdf_page(data, 1)
        with self.assertRaises(ValueError): d.pdf_page_count(b'bad pdf')
        from unittest.mock import patch
        with patch.object(d, 'MAX_PDF_PAGES', 0):
            with self.assertRaises(ValueError): d.pdf_page_count(data)

if __name__ == '__main__': unittest.main()
