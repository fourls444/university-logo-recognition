"""Reproduce the reviewed logo crops from the downloaded official KU manual."""
import hashlib
import json
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PDF = HERE/'sources/documents/kasetsart-university-identity-manual.pdf'
URL = 'https://www.ku.ac.th/web-upload/m_document/KSU_Manual.pdf'
# Coordinates refer to PDF pages rendered at scale 1.5; no automatic detection.
CROPS = [
    (12, 'seal-thai-color', (176, 290, 486, 607)),
    (12, 'seal-english-color', (755, 290, 1065, 607)),
    (22, 'horizontal-primary', (64, 237, 305, 327)),
    (22, 'horizontal-black', (340, 237, 599, 327)),
    (22, 'horizontal-reversed', (620, 224, 879, 345)),
    (22, 'horizontal-green-background', (897, 224, 1157, 345)),
    (22, 'horizontal-brown-background', (897, 345, 1157, 465)),
    (22, 'horizontal-gray-background', (897, 465, 1157, 585)),
    (22, 'horizontal-yellow-background', (897, 585, 1157, 712)),
]

if __name__ == '__main__':
    output = HERE/'derived/kasetsart-university'
    output.mkdir(parents=True, exist_ok=True)
    (HERE/'manifests').mkdir(exist_ok=True)
    (HERE/'previews').mkdir(exist_ok=True)
    rows = []
    pdf = pdfium.PdfDocument(PDF)
    canvas = Image.new('RGB', (900, 3*220), '#e5e7eb')
    draw = ImageDraw.Draw(canvas)
    for index, (page_number, variant, box) in enumerate(CROPS):
        page = pdf[page_number-1]
        bitmap = page.render(scale=1.5)
        try:
            cropped = bitmap.to_pil().convert('RGB').crop(box)
        finally:
            bitmap.close()
            page.close()
        target = output/f'kasetsart-university-logo-{variant}-01.png'
        cropped.save(target)
        rows.append({'file':target.relative_to(ROOT).as_posix(),
            'university':'Kasetsart University', 'source_page':'https://ku.ac.th/th/kulogo?category=64',
            'document_url':URL, 'parent_file':PDF.relative_to(ROOT).as_posix(),
            'parent_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
            'pdf_page':page_number, 'render_scale':1.5, 'crop_xyxy':list(box),
            'source_group':'Kasetsart University official branding (keep related variants in one split)',
            'split':'validation',
            'split_reason':'Same official branding family already held out by active manifest; never import these as training.',
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'status':'visually_reviewed_candidate_not_in_active_manifest',
            'rights':'Official university artwork. Download access does not establish redistribution permission.'})
        cropped.thumbnail((280, 180))
        x, y = index%3*300, index//3*220
        canvas.paste(cropped, (x+(280-cropped.width)//2, y+(180-cropped.height)//2))
        draw.text((x+5,y+185), variant, fill='#111827')
    pdf.close()
    canvas.save(HERE/'previews/kasetsart-university-document-crops.jpg', quality=92)
    record = {'schema_version':1, 'document_url':URL,
        'source_document_bytes':PDF.stat().st_size, 'source_document_pages':192, 'images':rows,
        'failed_sources':[{'url':'https://mahidol.ac.th/wp-content/uploads/document/logo_mahidol/MU-Visual-Identity-Guideline01.pdf', 'status':403, 'action':'Stopped; no bypass attempted'}]}
    (HERE/'manifests/sources.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Extracted {len(rows)} official document logo candidates; held-out family remains isolated')
