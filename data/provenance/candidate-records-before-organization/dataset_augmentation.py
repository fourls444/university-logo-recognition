"""Create isolated, reproducible logo variants from training families only.

Run: python dataset_augmentation.py
This does not change dataset_manifest.json or train/promote a model.
"""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from dataset import preprocess_image


def _variants(image):
    gray = ImageOps.grayscale(image).convert('RGB')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=35)
    compressed = Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')
    return {
        'grayscale': gray,
        'rotate_plus6': image.rotate(6, Image.Resampling.BICUBIC, fillcolor=image.getpixel((0, 0))),
        'rotate_minus6': image.rotate(-6, Image.Resampling.BICUBIC, fillcolor=image.getpixel((0, 0))),
        'dim_light': ImageEnhance.Brightness(image).enhance(.75),
        'jpeg35': compressed,
        'scan_blur': ImageEnhance.Contrast(gray).enhance(1.2).filter(ImageFilter.GaussianBlur(.5)),
    }


def build_candidates(root):
    root = Path(root).resolve()
    manifest_path = root/'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = manifest['images']
    held_out = {(row['university'], row['source_group']) for row in entries if row['split']=='validation'}
    families = {}
    destination = root/'dataset_candidates'/'augmentation_20261022'
    parents_path = destination/'manifests'/'parents.json'
    preferred = {}
    if parents_path.exists():
        preferred = {(row['university'],row['source_group']):row['parent_file']
                     for row in json.loads(parents_path.read_text(encoding='utf-8'))['families']}
        inventory = {(row['university'],row['source_group'],row['file']) for row in entries}
        if any((key[0],key[1],path) not in inventory for key,path in preferred.items()):
            raise ValueError('Preferred parent is absent from active manifest')
    def parent_priority(item):
        family = item['university'],item['source_group']
        return item['file'] != preferred.get(family), item['file']
    for row in sorted(entries, key=parent_priority):
        family = row['university'], row['source_group']
        if row['split']=='train' and family not in held_out:
            families.setdefault(family, row)
    def pixel_digest(image):
        return hashlib.sha256(preprocess_image(image).tobytes()).digest()
    held_out_pixels = set()
    if families:
        for row in entries:
            if row['split']=='validation':
                path = (root/row['file']).resolve()
                if not path.is_relative_to(root):
                    raise ValueError('Source path escapes dataset root')
                with Image.open(path) as image:
                    held_out_pixels.add(pixel_digest(image))
    destination.mkdir(parents=True, exist_ok=True)
    rows, tiles, family_numbers = [], [], {}
    for family, source in sorted(families.items()):
        path = (root/source['file']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source path escapes dataset root')
        source_bytes = path.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        if source.get('sha256') and source['sha256'] != digest:
            raise ValueError('Source digest mismatch')
        with Image.open(path) as original:
            image = preprocess_image(original)
        if pixel_digest(image) in held_out_pixels:
            continue
        slug = re.sub(r'[^a-z0-9]+', '-', family[0].lower()).strip('-')
        family_numbers[family[0]] = family_numbers.get(family[0], 0)+1
        image_directory = destination/'derived'/slug
        image_directory.mkdir(parents=True, exist_ok=True)
        for transform, variant in _variants(image).items():
            target = image_directory/f'{slug}-logo-{transform.replace("_", "-")}-{family_numbers[family[0]]:02d}.png'
            variant.save(target)
            rows.append({'file':target.relative_to(root).as_posix(), 'university':family[0],
                         'source_group':family[1], 'split':'train', 'parent_file':source['file'],
                         'parent_sha256':digest, 'transform':transform,
                         'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                         'status':'generated_candidate_not_in_active_manifest',
                         'provenance':source.get('provenance', 'Inherited from active training source')})
            if len(tiles)<42:
                tiles.append((variant.copy(), family[0], transform))
    record = {'schema_version':1, 'input_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'policy':'Training families only; one parent per family; all derivatives retain parent split/group. No automatic import.',
              'family_count':len(rows)//6, 'images':rows}
    (destination/'manifests').mkdir(exist_ok=True)
    (destination/'manifests'/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    if tiles:
        canvas = Image.new('RGB', (900, ((len(tiles)+5)//6)*190), '#e5e7eb')
        draw = ImageDraw.Draw(canvas)
        for index, (image, name, transform) in enumerate(tiles):
            image.thumbnail((140, 140))
            x, y = index%6*150, index//6*190
            canvas.paste(image, (x+(140-image.width)//2, y))
            draw.text((x+2, y+143), name[:22], fill='#111827')
            draw.text((x+2, y+159), transform, fill='#111827')
        (destination/'previews').mkdir(exist_ok=True)
        canvas.save(destination/'previews'/'preview.jpg', quality=90)
    return record


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    record = build_candidates(parser.parse_args().root)
    print(f"Generated {len(record['images'])} candidates from {record['family_count']} training families")
