"""Reproducible source-group splits and shared logo preprocessing.

The manifest records known source families; legacy provenance remains unknown.
Validation measures this small collection, not independent real-world accuracy.
"""
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def preprocess_image(image):
    """Composite transparency with contrast, then letterbox without distortion."""
    rgba = ImageOps.exif_transpose(image).convert('RGBA')
    pixels = np.asarray(rgba)
    visible = pixels[..., 3] > 0
    light = visible.any() and np.mean(pixels[..., :3][visible]) > 200
    background = (40, 40, 40) if light and np.any(pixels[..., 3] < 255) else (255, 255, 255)
    canvas = Image.new('RGBA', rgba.size, background + (255,))
    canvas.alpha_composite(rgba)
    image = ImageOps.contain(canvas.convert('RGB'), (256, 256), Image.Resampling.LANCZOS)
    output = Image.new('RGB', (256, 256), background)
    output.paste(image, ((256-image.width)//2, (256-image.height)//2))
    return output


def prepare_dataset(root, seed=42):
    root = Path(root)
    with (root/'university_info.csv').open(encoding='utf-8-sig', newline='') as stream:
        names = [row['name'] for row in csv.DictReader(stream)]
    manifest = root/'dataset_manifest.json'
    if not manifest.exists():
        raise ValueError('Missing dataset_manifest.json; source grouping must be reviewed first')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('seed', seed) != seed:
        raise ValueError('Requested seed differs from the persisted manifest split')
    entries = data['images']
    parent = list(range(len(entries)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a,b):
        parent[find(b)] = find(a)
    groups, hashes = {}, {}
    paths = []
    for i, entry in enumerate(entries):
        path = (root/entry['file']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Manifest path escapes dataset root')
        if entry['university'] not in names:
            raise ValueError('Unknown university in manifest')
        if entry['split'] not in {'train','validation'}:
            raise ValueError('Invalid manifest split')
        if entry.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Manifest digest mismatch: {path.name}')
        paths.append(path)
        group = (entry['university'], entry['source_group'])
        if group in groups: union(i,groups[group])
        groups[group] = i
        with Image.open(path) as image:
            rgba = image.convert('RGBA')
            fingerprint = (rgba.size, hashlib.sha256(rgba.tobytes()).digest())
        if fingerprint in hashes:
            previous = hashes[fingerprint]
            if entries[previous]['university'] != entry['university']:
                raise ValueError('Identical image assigned conflicting university labels')
            union(i,previous)
        hashes[fingerprint] = i
    # A duplicate crossing a persisted boundary pulls its whole family into validation.
    validation_groups = {find(i) for i,e in enumerate(entries) if e['split']=='validation'}
    train, validation = [], []
    for i,e in enumerate(entries):
        target = validation if find(i) in validation_groups else train
        target.append((paths[i], names.index(e['university'])))
    if {y for _,y in train} != set(range(len(names))):
        raise ValueError('Each university needs a training source group; duplicate-held-out families cannot also train')
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation, dict(enumerate(names))


def load_images(rows, augment=False, seed=42):
    """Return normalized arrays; augmentation must only be enabled for training."""
    rng = random.Random(seed)
    images, labels = [], []
    for path, label in rows:
        with Image.open(path) as source:
            image = preprocess_image(source)
        if augment:
            image = image.rotate(rng.uniform(-8,8), Image.Resampling.BICUBIC, fillcolor=image.getpixel((0,0)))
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(.8,1.2))
            if rng.random() < .3: image = ImageOps.grayscale(image).convert('RGB')
            if rng.random() < .4: image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.2,.8)))
        images.append(np.asarray(image, dtype=np.float32)/255.)
        labels.append(label)
    return (np.stack(images) if images else np.empty((0,256,256,3),dtype=np.float32), np.asarray(labels,dtype=np.int64))
