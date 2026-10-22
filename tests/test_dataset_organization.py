"""Prove path migration preserves source bytes, labels and persisted splits."""
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_organization_preserves_original_manifest_and_images():
    provenance = ROOT / 'data/provenance'
    mapping = json.loads((provenance / 'dataset-path-mapping.json').read_text(encoding='utf-8'))
    snapshot = (provenance / 'dataset-manifest-before-organization.json').read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == mapping['snapshot_sha256']
    old = json.loads(snapshot)
    current = json.loads((ROOT / 'dataset_manifest.json').read_text(encoding='utf-8'))
    moved = {item['old']: item for item in mapping['moves']}
    current_rows = {row['file']: row for row in current['images']}
    for row in old['images']:
        expected = dict(row, file=moved.get(row['file'], {}).get('new', row['file']))
        assert current_rows[expected['file']] == expected
    assert len(current_rows) == len(old['images']) == mapping['baseline']['manifest_images']
    assert sum(item['kind'] == 'legacy' for item in moved.values()) == 120
    for item in moved.values():
        path = (ROOT / item['new']).resolve()
        assert path.is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        assert not (ROOT / item['old']).exists()


def test_example_names_have_exact_pixel_label_evidence():
    mapping = json.loads((ROOT / 'data/provenance/dataset-path-mapping.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'dataset_manifest.json').read_text(encoding='utf-8'))
    def pixels(path):
        with Image.open(path) as image:
            image = image.convert('RGBA')
            return image.size, hashlib.sha256(image.tobytes()).digest()
    known = {}
    for row in manifest['images']:
        known.setdefault(pixels(ROOT / row['file']), set()).add(row['university'])
    examples = [item for item in mapping['moves'] if item['kind'] == 'example']
    assert len(examples) == 28
    for item in examples:
        assert known[pixels(ROOT / item['new'])] == {item['university']}


def test_candidate_paths_are_grouped_and_training_utility_is_importable():
    assert not (ROOT/'dataset_candidates').exists()
    assert (ROOT/'data/candidates').is_dir()
    mapping = json.loads((ROOT/'data/provenance/dataset-path-mapping.json').read_text(encoding='utf-8'))
    candidates = [row for row in mapping['moves'] if row['kind']=='candidate']
    assert candidates
    assert all(row['new'].startswith('data/candidates/') for row in candidates)
    assert all('logo-candidate-' in Path(row['new']).name for row in candidates if row.get('renamed_university_asset'))
    from scripts.augment_dataset import build_candidates
    assert callable(build_candidates)
