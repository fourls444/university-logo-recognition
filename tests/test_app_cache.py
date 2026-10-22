import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from app import image_digest, artifact_fingerprint


class AppCacheTests(unittest.TestCase):
    def test_palette_and_transparency_change_digest(self):
        first = Image.new('P', (2, 2), 0)
        first.putpalette([255, 0, 0] + [0, 0, 0] * 255)
        second = first.copy()
        second.putpalette([0, 0, 255] + [0, 0, 0] * 255)
        self.assertNotEqual(image_digest(first), image_digest(second))
        second = first.copy()
        second.info['transparency'] = 0
        self.assertNotEqual(image_digest(first), image_digest(second))

    def test_artifact_changes_invalidate_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = root / 'artifacts'
            artifacts.mkdir()
            initial = artifact_fingerprint(root)
            (artifacts / 'model.keras').write_bytes(b'model one')
            first = artifact_fingerprint(root)
            self.assertNotEqual(initial, first)
            (artifacts / 'model.keras').write_bytes(b'model two')
            self.assertNotEqual(first, artifact_fingerprint(root))
            version = artifacts / 'versions' / 'v1'
            version.mkdir(parents=True)
            (version / 'model.keras').write_bytes(b'versioned model')
            (artifacts / 'current.json').write_text(json.dumps({'version': 'v1'}))
            first = artifact_fingerprint(root)
            (version / 'labels.json').write_text('{"0":"new label"}')
            self.assertNotEqual(first, artifact_fingerprint(root))
