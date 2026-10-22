import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image


class DatasetAugmentationTests(unittest.TestCase):
    def test_derivatives_stay_with_training_family_and_are_reproducible(self):
        from scripts.augment_dataset import build_candidates
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = Image.new('RGB', (80, 60), 'white')
            image.paste('red', (20, 10, 60, 50))
            for name in ('train.png', 'related.png', 'heldout.png'):
                image.save(root / name)
            Image.new('RGB', (80, 60), 'blue').save(root/'heldout.png')
            entries = [
                {'file':'train.png', 'university':'Example', 'source_group':'family', 'split':'train'},
                {'file':'related.png', 'university':'Example', 'source_group':'family', 'split':'train'},
                {'file':'heldout.png', 'university':'Example', 'source_group':'heldout', 'split':'validation'},
            ]
            manifest = root / 'dataset_manifest.json'
            manifest.write_text(json.dumps({'images':entries}), encoding='utf-8')
            original = manifest.read_bytes(), (root/'train.png').read_bytes()
            first = build_candidates(root)
            self.assertEqual(len(first['images']), 6)
            self.assertTrue(all(row['source_group']=='family' and row['split']=='train' for row in first['images']))
            self.assertTrue(all(row['parent_file'] != 'heldout.png' for row in first['images']))
            for row in first['images']:
                self.assertEqual(hashlib.sha256((root/row['file']).read_bytes()).hexdigest(), row['sha256'])
            self.assertEqual(first, build_candidates(root))
            self.assertEqual(original, (manifest.read_bytes(), (root/'train.png').read_bytes()))

    def test_shared_validation_family_is_never_augmented(self):
        from scripts.augment_dataset import build_candidates
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'dataset_manifest.json').write_text(json.dumps({'images':[
                {'file':'must_not_read.png', 'university':'Example', 'source_group':'heldout', 'split':'train'},
                {'file':'also_not_read.png', 'university':'Example', 'source_group':'heldout', 'split':'validation'},
            ]}), encoding='utf-8')
            self.assertEqual(build_candidates(root)['images'], [])

    def test_duplicate_pixels_in_other_validation_family_are_excluded(self):
        from scripts.augment_dataset import build_candidates
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = Image.new('RGB', (80, 60), 'red')
            image.save(root/'train.png')
            image.save(root/'heldout.bmp')
            (root/'dataset_manifest.json').write_text(json.dumps({'images':[
                {'file':'train.png', 'university':'Example', 'source_group':'train-family', 'split':'train'},
                {'file':'heldout.bmp', 'university':'Example', 'source_group':'different-family', 'split':'validation'},
            ]}), encoding='utf-8')
            self.assertEqual(build_candidates(root)['images'], [])
