import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from dataset import prepare_dataset, load_images

class DatasetTests(unittest.TestCase):
    def test_real_manifest_coverage(self):
        root = Path(__file__).resolve().parents[1]
        train, validation, names = prepare_dataset(root)
        self.assertEqual(len(names), 30)
        self.assertEqual({y for _, y in train}, set(range(30)))
        self.assertTrue({y for _, y in validation}.issubset(set(range(30))))
        self.assertFalse({p.parent.name for p, _ in train if p.parent.parent.name == 'logos'} & {p.parent.name for p, _ in validation if p.parent.parent.name == 'logos'})
        self.assertGreater(len(train) + len(validation), 120)
        self.assertEqual((train, validation, names), prepare_dataset(root))

    def test_source_groups_and_duplicates_never_cross_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'university_info.csv').write_text('name\nExample\n')
            folder = root / 'Logo' / 'Example'
            folder.mkdir(parents=True)
            entries = []
            for i, color in enumerate(['red', 'green', 'blue', 'red']):
                p = folder / f'{i}.png'
                Image.new('RGB', (20, 10), color).save(p)
                entries.append({'file': p.relative_to(root).as_posix(), 'university':'Example', 'source_group': 'related' if i < 2 else f'g{i}', 'split': 'validation' if i < 2 else 'train'})
            (root / 'dataset_manifest.json').write_text(json.dumps({'seed':42, 'images':entries}))
            train, valid, _ = prepare_dataset(root)
            # Duplicate red bridges g3 with related; blue is the sole train group.
            self.assertEqual({p.name for p, _ in train}, {'2.png'})
            self.assertEqual({p.name for p, _ in valid}, {'0.png','1.png','3.png'})

    def test_manifest_digest_rejects_changed_source(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'university_info.csv').write_text('name\nExample\n')
            p=root/'image.png'; Image.new('RGB',(10,10),'red').save(p)
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'dataset_manifest.json').write_text(json.dumps({'seed':42,'images':[{'file':'image.png','university':'Example','source_group':'one','split':'train','sha256':digest}]}))
            Image.new('RGB',(10,10),'blue').save(p)
            with self.assertRaisesRegex(ValueError,'digest'):
                prepare_dataset(root)

    def test_preprocessing_preserves_aspect_and_white_alpha_logo(self):
        from dataset import preprocess_image
        image = Image.new('RGBA', (100,50), (255,255,255,0))
        for x in range(20,80):
            for y in range(10,40): image.putpixel((x,y),(255,255,255,255))
        result = np.array(preprocess_image(image))
        self.assertEqual(result.shape, (256,256,3))
        self.assertGreater(result[128,128,0], 240)
        self.assertLess(result[0,0,0], 100)
        self.assertLess(result[70,128,0], 100)

    def test_augmentation_is_opt_in_reproducible_and_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'image.png'
            image = Image.new('RGB',(80,40),'white')
            for x in range(20,60):
                for y in range(10,30): image.putpixel((x,y),(200,0,0))
            image.save(p)
            before = p.read_bytes()
            plain, labels = load_images([(p,3)])
            aug, _ = load_images([(p,3)], augment=True, seed=42)
            self.assertEqual(plain.dtype,np.float32)
            self.assertEqual(plain.shape,(1,256,256,3))
            self.assertEqual(labels.tolist(),[3])
            self.assertFalse(np.array_equal(plain,aug))
            np.testing.assert_array_equal(aug,load_images([(p,3)],augment=True,seed=42)[0])
            np.testing.assert_array_equal(plain,load_images([(p,3)])[0])
            self.assertEqual(before,p.read_bytes())
