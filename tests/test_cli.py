import subprocess
import sys
from pathlib import Path
import unittest


class PredictionCliTests(unittest.TestCase):
    def test_help_does_not_load_a_model(self):
        root = Path(__file__).resolve().parents[1]
        try:
            result = subprocess.run(
                [sys.executable, str(root / 'test.py'), '--help'],
                cwd=root, capture_output=True, text=True, timeout=8,
                encoding='utf-8', errors='replace',
            )
        except subprocess.TimeoutExpired:
            self.fail('Help must return without importing TensorFlow or opening a model/plot')
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        self.assertIn('--image', result.stdout)


if __name__ == '__main__':
    unittest.main()
