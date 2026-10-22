import io
from pathlib import Path
import unittest
from PIL import Image
from streamlit.testing.v1 import AppTest


class ImageSearchTests(unittest.TestCase):
    def test_default_uses_full_image_without_crop_controls(self):
        source = '''import streamlit as st
from PIL import Image
from app import select_search_image
image = Image.new('RGB', (80, 40), 'red')
result = select_search_image(image, 'sample')
st.session_state['result_size'] = result.size
'''
        ui = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state['result_size'], (80, 40))
        self.assertFalse(ui.checkbox)
        self.assertTrue(any(button.label == 'Crop image' for button in ui.button))
        self.assertFalse(ui.slider)
        self.assertTrue(any('ภาพเต็ม' in item.value for item in ui.caption))

    def test_display_crop_maps_back_to_original_and_clamps_bounds(self):
        from app import crop_from_display_box
        image = Image.new('RGB', (800, 400), 'white')
        image.paste('red', (200, 100, 600, 300))
        result = crop_from_display_box(image, {'left': 50, 'top': 25, 'width': 100, 'height': 50}, (200, 100))
        self.assertEqual(result.size, (400, 200))
        self.assertEqual(result.getextrema(), ((255, 255), (0, 0), (0, 0)))
        result = crop_from_display_box(image, {'left': -20, 'top': 0, 'width': 240, 'height': 100}, (200, 100))
        self.assertEqual(result.size, image.size)
        with self.assertRaises(ValueError):
            crop_from_display_box(image, {'left': 0, 'top': 0, 'width': 0, 'height': 10}, (200, 100))

    def test_main_uploader_accepts_only_images(self):
        root = Path(__file__).resolve().parents[1]
        source = f'''import app
from pathlib import Path
from unittest.mock import patch
import streamlit as st
def uploader(label, **kwargs):
    st.session_state['accepted_types'] = kwargs['type']
    return None
with patch.object(app.st, 'file_uploader', uploader):
    app.main()
'''
        ui = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(ui.exception)
        accepted = ui.session_state['accepted_types']
        self.assertIn('png', accepted)
        self.assertNotIn('pdf', accepted)
        self.assertNotIn('docx', accepted)


if __name__ == '__main__':
    unittest.main()
