import unittest
from streamlit.testing.v1 import AppTest

SOURCE = '''import streamlit as st
from PIL import Image
import crop_editor
from app import select_search_image
image = Image.new('RGB', (1280, 840), 'white')
image.paste('red', (320, 210, 960, 630))
def cropper(image, **kwargs):
    st.session_state['canvas_size'] = image.size
    st.session_state['default_box'] = kwargs['initial_box']
    return {'left': 320, 'top': 210, 'width': 640, 'height': 420}
crop_editor.crop_editor = cropper
result = select_search_image(image, st.session_state.get('source', 'first'))
st.session_state['result_size'] = result.size
st.session_state['result_pixel'] = result.getpixel((0, 0))
'''


def click(ui, label):
    matches = [button for button in ui.button if button.label == label]
    assert matches, f'Missing action: {label}'
    matches[0].click().run()
    if ui.exception:
        raise AssertionError(ui.exception[0].message)
    return ui


class CropDialogTests(unittest.TestCase):
    def setUp(self):
        # The SVG component is a browser boundary unavailable to AppTest. Keep app and
        # dialog execution real, and replace only its coordinate response.
        import crop_editor
        self.original_cropper = crop_editor.crop_editor

    def tearDown(self):
        import crop_editor
        crop_editor.crop_editor = self.original_cropper

    def test_cancel_preserves_full_source_and_apply_maps_original_pixels(self):
        ui = AppTest.from_string(SOURCE).run(timeout=30)
        click(ui, 'Crop image')
        self.assertEqual(ui.session_state['canvas_size'], (1280, 840))
        self.assertEqual(ui.session_state['default_box'], {'left': 0, 'top': 0, 'width': 1280, 'height': 840})
        self.assertEqual(ui.session_state['result_size'], (1280, 840))
        click(ui, 'Cancel')
        self.assertEqual(ui.session_state['result_size'], (1280, 840))
        click(ui, 'Crop image')
        click(ui, 'Apply crop')
        self.assertEqual(ui.session_state['result_size'], (640, 420))
        self.assertEqual(ui.session_state['result_pixel'], (255, 0, 0))
        click(ui, 'Crop image')
        click(ui, 'Cancel')
        self.assertEqual(ui.session_state['result_size'], (640, 420))
        click(ui, 'Use full image')
        self.assertEqual(ui.session_state['result_size'], (1280, 840))

    def test_new_upload_has_independent_crop(self):
        ui = AppTest.from_string(SOURCE).run(timeout=30)
        click(ui, 'Crop image')
        click(ui, 'Apply crop')
        ui.session_state['source'] = 'second'
        ui.run()
        self.assertEqual(ui.session_state['result_size'], (1280, 840))
        self.assertEqual(ui.session_state['result_pixel'], (255, 255, 255))



class CropCoordinateTests(unittest.TestCase):
    def test_phone_crop_still_maps_to_source_pixels(self):
        from PIL import Image
        from app import crop_from_display_box
        image = Image.new('RGB', (1280, 840), 'white')
        image.paste('red', (320, 210, 960, 630))
        result = crop_from_display_box(image, {'left': 80, 'top': 52.5, 'width': 160, 'height': 105}, (320, 210))
        self.assertEqual(result.size, (640, 420))
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))


class CropDraftPersistenceTests(unittest.TestCase):
    def test_rerender_restores_draft_instead_of_original_rectangle(self):
        source = '''import streamlit as st
from PIL import Image
from types import SimpleNamespace
from unittest.mock import patch
import crop_editor
st.session_state['draft_probe'] = {'box': {'left': 25, 'top': 50, 'width': 100, 'height': 200}}
def mount(**kwargs):
    return SimpleNamespace(box=kwargs['data']['box'])
with patch.object(crop_editor, '_component', mount):
    box = crop_editor.crop_editor(Image.new('RGBA', (200, 400)), {'left': 0, 'top': 0, 'width': 200, 'height': 400}, 'draft_probe')
st.session_state['restored_box'] = box
'''
        ui = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state['restored_box'], {'left': 25, 'top': 50, 'width': 100, 'height': 200})
