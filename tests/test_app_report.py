import json
import tempfile
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class AppReportTests(unittest.TestCase):
    def test_report_reads_selected_metadata_without_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'university_info.csv').write_text('name,name_th,url\nTest,ทดสอบ,\n', encoding='utf-8')
            version = root / 'artifacts' / 'versions' / 'test'
            version.mkdir(parents=True)
            (root / 'artifacts' / 'current.json').write_text(json.dumps({'version':'test'}))
            (version / 'metadata.json').write_text(json.dumps({
                'validation_accuracy': .75, 'train_source_count':12,
                'train_array_count':48, 'validation_count':4,
                'class_metrics': {'Test':{'validation_sources':4,'recall':.75}},
                'history':{'accuracy':[.5,.9], 'val_accuracy':[.5,.75]},
                'uncertainty':{'calibrated':False}}))
            source = f'''import app
from pathlib import Path
from unittest.mock import patch
with patch.object(app, 'get_recognizer', side_effect=AssertionError('should not load model')):
    app.render_training_report(Path({str(root)!r}))
'''
            app = AppTest.from_string(source).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn('75.0%', [metric.value for metric in app.metric])
            self.assertTrue(app.dataframe)
            self.assertTrue(app.get('vega_lite_chart'))

    def test_report_rejects_parent_directory_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = root / 'artifacts'
            artifacts.mkdir()
            (artifacts / 'current.json').write_text('{"version":".."}')
            source = f"from app import render_training_report\nrender_training_report({str(root)!r})"
            app = AppTest.from_string(source).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.info)
            self.assertFalse(app.metric)

    def test_correction_requires_explicit_choice(self):
        source = '''import streamlit as st
from app import render_correction_controls
choice, submitted = render_correction_controls({'Test':'ทดสอบ'})
st.session_state['choice'] = choice
'''
        app = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIsNone(app.selectbox[0].value)
        self.assertTrue(app.button[0].disabled)
        app.selectbox[0].select('__unknown__').run(timeout=30)
        self.assertFalse(app.button[0].disabled)
        self.assertEqual(app.session_state['choice'], '__unknown__')


class CorrectionIdentityTests(unittest.TestCase):
    def test_saved_correction_cannot_be_submitted_twice(self):
        source = """import streamlit as st
from app import render_correction_controls
st.session_state['saved_feedback_receipt'] = ('result-a', 'Test')
render_correction_controls({'Test':'ทดสอบ'}, 'result-a')
"""
        app = AppTest.from_string(source).run(timeout=30)
        app.selectbox[0].select('Test').run(timeout=30)
        self.assertTrue(app.button[0].disabled)
        app.selectbox[0].select('__unknown__').run(timeout=30)
        self.assertFalse(app.button[0].disabled)

    def test_new_result_requires_new_explicit_correction(self):
        source = """import streamlit as st
from app import render_correction_controls
identity = st.session_state.get('test_identity', 'first')
render_correction_controls({'Test':'ทดสอบ'}, identity)
"""
        app = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(app.exception)
        app.selectbox[0].select('__unknown__').run(timeout=30)
        self.assertFalse(app.button[0].disabled)
        app.session_state['test_identity'] = 'second'
        app.run(timeout=30)
        self.assertIsNone(app.selectbox[0].value)
        self.assertTrue(app.button[0].disabled)
