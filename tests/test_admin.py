from streamlit.testing.v1 import AppTest


def gate_app(password='test-admin-password-123'):
    return AppTest.from_string(f'''import streamlit as st
from unittest.mock import patch
import admin
with patch.object(admin, 'get_review_password', return_value=st.session_state.get('test_configured_password', {password!r})):
    if admin.require_review_admin():
        st.button('approval action')
''')


def test_review_is_closed_without_configuration():
    ui = gate_app(None).run()
    assert not ui.exception
    assert not ui.button
    assert ui.info


def test_wrong_password_cannot_approve_correct_password_unlocks_and_logout_locks():
    ui = gate_app().run()
    assert not any(b.label == 'approval action' for b in ui.button)
    ui.text_input[0].input('wrong')
    next(b for b in ui.button if b.label == 'Unlock').click().run()
    assert ui.error
    assert not any(b.label == 'approval action' for b in ui.button)
    ui.text_input[0].input('test-admin-password-123')
    next(b for b in ui.button if b.label == 'Unlock').click().run()
    assert not ui.exception
    assert any(b.label == 'approval action' for b in ui.button)
    assert not ui.session_state['review_password_input']
    next(b for b in ui.button if b.label == 'Lock').click().run()
    assert not any(b.label == 'approval action' for b in ui.button)


def test_password_rotation_invalidates_session():
    ui = gate_app().run()
    ui.text_input[0].input('test-admin-password-123')
    next(b for b in ui.button if b.label == 'Unlock').click().run()
    assert ui.session_state['review_admin_session']
    ui.session_state['test_configured_password'] = 'rotated-admin-password-456'
    ui.run()
    assert not any(b.label == 'approval action' for b in ui.button)


def test_main_does_not_read_pending_or_expose_export_when_locked():
    ui = AppTest.from_string('''import app
from unittest.mock import patch
with patch.object(app, 'require_review_admin', return_value=False), patch.object(app.FeedbackStore, 'list_pending', side_effect=AssertionError('private list')):
    app.main()
''').run()
    assert not ui.exception
    assert not any(b.label in {'อนุมัติ', 'ไม่อนุมัติ', 'ส่งออกข้อมูลที่อนุมัติแล้ว'} for b in ui.button)
