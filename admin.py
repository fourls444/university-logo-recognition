"""Server-side password gate for review actions; submissions remain public."""
import hashlib
import hmac
import os

import streamlit as st


def get_review_password():
    password = os.environ.get('REVIEW_ADMIN_PASSWORD')
    if password is None:
        try:
            password = st.secrets.get('review_admin_password')
        except (FileNotFoundError, KeyError):
            return None
    return password if isinstance(password, str) and len(password) >= 16 and not password.startswith('REPLACE-') else None


def _fingerprint(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def _unlock(password):
    submitted = st.session_state.get('review_password_input', '')
    valid = hmac.compare_digest(_fingerprint(submitted), _fingerprint(password))
    st.session_state['review_admin_session'] = _fingerprint(password) if valid else None
    st.session_state['review_login_error'] = not valid
    st.session_state['review_password_input'] = ''


def _lock():
    st.session_state.pop('review_admin_session', None)
    st.session_state.pop('review_login_error', None)


def require_review_admin():
    password = get_review_password()
    if not password:
        st.info('Review ยังไม่เปิดใช้งาน • ตั้งรหัสแอดมินก่อน')
        return False
    authenticated = hmac.compare_digest(
        str(st.session_state.get('review_admin_session', '')), _fingerprint(password))
    if authenticated:
        st.button('Lock', on_click=_lock, key='review_lock')
        return True
    with st.container(key='review_access'):
        with st.form('review_login', clear_on_submit=True):
            st.markdown('<div class="access-icon" aria-hidden="true"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5" y="10" width="14" height="11" rx="3"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg></div>', unsafe_allow_html=True)
            st.subheader('Admin access')
            st.caption('เข้าสู่ระบบเพื่อตรวจทานคำตอบ')
            st.text_input('Password', type='password', placeholder='ใส่รหัสแอดมิน', key='review_password_input')
            st.form_submit_button('Unlock', type='primary', width='stretch', on_click=_unlock, args=(password,))
            if st.session_state.get('review_login_error'):
                st.error('รหัสผ่านไม่ถูกต้อง ลองอีกครั้ง')
    return False
