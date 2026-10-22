"""Presentation helpers; display changes never modify prediction inputs."""
import base64
import io
from html import escape
from urllib.parse import urlparse
import streamlit as st


def preview_image(image):
    preview = image.copy()
    preview.thumbnail((480, 320))
    return preview


def image_preview_html(image, height=320, label='Logo image preview', width=None):
    """Contain a display copy in a stable frame, including transparent logos."""
    preview = preview_image(image)
    output = io.BytesIO()
    preview.save(output, format='PNG')
    encoded = base64.b64encode(output.getvalue()).decode('ascii')
    frame_width = f'{int(width)}px' if width else '100%'
    return (f'<div class="image-preview-stage" style="width:{frame_width};max-width:100%;height:{int(height)}px;'
            'display:flex;align-items:center;justify-content:center;overflow:hidden;box-sizing:border-box;'
            'border:1px solid #303a49;border-radius:12px;'
            'background-color:#171d27;background-image:conic-gradient(#202936 25%,transparent 0 50%,#202936 0 75%,transparent 0);background-size:20px 20px;">'
            f'<img alt="{escape(label, quote=True)}" src="data:image/png;base64,{encoded}" '
            'style="display:block;width:100%;height:100%;object-fit:contain;padding:12px;box-sizing:border-box;" /></div>')


def render_image_preview(image, height=320, caption=None, width=None):
    st.markdown(image_preview_html(image, height, width=width), unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def results_html(candidates):
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        name = escape(candidate.get('name_th') or candidate['name'])
        url = candidate.get('url', '')
        link = (f'<a class="university-link" href="{escape(url, quote=True)}" '
                f'target="_blank" rel="noopener noreferrer" aria-label="เว็บไซต์ {name} (เปิดแท็บใหม่)">Website ↗</a>'
                if urlparse(url).scheme in {'https', 'http'} else '')
        rows.append(f'<li class="rank-row"><span class="rank-number">{rank:02d}</span>'
                    f'<div class="rank-name"><strong>{name}</strong>{link}</div>'
                    f'<span class="rank-score">{candidate["score"]:.1%}</span></li>')
    return '<ol class="rank-list" aria-label="ตัวเลือกมหาวิทยาลัยเรียงตามคะแนน">'+''.join(rows)+'</ol>'


def empty_state(title, description):
    st.markdown(f'''<div class="empty-state">
<svg aria-hidden="true" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></svg>
<strong>{escape(title)}</strong><p>{escape(description)}</p></div>''', unsafe_allow_html=True)


def apply_style():
    st.markdown('''<style>
    .stMainBlockContainer { max-width:1120px; padding-top:2.25rem; padding-bottom:3rem; }
    h1 { font-size:clamp(1.65rem,3vw,2.1rem)!important; font-weight:650!important; letter-spacing:-.035em; line-height:1.3!important; padding-bottom:.35rem!important; }
    h2 { font-size:1.4rem!important; } h3 { font-size:1.12rem!important; line-height:1.55!important; }
    [data-testid="stCaptionContainer"] { font-size:.88rem; line-height:1.65; }
    [data-testid="stImage"] { margin-inline:auto; }
    [data-testid="stImage"] img { object-fit:contain; }
    .stTabs [role="tablist"] { gap:1.4rem; margin-top:.5rem; margin-bottom:.7rem; }
    .stTabs [role="tab"] { min-height:44px; font-size:.94rem; }
    [data-testid="stMetricValue"] { font-size:1.8rem; }
    [data-testid="stMetricLabel"] { font-size:.85rem; }
    .st-key-review_access { max-width:420px; margin:2rem auto 3rem; }
    .st-key-review_access [data-testid="stForm"] { padding:28px; border:1px solid #303744; border-radius:16px; background:#12161e; }
    .st-key-review_access h3 { font-size:1.35rem!important; padding-bottom:0!important; }
    .access-icon { width:46px; height:46px; display:grid; place-items:center; color:#a8c5fa; background:#1d2b43; border-radius:14px; margin-bottom:4px; }
    .st-key-review_access button { min-height:44px; }
    .st-key-review_access [data-testid="stTextInput"] input { min-height:46px; }
    .st-key-upload_panel { padding-top:16px; }
    [role="dialog"]:has(.crop-dialog-marker) { width:min(760px,calc(100vw - 32px))!important; max-width:760px!important; }
    [data-rac]:has(> [role="dialog"] .crop-dialog-marker) { width:min(760px,calc(100vw - 32px))!important; max-width:760px!important; }
    [role="dialog"]:has(.crop-dialog-marker) [data-testid="stElementContainer"]:has(.crop-dialog-marker) { display:none; }
    [role="dialog"]:has(.crop-dialog-marker) button { min-height:44px; }
    [role="dialog"]:has(.crop-dialog-marker) button[aria-label="Close"] { width:44px; height:44px; right:10px; top:10px; }
    .st-key-logo_upload [data-testid="stFileUploaderDropzone"] { min-height:154px; padding:24px 20px; gap:16px; flex-direction:column; justify-content:center; align-items:center; text-align:center; border:1px dashed #485264; border-radius:14px; background:#151922; transition:background .16s ease,border-color .16s ease; }
    .st-key-logo_upload [data-testid="stFileUploaderDropzone"]:hover { border-color:#7caeff; background:#1b2331; }
    .st-key-logo_upload [data-testid="stFileUploaderDropzone"]:focus-within { border-color:#93bfff; outline:2px solid #93bfff; outline-offset:3px; }
    .st-key-logo_upload [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] { min-height:44px; padding-inline:24px; border-color:#526279; background:#25344b; color:#f0f4fc; white-space:nowrap; }
    .st-key-logo_upload [data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChips"]) { min-height:82px; padding:12px 14px; flex-direction:row; align-items:center; text-align:left; }
    .st-key-logo_upload [data-testid="stFileChips"] { min-width:0; width:100%; }
    .st-key-logo_upload [data-testid="stFileChip"] { min-width:0; background:#151922; border:1px solid #303a49; border-radius:10px; padding:6px 8px; }
    .st-key-logo_upload button[data-testid="stBaseButton-minimal"], .st-key-logo_upload button[data-testid="stBaseButton-borderlessIcon"] { width:44px; min-width:44px; height:44px; min-height:44px; padding:0; border-radius:9px; color:#b9c4d5; }
    .st-key-logo_upload button[data-testid="stBaseButton-minimal"] { background:transparent; border:0; }
    .st-key-logo_upload button[data-testid="stBaseButton-minimal"]:hover { background:#42252d; color:#ffb3be; }
    .st-key-logo_upload button[data-testid="stBaseButton-borderlessIcon"] { background:#1e2531; border:1px solid #354154; }
    .st-key-logo_upload [data-testid="stFileUploaderDropzoneInstructions"] { gap:8px; }
    .st-key-logo_upload [data-testid="stFileUploaderDropzoneInstructions"] small { color:#b4c2d9; line-height:1.6; }
    .st-key-logo_upload [data-testid="stFileUploaderDropzone"] svg { color:#8cb8ff; }
    .st-key-search_action button { min-height:46px; }
    .empty-state { min-height:200px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; gap:12px; padding:24px 18px; color:var(--text-color); }
    .empty-state svg { opacity:.45; } .empty-state strong { font-size:1rem; font-weight:550; }
    .empty-state p { opacity:.65; font-size:.9rem; line-height:1.7; max-width:340px; margin:0; }
    .rank-list { list-style:none!important; padding:0!important; margin:0!important; }
    .rank-row { display:grid; grid-template-columns:30px minmax(0,1fr) auto; align-items:center; gap:12px; padding:14px 0; margin:0!important; border-bottom:1px solid color-mix(in srgb,var(--text-color) 12%,transparent); }
    .rank-row:first-child { padding-top:12px; } .rank-row:last-child { border-bottom:0; }
    .rank-number { font-size:.78rem; opacity:.5; font-variant-numeric:tabular-nums; }
    .rank-name { min-width:0; } .rank-name strong { display:block; font-size:.95rem; font-weight:500; line-height:1.55; overflow-wrap:anywhere; }
    .rank-row:first-child .rank-name strong { font-weight:650; font-size:1.02rem; }
    .university-link { display:inline-block; font-size:.78rem; line-height:1.8; text-decoration:none!important; opacity:.8; }
    .university-link:hover { text-decoration:underline!important; opacity:1; }
    .rank-score { font-size:1rem; font-weight:550; font-variant-numeric:tabular-nums; white-space:nowrap; }
    .rank-row:first-child .rank-score { font-size:1.25rem; }
    @media(max-width:640px) {
      .st-key-review_access { margin:1rem auto; }
      .st-key-review_access [data-testid="stForm"] { padding:22px; }
      .st-key-logo_upload [data-testid="stFileUploaderDropzone"] { flex-wrap:wrap; min-height:154px; padding:20px 16px; gap:12px; }
      .stMainBlockContainer { padding:1.2rem 1rem 2rem; }
      .stTabs [role="tablist"] { gap:.9rem; }
      .stTabs [role="tab"] { font-size:.82rem; }
      .rank-row { gap:8px; grid-template-columns:22px minmax(0,1fr) auto; }
      .rank-name strong { font-size:.9rem; }
    }
    </style>''', unsafe_allow_html=True)
