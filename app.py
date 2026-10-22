"""Local Thai university logo recognition interface."""
import csv
import hashlib
import json
import math
from pathlib import Path
import streamlit as st
from PIL import Image
from documents import load_image
from recognition import LogoRecognizer
from feedback import FeedbackStore
from admin import require_review_admin
from ui import apply_style, empty_state, render_image_preview, results_html

ROOT = Path(__file__).resolve().parent


def image_digest(image):
    canonical = image.convert('RGBA')
    return hashlib.sha256(b'RGBA' + str(canonical.size).encode() + canonical.tobytes()).hexdigest()


def artifact_fingerprint(root):
    artifacts = Path(root) / 'artifacts'
    digest = hashlib.sha256()
    pointer = artifacts / 'current.json'
    selected = artifacts
    if pointer.exists():
        content = pointer.read_bytes()
        digest.update(content)
        try:
            version = json.loads(content)['version']
            if isinstance(version, str) and Path(version).name == version:
                selected = artifacts / 'versions' / version
        except (ValueError, KeyError, TypeError):
            pass  # The recognizer reports invalid artifact metadata to the user.
    for name in ('metadata.json', 'labels.json', 'model.keras'):
        path = selected / name
        digest.update(name.encode())
        if path.is_file():
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
        else:
            digest.update(b'missing')
    return digest.hexdigest()


@st.cache_resource(show_spinner=False, max_entries=2)
def get_recognizer(fingerprint):
    return LogoRecognizer(ROOT)


class ModelUnavailableError(RuntimeError):
    """Keep model setup guidance separate from an inference error."""


def predict_with_loading(image, model_identity):
    """Reserve the results area through model startup and prediction."""
    with st.skeleton(height=350):
        recognizer = get_recognizer(model_identity)
        if recognizer.error:
            raise ModelUnavailableError(str(recognizer.error))
        return recognizer.predict(image, top_k=5)


def render_training_report(root):
    st.subheader('Training results')
    try:
        artifacts = Path(root) / 'artifacts'
        selected = artifacts
        pointer = artifacts / 'current.json'
        if pointer.exists():
            if pointer.stat().st_size > 4096:
                raise ValueError('ข้อมูลรุ่นโมเดลใหญ่เกินกำหนด')
            version = json.loads(pointer.read_text(encoding='utf-8'))['version']
            if not isinstance(version, str) or not version or Path(version).name != version or version in {'.', '..'}:
                raise ValueError('ข้อมูลรุ่นโมเดลไม่ถูกต้อง')
            selected = artifacts / 'versions' / version
            if not selected.resolve().is_relative_to(artifacts.resolve()):
                raise ValueError('ตำแหน่งรุ่นโมเดลไม่ถูกต้อง')
        metadata_path = selected / 'metadata.json'
        if metadata_path.stat().st_size > 1024 * 1024:
            raise ValueError('รายงานมีขนาดใหญ่เกินกำหนด')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        accuracy = float(metadata['validation_accuracy'])
        if not 0 <= accuracy <= 1:
            raise ValueError('ค่าความแม่นยำไม่ถูกต้อง')
        counts = [int(metadata[key]) for key in ('train_source_count', 'train_array_count', 'validation_count')]
        if any(count < 0 for count in counts):
            raise ValueError('จำนวนภาพไม่ถูกต้อง')
        classes = metadata['class_metrics']
        covered = sum(int(value.get('validation_sources', 0)) > 0 for value in classes.values())
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        st.info('ยังไม่มีรายงานการฝึกที่อ่านได้ กรุณาฝึกโมเดลด้วย python train.py --epochs 10')
        return
    cols = st.columns(4)
    cols[0].metric('ทายอันดับแรกถูก', f'{accuracy:.1%}', border=True)
    cols[1].metric('ภาพต้นทาง', counts[0], border=True)
    cols[2].metric('ตัวอย่างที่ใช้ฝึก', counts[1], border=True)
    cols[3].metric('ภาพตรวจผล', counts[2], border=True)
    st.caption(f'ผลจากชุดตรวจสอบ • ครอบคลุม {covered}/{len(classes)} มหาวิทยาลัย')
    history = metadata.get('history', {})
    if isinstance(history, dict):
        training = history.get('accuracy', [])
        validation = history.get('val_accuracy', [])
        if isinstance(training, list) and isinstance(validation, list) and training and len(training) == len(validation):
            st.caption('Accuracy by epoch')
            st.line_chart({'ชุดฝึก': training, 'ชุดตรวจสอบ': validation}, x_label='รอบการฝึก (เริ่มจาก 0)', y_label='ความแม่นยำ', height=260)
    rows = [{'มหาวิทยาลัย': name, 'ภาพตรวจสอบ': values.get('validation_sources', 0), 'สัดส่วนที่ทายถูกในมหาวิทยาลัยนี้': values.get('recall')} for name, values in classes.items()]
    with st.expander('ดูผลแยกตามมหาวิทยาลัย'):
        st.dataframe(rows, hide_index=True, width='stretch', height=300)


def render_correction_controls(universities, result_identity=None):
    options = list(universities) + ['__unknown__']
    control_identity = hashlib.sha256(repr(result_identity).encode()).hexdigest()
    correction = st.selectbox('คำตอบที่ถูกต้อง', options, index=None, key=f'correction_{control_identity}', placeholder='เลือกมหาวิทยาลัย หรือไม่รู้จัก', format_func=lambda name: 'ไม่รู้จัก / ไม่ใช่โลโก้ในรายการ' if name == '__unknown__' else universities[name])
    st.caption('บันทึกภาพและคำตอบไว้ในเครื่อง')
    already_saved = st.session_state.get('saved_feedback_receipt') == (result_identity, correction)
    submitted = st.button('บันทึกคำตอบเพื่อรอตรวจทาน', disabled=correction is None or already_saved, key=f'save_correction_{control_identity}')
    return correction, submitted


def crop_from_display_box(image, box, display_size):
    """Map the draggable preview rectangle back to full-resolution source pixels."""
    values = [float(box[key]) for key in ('left', 'top', 'width', 'height')]
    if not all(math.isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
        raise ValueError('กรอบต้องมีความกว้างและความสูงมากกว่าศูนย์')
    sx, sy = image.width / display_size[0], image.height / display_size[1]
    left, top, width, height = values
    coords = (max(0, min(image.width, round(left * sx))),
              max(0, min(image.height, round(top * sy))),
              max(0, min(image.width, round((left + width) * sx))),
              max(0, min(image.height, round((top + height) * sy))))
    if coords[0] >= coords[2] or coords[1] >= coords[3]:
        raise ValueError('เลือกกรอบที่อยู่ภายในภาพและมีพื้นที่มากกว่าศูนย์')
    return image.crop(coords)


def dismiss_crop_dialog():
    st.session_state.pop('crop_dialog_open', None)


@st.dialog('Crop image', width='large', on_dismiss=dismiss_crop_dialog)
def crop_image_dialog(image, identity, revision):
    from crop_editor import crop_editor

    st.markdown('<span class="crop-dialog-marker" aria-hidden="true"></span>', unsafe_allow_html=True)
    committed = st.session_state.get(f'crop_selection_{identity}')
    if committed:
        previous_box, previous_size = committed
        sx, sy = image.width / previous_size[0], image.height / previous_size[1]
        initial_box = {key: previous_box[key] * (sx if key in ('left', 'width') else sy)
                       for key in ('left', 'top', 'width', 'height')}
    else:
        initial_box = {'left': 0, 'top': 0, 'width': image.width, 'height': image.height}
    st.caption('ลากจุดหรือขอบเพื่อปรับกรอบ • ลากตรงกลางเพื่อย้าย')
    box = crop_editor(image, initial_box=initial_box, key=f'crop_box_{identity}_{revision}')
    try:
        crop_from_display_box(image, box, image.size)
        valid = True
    except (ValueError, TypeError, KeyError) as exc:
        st.warning(f'กรอบภาพยังไม่พร้อม: {exc}')
        valid = False
    apply, cancel = st.columns(2)
    if apply.button('Apply crop', type='primary', disabled=not valid, width='stretch'):
        st.session_state[f'crop_selection_{identity}'] = (dict(box), image.size)
        dismiss_crop_dialog()
        st.rerun()
    if cancel.button('Cancel', width='stretch'):
        dismiss_crop_dialog()
        st.rerun()


def select_search_image(image, identity):
    committed = st.session_state.get(f'crop_selection_{identity}')
    selected = crop_from_display_box(image, *committed) if committed else image
    render_image_preview(selected, caption='ภาพที่จะใช้ค้นหา' if committed else None)
    crop_action, reset_action = st.columns(2) if committed else (st.container(), None)
    if crop_action.button('Crop image', key=f'open_crop_{identity}', icon=':material/crop:', width='stretch'):
        revision_key = f'crop_revision_{identity}'
        revision = st.session_state.get(revision_key, 0) + 1
        st.session_state[revision_key] = revision
        st.session_state['crop_dialog_open'] = (identity, revision)
    opened = st.session_state.get('crop_dialog_open')
    if opened and opened[0] == identity:
        crop_image_dialog(image, identity, opened[1])
    elif opened:
        dismiss_crop_dialog()
    if committed and reset_action.button('Use full image', key=f'reset_crop_{identity}', width='stretch'):
        del st.session_state[f'crop_selection_{identity}']
        st.rerun()
    st.caption('ใช้ภาพเต็มได้ • หากพื้นหลังเยอะ แนะนำ Crop')
    return selected


def main():
    st.set_page_config(page_title='University Logo Recognition', layout='wide')
    apply_style()
    st.title('University Logo Recognition')
    st.caption('ค้นหาจากโลโก้ภาพสีหรือขาวดำ')
    recognize_tab, review_tab, report_tab = st.tabs(['Search', 'Review', 'Training'])
    with (ROOT / 'university_info.csv').open(encoding='utf-8-sig', newline='') as handle:
        universities = {row['name']: row['name_th'] for row in csv.DictReader(handle) if row.get('name')}
    store = FeedbackStore(ROOT)
    with report_tab:
        render_training_report(ROOT)
    with review_tab:
        if require_review_admin():
            st.subheader('ตรวจทานคำตอบ')
            receipt = st.session_state.pop('review_action_receipt', None)
            if receipt:
                st.toast(receipt)
            pending = store.list_pending()
            st.caption(f'รอตรวจ {len(pending)} รายการ')
            if not pending:
                empty_state('ยังไม่มีรายการ', 'คำตอบที่บันทึกจะปรากฏที่นี่')
            for number, item in enumerate(pending, 1):
                label = universities.get(item['correct_name'], 'ไม่ใช่มหาวิทยาลัยในรายการ')
                with st.expander(f'{number}. {label}'):
                    thumb, details = st.columns([1, 3], gap='medium')
                    with thumb:
                        with Image.open(ROOT / item['image_path']) as review_image:
                            render_image_preview(review_image, height=120, width=120)
                    with details:
                        st.write('คำตอบที่เสนอ:', label)
                        previous = item['prediction'].get('candidates', [])
                        if previous:
                            st.caption('เดิมระบบเสนอ: ' + (previous[0].get('name_th') or previous[0]['name']))
                        approve, reject = st.columns(2)
                        if approve.button('อนุมัติ', key=f"approve_{item['id']}", type='primary', width='stretch'):
                            store.review(item['id'], True)
                            st.session_state['review_action_receipt'] = 'อนุมัติคำตอบแล้ว'
                            st.rerun()
                        if reject.button('ไม่อนุมัติ', key=f"reject_{item['id']}", width='stretch'):
                            store.review(item['id'], False)
                            st.session_state['review_action_receipt'] = 'ไม่อนุมัติคำตอบแล้ว'
                            st.rerun()
            st.divider()
            st.caption('ส่งออกคำตอบที่อนุมัติแล้วสำหรับฝึกครั้งถัดไป')
            if st.button('ส่งออกข้อมูลที่อนุมัติแล้ว'):
                export = store.export_approved()
                st.success('เตรียมไฟล์ส่งออกแล้ว')
                if Path(export).is_file():
                    st.download_button('ดาวน์โหลดไฟล์ส่งออก', Path(export).read_bytes(), file_name=Path(export).name)
    with recognize_tab:
        left, right = st.columns([1, 1.1], gap='large')
        crop = None
        requested = False
        with left:
            with st.container(border=True, key='upload_panel'):
                st.subheader('Logo image')
                uploaded = st.file_uploader('เลือกรูปโลโก้', key='logo_upload', type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'], max_upload_size=20,
                                            label_visibility='collapsed', help='PNG, JPG, WEBP, BMP หรือ TIFF ขนาดไม่เกิน 20 MB')
                if uploaded is not None:
                    data = uploaded.getvalue()
                    digest = hashlib.sha256(data).hexdigest()[:16]
                    try:
                        image = load_image(data)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        crop = select_search_image(image, digest)
                requested = st.button('ค้นหาโลโก้', type='primary', width='stretch', disabled=crop is None, key='search_action')
        with right:
            with st.container(border=True, key='results_panel'):
                st.subheader('Results')
                if crop is None:
                    empty_state('ยังไม่มีผลค้นหา', 'เลือกรูปภาพ แล้วกดค้นหาโลโก้')
                    return
                model_identity = artifact_fingerprint(ROOT)
                result_identity = (image_digest(crop), model_identity)
                if requested:
                    try:
                        prediction = predict_with_loading(crop, model_identity)
                        st.session_state['logo_result'] = (result_identity, prediction)
                    except ModelUnavailableError as exc:
                        st.warning('ยังเปิดโมเดลไม่ได้ กรุณาตรวจสอบไฟล์หรือฝึกโมเดลก่อน')
                        st.code('python train.py --epochs 10')
                        with st.expander('รายละเอียดปัญหา'):
                            st.caption(str(exc))
                        return
                    except (ValueError, RuntimeError) as exc:
                        st.error(f'ประมวลผลไม่ได้: {exc}')
                result = st.session_state.get('logo_result')
                if not result or result[0] != result_identity:
                    empty_state('พร้อมค้นหา', 'กดค้นหาโลโก้เพื่อดู Top 5')
                    return
                prediction = result[1]
                if prediction['uncertain']:
                    st.warning('ยังไม่มั่นใจ โปรดตรวจสอบคำตอบ')
                st.caption('Model score • ไม่ใช่ % ความถูกต้อง')
                st.markdown(results_html(prediction['candidates']), unsafe_allow_html=True)
                with st.expander('เกี่ยวกับคะแนน'):
                    st.write('คะแนนใช้จัดอันดับคำตอบ ไม่ใช่เปอร์เซ็นต์ความเหมือนของภาพหรือการรับประกันความถูกต้อง')
                    if prediction.get('reason'):
                        st.caption(prediction['reason'])
            with st.expander('แก้ไขคำตอบ'):
                correction, submitted = render_correction_controls(universities, result_identity)
                if submitted and correction is not None:
                    store.add(crop, prediction, correction)
                    st.session_state['saved_feedback_receipt'] = (result_identity, correction)
                    st.rerun()
                if st.session_state.get('saved_feedback_receipt') == (result_identity, correction):
                    st.success('บันทึกแล้ว • รอผู้ดูแลตรวจทาน')


if __name__ == '__main__':
    main()
