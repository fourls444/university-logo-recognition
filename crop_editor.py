"""Direct crop editor: a fixed stage and original-pixel coordinates."""
import base64
import io
from pathlib import Path
import streamlit as st

_HTML = '''<div class="editor">
<div class="stage"><svg aria-label="Crop selection. Drag handles to resize or center to move." role="img"><image/><path class="shade" fill-rule="evenodd"/><rect class="selection" data-handle="move"/></svg></div>
<div class="toolbar"><button type="button" class="reset">Full image</button><output aria-live="polite"></output></div>
<details><summary>Exact crop (pixels)</summary><div class="coordinates">
<label>Left<input data-field="left" type="number" min="0" step="1"></label><label>Top<input data-field="top" type="number" min="0" step="1"></label><label>Width<input data-field="width" type="number" min="1" step="1"></label><label>Height<input data-field="height" type="number" min="1" step="1"></label>
</div></details></div>'''
_CSS = '''.editor{font:14px system-ui;color:#e5ebf5}.stage{height:320px;width:100%;box-sizing:border-box;padding:18px;border:1px solid #3c485b;border-radius:12px;background-color:#171d27;background-image:conic-gradient(#202936 25%,transparent 0 50%,#202936 0 75%,transparent 0);background-size:20px 20px}svg{display:block;width:100%;height:100%;overflow:visible;touch-action:none;user-select:none}.shade{fill:#080d19;fill-opacity:.65;pointer-events:none}.selection{fill:transparent;stroke:#8dbaff;stroke-width:2;vector-effect:non-scaling-stroke;cursor:move}.handle{cursor:var(--cursor);outline:none}.handle:focus .dot{stroke:#f7d47b;stroke-width:3}.dot{fill:#e7efff;stroke:#2869d0;stroke-width:2;vector-effect:non-scaling-stroke;pointer-events:none}.hit{fill:transparent}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:10px 0}button{background:#25344b;color:#edf3fd;border:1px solid #526279;border-radius:8px;padding:8px 14px;font:inherit;min-height:40px;cursor:pointer}button:hover{background:#344765}button:focus-visible,summary:focus-visible,input:focus-visible{outline:2px solid #9bc2ff;outline-offset:2px}output{font-variant-numeric:tabular-nums;color:#b9c7da;font-size:13px}summary{cursor:pointer;padding:6px 0;color:#b9c7da}.coordinates{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:10px}label{font-size:12px;color:#b9c7da}input{width:100%;box-sizing:border-box;min-height:38px;background:#151c28;color:#e5ebf5;border:1px solid #526279;border-radius:6px;padding:6px;font:inherit;margin-top:4px}@media(max-width:450px){.coordinates{grid-template-columns:repeat(2,minmax(0,1fr))}.stage{height:320px}}'''
_component = st.components.v2.component('direct_crop_editor', html=_HTML, css=_CSS,
                                       js=Path(__file__).with_suffix('.js').read_text(encoding='utf-8'))


def crop_editor(image, initial_box, key):
    # Only the display copy is bounded; coordinates always address original pixels.
    preview = image.copy()
    preview.thumbnail((1600, 1600))
    encoded = io.BytesIO()
    preview.save(encoded, format='PNG')
    saved = st.session_state.get(key)
    draft = saved.get('box', initial_box) if saved else initial_box
    result = _component(data={'src': 'data:image/png;base64,' + base64.b64encode(encoded.getvalue()).decode('ascii'),
                              'width': image.width, 'height': image.height, 'box': draft},
                        default={'box': initial_box}, on_box_change=lambda: None,
                        key=key, height='content', width='stretch')
    return result.box or initial_box
