from PIL import Image


def test_preview_is_bounded_without_mutating_model_input():
    import ui
    assert hasattr(ui, 'preview_image'), 'Need a bounded display preview independent of prediction input'
    original = Image.new('RGBA', (1200, 2400), (255, 0, 0, 100))
    preview = ui.preview_image(original)
    assert preview.width <= 480 and preview.height <= 320
    assert original.size == (1200, 2400)
    assert original.mode == 'RGBA'
    assert preview.width / preview.height == .5


def test_ranked_results_escape_names_and_reject_unsafe_links():
    import ui
    assert hasattr(ui, 'results_html'), 'Need compact ranked results with safe presentation of labels'
    html = ui.results_html([
        {'name': 'A', 'name_th': '<script>bad</script>', 'score': .81, 'url': 'javascript:alert(1)'},
        {'name': 'B', 'name_th': 'มหาวิทยาลัย บี', 'score': .12, 'url': 'https://example.edu/?a=1&b=2'},
    ])
    assert '<script>' not in html and 'javascript:' not in html
    assert '&lt;script&gt;' in html
    assert html.index('81.0%') < html.index('12.0%')
    assert 'https://example.edu/?a=1&amp;b=2' in html
    assert 'มหาวิทยาลัย บี' in html
    assert 'aria-label="เว็บไซต์ มหาวิทยาลัย บี (เปิดแท็บใหม่)"' in html


def test_preview_stage_has_constant_dimensions_for_all_aspect_ratios():
    from html.parser import HTMLParser
    import ui
    assert hasattr(ui, 'image_preview_html'), 'Need fixed preview stage renderer'
    class Images(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == 'img':
                self.image = dict(attrs)
            if tag == 'div':
                self.stage = dict(attrs)
    for size in [(2400, 200), (200, 2400), (640, 640)]:
        original = Image.new('RGBA', size, (20, 30, 40, 0))
        parser = Images()
        parser.feed(ui.image_preview_html(original))
        assert 'height:320px' in parser.stage['style']
        assert 'width:100%' in parser.stage['style']
        assert 'object-fit:contain' in parser.image['style']
        assert original.size == size
