import json
from PIL import Image
import pytest


def test_persistence_review_and_export_boundary(tmp_path):
    from feedback import FeedbackStore
    store = FeedbackStore(tmp_path)
    key = store.add(Image.new('RGBA',(10,20),(255,0,0,0)), {'candidates':[]}, 'Bangkok University')
    assert len(FeedbackStore(tmp_path).list_pending()) == 1
    manifest = json.loads(store.export_approved().read_text())
    assert manifest['rows'] == []
    store.review(key, True)
    manifest = json.loads(store.export_approved().read_text())
    assert len(manifest['rows']) == 1
    row = manifest['rows'][0]
    assert row['correct_name'] == 'Bangkok University'
    assert row['source_group'].startswith('feedback:')
    assert Image.open(tmp_path / row['image_path']).mode == 'RGB'
    assert store.list_pending() == []
    store.review(key, False)
    assert json.loads(store.export_approved().read_text())['rows'] == []


def test_invalid_labels_and_paths_are_rejected(tmp_path):
    from feedback import FeedbackStore
    store = FeedbackStore(tmp_path)
    with pytest.raises(ValueError):
        store.add(Image.new('RGB',(2,2)), {}, '../../bad')
    with pytest.raises(TypeError):
        store.add('/etc/passwd', {}, '__unknown__')
    key = store.add(Image.new('RGB',(2,2)), {}, '__unknown__')
    store.review(key, True)
    assert len(json.loads(store.export_approved().read_text())['rows']) == 1
    with pytest.raises(KeyError):
        store.review('missing', True)

def test_feedback_training_excludes_pending_unknown_and_validation_matches(tmp_path):
    from feedback import FeedbackStore
    from train import reviewed_feedback_rows
    store = FeedbackStore(tmp_path)
    held = tmp_path/'held.png'
    Image.new('RGB',(40,40),'red').save(held)
    match = store.add(Image.new('RGB',(40,40),'red'),{},'Bangkok University')
    store.review(match,True)
    store.add(Image.new('RGB',(40,40),'blue'),{},'Bangkok University')
    unknown = store.add(Image.new('RGB',(40,40),'green'),{},'__unknown__')
    store.review(unknown,True)
    accepted = store.add(Image.new('RGB',(40,40),'black'),{},'Bangkok University')
    store.review(accepted,True)
    rows,skipped = reviewed_feedback_rows(tmp_path,[],[(held,0)],{0:'Bangkok University'})
    assert len(rows) == 1
    assert rows[0][1] == 0
    assert skipped == 2


def test_feedback_conflicting_approved_labels_excluded(tmp_path):
    from feedback import FeedbackStore
    from train import reviewed_feedback_rows
    store = FeedbackStore(tmp_path)
    for label in ['Bangkok University','Burapha University']:
        key = store.add(Image.new('RGB',(20,20),'purple'),{},label)
        store.review(key,True)
    rows,skipped = reviewed_feedback_rows(tmp_path,[],[],{0:'Bangkok University',1:'Burapha University'})
    assert rows == []
    assert skipped == 2

def test_feedback_releases_database_file_after_each_operation(tmp_path):
    import os
    from feedback import FeedbackStore
    store = FeedbackStore(tmp_path)
    key = store.add(Image.new('RGB',(3,3)),{},'__unknown__')
    store.list_pending()
    store.review(key, True)
    store.export_approved()
    # Windows refuses removal of a database file while a SQLite handle is open.
    store.db.unlink()
    assert not store.db.exists()
