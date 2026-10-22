"""Candidate data must remain in its original training source families."""
import hashlib
import json
import pytest
from PIL import Image


def test_experiment_rejects_derivative_of_validation_parent(tmp_path):
    from scripts.experiment_training import augmentation_rows
    Image.new('RGB',(9,9),'red').save(tmp_path/'held.png')
    row={'file':'derived.png','parent_file':'held.png','parent_sha256':hashlib.sha256((tmp_path/'held.png').read_bytes()).hexdigest(),'university':'a','source_group':'held','split':'train'}
    manifest=tmp_path/'aug.json'; manifest.write_text(json.dumps({'images':[row]}))
    with pytest.raises(ValueError,match='training family'):
        augmentation_rows(tmp_path,manifest,[{'file':'held.png','university':'a','source_group':'held','split':'validation'}],{0:'a'})


def test_experiment_rejects_changed_augmentation_file(tmp_path):
    from scripts.experiment_training import augmentation_rows
    Image.new('RGB',(9,9),'red').save(tmp_path/'parent.png')
    Image.new('RGB',(9,9),'blue').save(tmp_path/'derived.png')
    row={'file':'derived.png','sha256':'0'*64,'parent_file':'parent.png','parent_sha256':hashlib.sha256((tmp_path/'parent.png').read_bytes()).hexdigest(),'university':'a','source_group':'family','split':'train'}
    manifest=tmp_path/'aug.json'; manifest.write_text(json.dumps({'images':[row]}))
    with pytest.raises(ValueError,match='digest'):
        augmentation_rows(tmp_path,manifest,[{'file':'parent.png','university':'a','source_group':'family','split':'train'}],{0:'a'})

def test_fine_tuning_opens_last_block_but_keeps_batchnorm_inference():
    import tensorflow as tf
    from train import build_model
    from scripts.experiment_training import configure_fine_tuning
    model=build_model(3,weights=None)
    configure_fine_tuning(model)
    base=model.get_layer('feature_extractor').layers[-1]
    assert base.get_layer('block_16_expand').trainable
    assert not base.get_layer('block_15_expand').trainable
    assert all(not layer.trainable for layer in base.layers if isinstance(layer,tf.keras.layers.BatchNormalization))
    assert model.get_layer('classifier').trainable
