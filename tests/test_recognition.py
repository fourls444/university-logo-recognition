import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pytest


def test_missing_artifacts_never_load_legacy(tmp_path):
    from recognition import LogoRecognizer
    (tmp_path / 'model.h5').write_bytes(b'legacy')
    result = LogoRecognizer(tmp_path).predict(Image.new('RGB', (20, 20)))
    assert result['uncertain'] and not result['candidates']
    assert 'artifact' in result['reason'].lower()


def test_real_model_ranking_and_checksum(tmp_path):
    import tensorflow as tf
    from train import publish_artifacts
    from recognition import LogoRecognizer
    model = tf.keras.Sequential([tf.keras.layers.Input((256,256,3)), tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='zeros', bias_initializer=tf.keras.initializers.Constant([2.,0.]))])
    publish_artifacts(tmp_path, model, {0:'Bangkok University',1:'Burapha University'}, {'status':'trained','seed':42})
    result = LogoRecognizer(tmp_path).predict(Image.new('RGB', (25, 30)), top_k=2)
    assert [c['name'] for c in result['candidates']] == ['Bangkok University','Burapha University']
    assert 0.85 < result['candidates'][0]['score'] < .9
    assert not result['uncertain']
    (tmp_path/'artifacts'/'versions'/json.loads((tmp_path/'artifacts'/'current.json').read_text())['version']/'labels.json').write_text('{"0":"Wrong"}')
    assert LogoRecognizer(tmp_path).predict(Image.new('RGB',(2,2)))['uncertain']


def test_close_scores_are_uncertain(tmp_path):
    import tensorflow as tf
    from train import publish_artifacts
    from recognition import LogoRecognizer
    model = tf.keras.Sequential([tf.keras.layers.Input((256,256,3)),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(2,activation='softmax',kernel_initializer='zeros',bias_initializer='zeros')])
    publish_artifacts(tmp_path, model, {0:'Bangkok University',1:'Burapha University'}, {'status':'trained'})
    assert LogoRecognizer(tmp_path).predict(Image.new('RGB',(5,5)))['uncertain']


def test_transfer_model_is_frozen_and_preserves_normalized_input_contract():
    import tensorflow as tf
    from train import build_model
    model = build_model(3, weights=None)
    extractor = model.get_layer('feature_extractor')
    assert not extractor.trainable
    assert {id(weight) for weight in model.trainable_weights} == {id(weight) for weight in model.get_layer('classifier').trainable_weights}
    assert model.count_params() < 3000000
    scaling = extractor.get_layer('mobilenet_scaling')
    np.testing.assert_allclose(scaling(tf.constant([0.,.5,1.])).numpy(), [-1.,0.,1.])
    scores = model(np.zeros((1,256,256,3),dtype=np.float32),training=False).numpy()
    assert scores.shape == (1,3)
    np.testing.assert_allclose(scores.sum(axis=1),[1.],atol=1e-6)

def test_head_training_stops_and_restores_best_validation_weights():
    import tensorflow as tf
    from train import train_head
    head = tf.keras.Sequential([tf.keras.layers.Input((2,)),tf.keras.layers.Dense(2,activation='softmax',kernel_initializer='zeros',bias_initializer='zeros')])
    features = np.ones((8,2),dtype=np.float32)
    history = train_head(head,features,np.zeros(8,dtype=np.int64),features,np.ones(8,dtype=np.int64),epochs=30,verbose=0)
    assert 1 < len(history.history['loss']) < 30
    restored = float(head.evaluate(features,np.ones(8,dtype=np.int64),verbose=0)[0])
    assert restored < history.history['val_loss'][-1]
    assert abs(restored-min(history.history['val_loss'])) < 1e-5

