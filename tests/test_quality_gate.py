from PIL import Image


def test_weak_validation_keeps_even_high_scores_uncertain(tmp_path):
    import tensorflow as tf
    from train import publish_artifacts
    from recognition import LogoRecognizer
    model = tf.keras.Sequential([
        tf.keras.layers.Input((256, 256, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='zeros',
                              bias_initializer=tf.keras.initializers.Constant([8., 0.])),
    ])
    publish_artifacts(tmp_path, model, {0: 'Bangkok University', 1: 'Burapha University'},
                      {'status': 'trained', 'validation_accuracy': 0.24})
    result = LogoRecognizer(tmp_path).predict(Image.new('RGB', (40, 40), 'white'))
    assert result['candidates'][0]['score'] > 0.99
    assert result['uncertain'], 'High softmax score must not hide observed weak model validation'
