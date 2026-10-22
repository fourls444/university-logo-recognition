"""Search loading must cover cold model startup and inference together."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

import app


@pytest.mark.parametrize('failure', [None, 'load', 'predict', 'unavailable'])
def test_search_loading_lifecycle_and_prediction_identity(failure):
    active = False
    exited = False
    image = Image.new('RGB', (3, 3))
    prediction = {'uncertain': False, 'candidates': []}

    @contextmanager
    def skeleton(**kwargs):
        nonlocal active, exited
        active = True
        try:
            yield
        finally:
            active = False
            exited = True

    def predict(received, top_k):
        assert active, 'Inference must remain inside the loading skeleton'
        assert received is image
        assert top_k == 5
        if failure == 'predict':
            raise ValueError('invalid image')
        return prediction

    def load(fingerprint):
        assert active, 'Cold model loading must be inside the loading skeleton'
        assert fingerprint == 'selected-model'
        if failure == 'load':
            raise RuntimeError('load failed')
        return SimpleNamespace(error='missing model' if failure == 'unavailable' else None,
                               predict=predict)

    with patch.object(app.st, 'skeleton', skeleton), patch.object(app, 'get_recognizer', load):
        if failure:
            with pytest.raises((ValueError, RuntimeError)):
                app.predict_with_loading(image, 'selected-model')
        else:
            assert app.predict_with_loading(image, 'selected-model') is prediction
    assert exited and not active, 'Loading must clear on success and every failure'
