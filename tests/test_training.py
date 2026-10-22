"""Regression coverage for bounded training experiments and original-image retention."""
import json
import numpy as np
from PIL import Image


def test_training_arrays_keep_clean_originals_and_reproducible_variants(tmp_path):
    from train import training_arrays
    from dataset import load_images
    image = Image.new('RGB',(33,19),'white')
    for x in range(8,25):
        for y in range(4,15):
            image.putpixel((x,y),(30,80,160))
    path=tmp_path/'logo.png'
    image.save(path)
    before=path.read_bytes()
    rows=[(path,2)]
    clean,_=load_images(rows,augment=False)
    images,labels=training_arrays(rows,seed=42)
    assert images.shape == (3,256,256,3)
    np.testing.assert_array_equal(images[:1],clean)
    np.testing.assert_array_equal(labels,[2,2,2])
    assert not np.array_equal(images[1],clean[0])
    again,_=training_arrays(rows,seed=42)
    np.testing.assert_array_equal(images,again)
    assert path.read_bytes() == before


def test_experimental_artifacts_leave_active_pointer_unchanged(tmp_path):
    import tensorflow as tf
    from train import publish_artifacts
    model=tf.keras.Sequential([tf.keras.layers.Input((2,)),tf.keras.layers.Dense(2,activation='softmax')])
    active=publish_artifacts(tmp_path,model,{0:'a',1:'b'},{'status':'trained'})
    before=(tmp_path/'artifacts/current.json').read_bytes()
    candidate=publish_artifacts(tmp_path,model,{0:'a',1:'b'},{'status':'trained','exploratory':True},promote=False)
    assert active != candidate
    assert (candidate/'model.keras').exists()
    assert (tmp_path/'artifacts/current.json').read_bytes() == before

def test_cosine_head_initializes_from_training_means_and_ignores_feature_scale():
    from train import build_cosine_head, initialize_prototypes
    head=build_cosine_head(2)
    features=np.zeros((4,1280),dtype=np.float32)
    features[:2,0]=[1.,3.]
    features[2:,1]=[2.,4.]
    initialize_prototypes(head,features,np.array([0,0,1,1]))
    probes=np.zeros((2,1280),dtype=np.float32)
    probes[0,0]=1
    probes[1,1]=1
    scores=np.asarray(head(probes,training=False))
    np.testing.assert_array_equal(scores.argmax(axis=1),[0,1])
    np.testing.assert_allclose(scores,np.asarray(head(probes*100,training=False)),atol=1e-6)
    assert scores.min() >= 0 and np.allclose(scores.sum(axis=1),1)

def test_default_model_classifier_does_not_confuse_embedding_magnitude_with_evidence():
    from train import build_model
    model=build_model(3,weights=None)
    head=model.get_layer('classifier')
    features=np.ones((1,1280),dtype=np.float32)
    np.testing.assert_allclose(np.asarray(head(features,training=False)),np.asarray(head(features*10,training=False)),atol=1e-6)

def test_train_model_non_promoting_run_preserves_existing_bundle(tmp_path, monkeypatch):
    """Dropping the train_model promote flag must catch an overwritten pointer."""
    import hashlib
    import tensorflow as tf
    import train
    (tmp_path/'university_info.csv').write_text('name\na\nb\n', encoding='utf-8')
    entries=[]
    for label, color in [('a','red'),('b','blue')]:
        for split in ['train','validation']:
            file=f'{label}-{split}.png'
            image=Image.new('RGB',(16,16),color)
            if split=='validation': image.putpixel((0,0),(0,255,0))
            image.save(tmp_path/file)
            entries.append({'file':file,'university':label,'split':split,'source_group':file})
    (tmp_path/'dataset_manifest.json').write_text(json.dumps({'seed':42,'images':entries}))
    def small_model(count):
        extractor=tf.keras.Sequential([tf.keras.layers.Input((256,256,3)),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(1280)],name='feature_extractor')
        extractor.trainable=False
        return tf.keras.Sequential([tf.keras.layers.Input((256,256,3)),extractor,train.build_cosine_head(count)])
    monkeypatch.setattr(train,'build_model',small_model)
    baseline=train.publish_artifacts(tmp_path,small_model(2),{0:'a',1:'b'},{'status':'baseline'})
    before=(tmp_path/'artifacts/current.json').read_bytes()
    digest=hashlib.sha256((baseline/'model.keras').read_bytes()).hexdigest()
    candidate=train.train_model(tmp_path,epochs=1,promote=False)
    assert candidate != baseline
    assert json.loads((candidate/'metadata.json').read_text())['validation_count']==2
    assert (tmp_path/'artifacts/current.json').read_bytes()==before
    assert hashlib.sha256((baseline/'model.keras').read_bytes()).hexdigest()==digest
