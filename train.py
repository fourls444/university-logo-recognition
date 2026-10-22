"""Explicit, reproducible CPU baseline training; importing never trains."""
import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def publish_artifacts(root, model, labels, metadata, promote=True):
    artifact = Path(root)/'artifacts'
    version = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    destination = artifact/'versions'/version
    destination.mkdir(parents=True)
    model.save(destination/'model.keras')
    (destination/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf-8')
    record = dict(metadata, class_count=len(labels), created_at=datetime.now(timezone.utc).isoformat())
    record['sha256'] = {name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in ('model.keras','labels.json')}
    (destination/'metadata.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    if not promote:
        return destination
    # This single small replacement promotes the complete immutable bundle atomically.
    temporary = artifact/('current-'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps({'version':version}),encoding='utf-8')
    os.replace(temporary,artifact/'current.json')
    return destination


def reviewed_feedback_rows(root, train_rows, validation_rows, labels):
    """Exclude held-out exact/perceptual matches; never import unknown or pending rows."""
    from feedback import FeedbackStore
    from dataset import preprocess_image
    from PIL import Image
    root = Path(root)
    manifest = json.loads(FeedbackStore(root).export_approved().read_text(encoding='utf-8'))
    reverse = {name:index for index,name in labels.items()}
    def fingerprint(path):
        with Image.open(path) as image:
            array = np.asarray(preprocess_image(image).resize((32,32)).convert('RGB'),dtype=np.float32)/255
            return array
    held_out = [fingerprint(path) for path,_ in validation_rows]
    seen = {hashlib.sha256(np.asarray(fingerprint(path)).tobytes()).hexdigest() for path,_ in train_rows}
    additions = []
    skipped = 0
    group_labels = {}
    for row in manifest['rows']:
        group_labels.setdefault(row['source_group'], set()).add(row['correct_name'])
    for row in manifest['rows']:
        if row['correct_name'] not in reverse or len(group_labels[row['source_group']]) > 1:
            skipped += 1
            continue
        path = (root/row['image_path']).resolve()
        if not path.is_relative_to((root/'feedback'/'images').resolve()):
            raise ValueError('Feedback path escapes image directory')
        array = fingerprint(path)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if digest in seen or any(float(np.mean(np.abs(array-other))) < .04 for other in held_out):
            skipped += 1
            continue
        seen.add(digest)
        additions.append((path,reverse[row['correct_name']]))
    return additions, skipped


def training_arrays(rows, seed=42):
    """Keep clean source images alongside two deterministic training-only views."""
    from dataset import load_images
    arrays = [load_images(rows,augment=False,seed=seed)]
    arrays.extend(load_images(rows,augment=True,seed=view_seed) for view_seed in (seed,seed+1))
    return np.concatenate([x for x,_ in arrays]), np.concatenate([y for _,y in arrays])


def build_model(class_count, weights='imagenet'):
    """Frozen pretrained representation with a small trainable classification head."""
    import tensorflow as tf
    if not isinstance(class_count, int) or class_count < 1:
        raise ValueError('class_count must be positive')
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(256,256,3), include_top=False, pooling='avg', weights=weights)
    except Exception as exc:
        raise RuntimeError('MobileNetV2 pretrained weights unavailable. Connect to download official Keras weights or restore the local Keras weight cache, then retry.') from exc
    base.trainable = False
    extractor = tf.keras.Sequential([
        tf.keras.layers.Input((256,256,3)),
        tf.keras.layers.Rescaling(2., offset=-1., name='mobilenet_scaling'),
        base,
    ], name='feature_extractor')
    extractor.trainable = False
    head = build_cosine_head(class_count)
    return tf.keras.Sequential([
        tf.keras.layers.Input((256,256,3)), extractor, head,
    ], name='university_logo_transfer')


def build_cosine_head(class_count):
    """A linear classifier over normalized frozen features for few-shot logos."""
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input((1280,)),
        tf.keras.layers.UnitNormalization(axis=-1),
        tf.keras.layers.Dense(class_count,use_bias=False,activation='softmax',name='class_prototypes'),
    ],name='classifier')


def initialize_prototypes(head, features, labels):
    """Initialize each class exclusively from its training feature mean."""
    normalized = features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-9)
    class_count = head.output_shape[-1]
    if set(np.asarray(labels).tolist()) != set(range(class_count)):
        raise ValueError('Every class needs a training feature for prototype initialization')
    prototypes = np.stack([normalized[labels==index].mean(axis=0) for index in range(class_count)])
    prototypes /= np.maximum(np.linalg.norm(prototypes,axis=1,keepdims=True),1e-9)
    head.get_layer('class_prototypes').set_weights([(10.*prototypes.T).astype(np.float32)])


def train_head(head, features, labels, validation_features, validation_labels, epochs, verbose=2, learning_rate=.0003):
    """Stop the small head when validation loss stalls and restore its best epoch."""
    import tensorflow as tf
    head.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return head.fit(features,labels,validation_data=(validation_features,validation_labels),
                    epochs=epochs,batch_size=16,verbose=verbose,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)])


def train_model(root=None, epochs=10, seed=42, include_feedback=False, promote=True):
    if not isinstance(epochs,int) or epochs < 1 or epochs > 100:
        raise ValueError('epochs must be an integer from 1 to 100')
    root = Path(root or Path(__file__).resolve().parent)
    from dataset import prepare_dataset, load_images
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    tf.keras.utils.set_random_seed(seed)
    train_rows, validation_rows, labels = prepare_dataset(root,seed=seed)
    additions, skipped = reviewed_feedback_rows(root,train_rows,validation_rows,labels) if include_feedback else ([],0)
    train_rows = list(train_rows)+additions
    if {label for _,label in train_rows} != set(labels):
        raise ValueError('Every university requires training images')
    x,y = training_arrays(train_rows,seed=seed)
    vx,vy = load_images(validation_rows,augment=False,seed=seed)
    if not len(x) or not len(vx):
        raise ValueError('Training and validation images are both required')
    model = build_model(len(labels))
    extractor = model.get_layer('feature_extractor')
    head = model.get_layer('classifier')
    # Cache frozen features once; only the small head trains each epoch.
    features = np.concatenate([np.asarray(extractor(x[start:start+16], training=False)) for start in range(0,len(x),16)])
    validation_features = np.concatenate([np.asarray(extractor(vx[start:start+16], training=False)) for start in range(0,len(vx),16)])
    initialize_prototypes(head,features,y)
    history = train_head(head,features,y,validation_features,vy,epochs)
    scores = np.asarray(head(validation_features,training=False))
    predicted = scores.argmax(axis=1)
    confusion = np.zeros((len(labels),len(labels)),dtype=int)
    for true,guess in zip(vy,predicted):
        confusion[int(true),int(guess)] += 1
    classes = {}
    for index,name in labels.items():
        index = int(index)
        tp = int(confusion[index,index])
        classes[name] = {'train_sources':sum(label==index for _,label in train_rows),'validation_sources':int((vy==index).sum()),'precision':tp/int(confusion[:,index].sum()) if confusion[:,index].sum() else None,'recall':tp/int(confusion[index].sum()) if confusion[index].sum() else None}
    metadata = {'status':'trained','seed':seed,'epochs':len(history.history['loss']),'epochs_requested':epochs,'epochs_run':len(history.history['loss']),'best_epoch':int(np.argmin(history.history['val_loss']))+1,'early_stopping':{'monitor':'val_loss','patience':8,'restore_best_weights':True},'architecture':'frozen ImageNet MobileNetV2, internal [-1,1] rescaling, global average pooling, normalized prototype-initialized linear head; cached base features','train_source_count':len(train_rows),'train_array_count':len(x),'validation_count':len(vx),'validation_accuracy':float(np.mean(predicted==vy)),'history':{key:[float(value) for value in values] for key,values in history.history.items()},'class_metrics':classes,'confusion_matrix':confusion.tolist(),'feedback_added':len(additions),'feedback_skipped':skipped,'feedback_policy':'approved known labels only; canonical near-duplicates of validation excluded; source identity for novel user uploads remains uncertain','uncertainty':{'threshold':.6,'margin':.15,'calibrated':False}}
    manifest_path = root/'dataset_manifest.json'
    metadata['validation_top5_accuracy'] = float(np.mean([truth in ranking for truth,ranking in zip(vy,np.argsort(-scores,axis=1)[:,:5])]))
    metadata['training_views'] = {'clean_originals':True,'augmented_views_per_source':2}
    metadata['classifier'] = {'initialization':'normalized training-only class means, scale10','learning_rate':.0003,'score_type':'uncalibrated softmax model score'}
    if manifest_path.exists():
        metadata['dataset_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return publish_artifacts(root,model,labels,metadata,promote=promote)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--include-feedback',action='store_true',help='Include approved known-label corrections; exclude validation matches')
    parser.add_argument('--no-promote',action='store_true',help='Save a candidate bundle without changing the active model')
    args = parser.parse_args()
    print(train_model(args.root,args.epochs,args.seed,args.include_feedback,promote=not args.no_promote))

if __name__ == '__main__':
    main()
