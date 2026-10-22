"""Run a bounded, family-balanced augmentation experiment without promotion.

The fixed validation set is reused for selection; it is not an independent test.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def augmentation_rows(root, manifest_path, source_entries, labels):
    root=Path(root).resolve()
    held={(r['university'],r['source_group']) for r in source_entries if r['split']=='validation'}
    train={(r['university'],r['source_group'],r['file']):r for r in source_entries if r['split']=='train'}
    reverse={v:int(k) for k,v in labels.items()}
    rows=[]
    for entry in json.loads(Path(manifest_path).read_text(encoding='utf-8'))['images']:
        family=entry['university'],entry['source_group']
        if entry.get('split')!='train' or family in held or (*family,entry['parent_file']) not in train:
            raise ValueError('Augmentation parent must belong to an exclusive training family')
        for key, digest in [('file','sha256'),('parent_file','parent_sha256')]:
            path=(root/entry[key]).resolve()
            if not path.is_relative_to(root): raise ValueError('Augmentation path escapes dataset root')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry[digest]: raise ValueError('Augmentation digest mismatch')
        rows.append(((root/entry['file']).resolve(),reverse[entry['university']],family,entry))
    return rows


def run(root=ROOT, epochs=5, seed=42):
    import tensorflow as tf
    from dataset import prepare_dataset,load_images
    from train import build_cosine_head,initialize_prototypes,train_head,publish_artifacts,training_arrays
    root=Path(root).resolve()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(seed)
    pointer=root/'artifacts/current.json'; before=pointer.read_bytes()
    baseline_dir=root/'artifacts/versions'/json.loads(before)['version']
    baseline_sha=hashlib.sha256((baseline_dir/'model.keras').read_bytes()).hexdigest()
    experiment=root/'artifacts/experiments'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    experiment.mkdir(parents=True)
    manifest=root/'dataset_manifest.json'
    entries=json.loads(manifest.read_text(encoding='utf-8'))['images']
    aug_manifest=root/'data/candidates/augmentation_20261022/manifests/manifest.json'
    train_rows,val_rows,labels=prepare_dataset(root,seed)
    additions=augmentation_rows(root,aug_manifest,entries,labels)
    allowed={str(path.resolve()) for path,_ in train_rows}
    if any(str((root/r[3]['parent_file']).resolve()) not in allowed for r in additions):
        raise ValueError('Augmentation parent excluded by effective duplicate-safe split')
    baseline=tf.keras.models.load_model(baseline_dir/'model.keras',compile=False)
    extractor=baseline.get_layer('feature_extractor')
    def features(rows,views=False):
        x,y=training_arrays(rows,seed) if views else load_images(rows,seed=seed)
        result=np.concatenate([np.asarray(extractor(x[i:i+8],training=False)) for i in range(0,len(x),8)])
        return result,y
    print('Extracting original training views',flush=True)
    original_features,original_labels=features(train_rows,True)
    print('Extracting 180 additional derived training views',flush=True)
    derived_features,_=features([(p,y) for p,y,_,_ in additions])
    val_features,vy=features(val_rows)
    def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)
    source_families={str((root/r['file']).resolve()):(r['university'],r['source_group']) for r in entries}
    family_features=[]; family_labels=[]; provenance=[]
    repeated_rows=train_rows*3
    for family in sorted({source_families[str(p.resolve())] for p,_ in train_rows}):
        original_indices=[i for i,(p,_) in enumerate(repeated_rows) if source_families[str(p.resolve())]==family]
        derived_indices=[i for i,r in enumerate(additions) if r[2]==family]
        original_mean=unit(original_features[original_indices]).mean(axis=0)
        # Equal weight to original-source views and the six new derived views per family.
        mean=.5*original_mean+.5*unit(derived_features[derived_indices]).mean(axis=0) if derived_indices else original_mean
        family_features.append(mean); family_labels.append(labels and original_labels[original_indices[0]])
        provenance.append({'university':family[0],'source_group':family[1],'original_files':sorted({str(p.relative_to(root)).replace('\\','/') for p,_ in train_rows if source_families[str(p.resolve())]==family}),'derived_files':[additions[i][3] for i in derived_indices]})
    fx=np.asarray(family_features); fy=np.asarray(family_labels)
    head=build_cosine_head(len(labels)); initialize_prototypes(head,fx,fy)
    history=train_head(head,fx,fy,val_features,vy,epochs,learning_rate=.0003)
    candidate=tf.keras.Sequential([tf.keras.layers.Input((256,256,3)),extractor,head],name='university_logo_transfer')
    def metrics(scores):
        pred=scores.argmax(axis=1)
        return {'correct':int((pred==vy).sum()),'count':len(vy),'top1_accuracy':float((pred==vy).mean()),'top5_accuracy':float(np.mean([y in ranks for y,ranks in zip(vy,np.argsort(-scores,axis=1)[:,:5])])),'predicted':pred.tolist()}
    baseline_scores=np.asarray(baseline.get_layer('classifier')(val_features,training=False))
    scores=np.asarray(head(val_features,training=False))
    baseline_metrics=metrics(baseline_scores); candidate_metrics=metrics(scores)
    record={'status':'trained','experiment':'family-balanced training augmentation','architecture':'frozen ImageNet MobileNetV2, normalized prototype-initialized head','seed':seed,'epochs_requested':epochs,'epochs_run':len(history.history['loss']),'best_epoch':int(np.argmin(history.history['val_loss']))+1,'history':{k:[float(v) for v in values] for k,values in history.history.items()},'train_source_count':len(train_rows),'training_families':len(fx),'derived_training_images':len(additions),'independent_new_training_sources':0,'feedback_added':0,'validation_count':len(vy),'validation_accuracy':candidate_metrics['top1_accuracy'],'validation_top5_accuracy':candidate_metrics['top5_accuracy'],'dataset_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'augmentation_manifest_sha256':hashlib.sha256(aug_manifest.read_bytes()).hexdigest(),'training_config':{'feature_extractor_trainable':False,'batchnorm_training':False,'training_source_views':3,'derived_views_per_family':6,'family_weighting':'one mean per source family; original and derived feature means weighted 0.5 each','learning_rate':.0003,'head_batch_size':16,'early_stopping':{'monitor':'val_loss','patience':8,'restore_best_weights':True}},'baseline':{'version':baseline_dir.name,'model_sha256':baseline_sha,'metrics':baseline_metrics},'candidate_metrics':candidate_metrics,'evaluation_policy':'Same persisted 31-image validation collection used for early stopping and candidate selection; not independent real-world testing; synthetic derivatives are not new independent sources. No automatic promotion.','independent_test':{'available':False,'count':0},'training_sources':provenance,'validation_sources':[{'file':str(p.relative_to(root)).replace('\\','/'),'label':int(y)} for p,y in val_rows],'uncertainty':{'threshold':.6,'margin':.15,'calibrated':False}}
    destination=publish_artifacts(root,candidate,labels,record,promote=False)
    unchanged=pointer.read_bytes()==before and hashlib.sha256((baseline_dir/'model.keras').read_bytes()).hexdigest()==baseline_sha
    record['candidate_version']=destination.name
    record['active_pointer_unchanged']=unchanged
    record['pointer_before']=json.loads(before); record['pointer_after']=json.loads(pointer.read_bytes())
    record['active_model_sha256_after']=hashlib.sha256((baseline_dir/'model.keras').read_bytes()).hexdigest()
    (experiment/'comparison.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(experiment/'validation_predictions.npz',truth=vy,baseline=baseline_scores,candidate=scores)
    if not unchanged: raise RuntimeError('Active model changed during isolated experiment')
    print(json.dumps({'candidate':str(destination),'comparison':str(experiment/'comparison.json'),'baseline':baseline_metrics,'candidate_metrics':candidate_metrics,'active_unchanged':unchanged}),flush=True)
    return destination


def configure_fine_tuning(model):
    """Expose the final convolution block while all BN statistics stay frozen."""
    import tensorflow as tf
    extractor=model.get_layer('feature_extractor')
    extractor.trainable=True
    base=extractor.layers[-1]
    base.trainable=True
    for layer in base.layers:
        layer.trainable=(layer.name.startswith('block_16_') or layer.name=='Conv_1') and not isinstance(layer,tf.keras.layers.BatchNormalization)
    model.get_layer('classifier').trainable=True


def run_fine_tuning(root=ROOT,epochs=3,seed=42):
    import tensorflow as tf
    from dataset import prepare_dataset,load_images
    from train import publish_artifacts
    root=Path(root).resolve()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(seed)
    pointer=root/'artifacts/current.json'; before=pointer.read_bytes()
    baseline_dir=root/'artifacts/versions'/json.loads(before)['version']
    baseline_sha=hashlib.sha256((baseline_dir/'model.keras').read_bytes()).hexdigest()
    experiment=root/'artifacts/experiments'/('finetune-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    experiment.mkdir(parents=True)
    train_rows,val_rows,labels=prepare_dataset(root,seed)
    entries=json.loads((root/'dataset_manifest.json').read_text(encoding='utf-8'))['images']
    lookup={str((root/r['file']).resolve()):r for r in entries}
    families={}
    for path,label in train_rows:
        row=lookup[str(path.resolve())]
        families.setdefault((row['university'],row['source_group']),[]).append((path,label))
    vx,vy=load_images(val_rows)
    model=tf.keras.models.load_model(baseline_dir/'model.keras',compile=False)
    def score():
        return np.concatenate([np.asarray(model(vx[i:i+8],training=False)) for i in range(0,len(vx),8)])
    def metrics(scores):
        pred=scores.argmax(axis=1)
        return {'correct':int((pred==vy).sum()),'count':len(vy),'top1_accuracy':float((pred==vy).mean()),'top5_accuracy':float(np.mean([y in ranks for y,ranks in zip(vy,np.argsort(-scores,axis=1)[:,:5])])),'loss':float(np.mean(-np.log(np.maximum(scores[np.arange(len(vy)),vy],1e-9)))),'predicted':pred.tolist()}
    baseline_scores=score(); baseline_metrics=metrics(baseline_scores)
    configure_fine_tuning(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    augmentation=tf.keras.Sequential([tf.keras.layers.RandomRotation(.02,fill_mode='constant',fill_value=1.,seed=seed),tf.keras.layers.RandomTranslation(.04,.04,fill_mode='constant',fill_value=1.,seed=seed+1),tf.keras.layers.RandomZoom((-.08,.08),fill_mode='constant',fill_value=1.,seed=seed+2),tf.keras.layers.RandomContrast(.15,seed=seed+3)])
    rng=np.random.default_rng(seed)
    best_loss=float('inf'); best_weights=None; best_epoch=0; stalled=0
    history=[]; sampled=[]
    for epoch in range(epochs):
        rows=[members[int(rng.integers(len(members)))] for _,members in sorted(families.items())]
        rng.shuffle(rows)
        x,y=load_images(rows)
        # Draw new augmentation for each minibatch on this epoch; each family contributes once.
        epoch_loss=[]; epoch_accuracy=[]
        for start in range(0,len(x),4):
            batch=augmentation(x[start:start+4],training=True)
            result=model.train_on_batch(batch,y[start:start+4],return_dict=True)
            epoch_loss.append(float(result['loss'])); epoch_accuracy.append(float(result['accuracy']))
        scores=score(); measured=metrics(scores)
        history.append({'epoch':epoch+1,'loss':epoch_loss[-1],'accuracy':epoch_accuracy[-1],'val_loss':measured['loss'],'val_accuracy':measured['top1_accuracy']})
        sampled.append([{'file':str(path.relative_to(root)).replace(chr(92),'/'),'label':int(label)} for path,label in rows])
        print(json.dumps(history[-1]),flush=True)
        model.reset_metrics()
        if measured['loss']<best_loss:
            best_loss=measured['loss']; best_weights=model.get_weights(); best_epoch=epoch+1; stalled=0
        else:
            stalled+=1
            if stalled>=2: break
    model.set_weights(best_weights)
    scores=score(); measured=metrics(scores)
    source_records=[dict(lookup[str(path.resolve())],sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path,_ in train_rows]
    record={'status':'trained','experiment':'last MobileNet block fine-tuning with online augmentation','architecture':'ImageNet MobileNetV2 baseline, last block_16 and Conv_1 plus classifier trainable; BatchNormalization frozen','seed':seed,'epochs_requested':epochs,'epochs_run':len(history),'best_epoch':best_epoch,'history':history,'train_source_count':len(train_rows),'training_families':len(families),'train_samples_per_epoch':len(families),'independent_new_training_sources':0,'feedback_added':0,'validation_count':len(vy),'validation_accuracy':measured['top1_accuracy'],'validation_top5_accuracy':measured['top5_accuracy'],'dataset_manifest_sha256':hashlib.sha256((root/'dataset_manifest.json').read_bytes()).hexdigest(),'training_config':{'initial_weights':'active baseline, not augmentation candidate','trainable_base_layers':[layer.name for layer in model.get_layer('feature_extractor').layers[-1].layers if layer.trainable],'batchnorm_training':False,'learning_rate':1e-5,'batch_size':4,'family_sampling':'one random original per source family per epoch, equally weighted','online_augmentation':{'rotation_turns':.02,'translation_fraction':.04,'zoom_fraction':.08,'contrast':.15,'fill':'white'},'early_stopping':{'monitor':'val_loss','patience':2,'restore_best_weights':True}},'baseline':{'version':baseline_dir.name,'model_sha256':baseline_sha,'metrics':baseline_metrics},'candidate_metrics':measured,'training_sources':source_records,'sampled_training_sources_per_epoch':sampled,'validation_sources':[dict(lookup[str(path.resolve())],label=int(label)) for path,label in val_rows],'evaluation_policy':'Fixed 31-image validation reused for early stopping and selection, not independent real-world testing. Online augmentation adds no independent sources. Candidate is not promoted.','independent_test':{'available':False,'count':0},'uncertainty':{'threshold':.6,'margin':.15,'calibrated':False}}
    destination=publish_artifacts(root,model,labels,record,promote=False)
    unchanged=pointer.read_bytes()==before and hashlib.sha256((baseline_dir/'model.keras').read_bytes()).hexdigest()==baseline_sha
    record.update(candidate_version=destination.name,active_pointer_unchanged=unchanged,pointer_before=json.loads(before),pointer_after=json.loads(pointer.read_bytes()),active_model_sha256_after=hashlib.sha256((baseline_dir/'model.keras').read_bytes()).hexdigest())
    (experiment/'comparison.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(experiment/'validation_predictions.npz',truth=vy,baseline=baseline_scores,candidate=scores)
    if not unchanged: raise RuntimeError('Active model changed during experiment')
    print(json.dumps({'candidate':str(destination),'comparison':str(experiment/'comparison.json'),'baseline':baseline_metrics,'candidate_metrics':measured,'active_unchanged':unchanged}),flush=True)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT); parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--fine-tune',action='store_true')
    args=parser.parse_args()
    if not 1<=args.epochs<=5: parser.error('Experiment supports one to five bounded epochs')
    (run_fine_tuning if args.fine_tune else run)(args.root,args.epochs)
