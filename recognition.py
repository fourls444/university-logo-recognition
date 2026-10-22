"""Inference using a verified versioned artifact bundle."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

class LogoRecognizer:
    def __init__(self, root):
        self.root = Path(root)
        self.labels = {}
        self.error = None
        self.model = None
        try:
            artifact = self.root / 'artifacts'
            pointer = artifact / 'current.json'
            if pointer.exists():
                version = json.loads(pointer.read_text(encoding='utf-8'))['version']
                if Path(version).name != version:
                    raise ValueError('Invalid artifact version')
                artifact = artifact / 'versions' / version
            self.metadata = json.loads((artifact/'metadata.json').read_text(encoding='utf-8'))
            if self.metadata.get('status') != 'trained':
                raise ValueError('Model artifact is not trained')
            for name in ('model.keras','labels.json'):
                digest = hashlib.sha256((artifact/name).read_bytes()).hexdigest()
                if digest != self.metadata['sha256'][name]:
                    raise ValueError('Artifact checksum mismatch')
            self.labels = {int(k):v for k,v in json.loads((artifact/'labels.json').read_text(encoding='utf-8')).items()}
            if set(self.labels) != set(range(len(self.labels))) or len(self.labels) != self.metadata['class_count']:
                raise ValueError('Artifact labels do not match class count')
            import tensorflow as tf
            self.model = tf.keras.models.load_model(artifact/'model.keras', compile=False)
            if int(self.model.output_shape[-1]) != len(self.labels):
                raise ValueError('Model output does not match labels')
        except Exception as exc:
            self.error = f'Model artifacts unavailable: {exc}'
            self.model = None
        csv_path = self.root/'university_info.csv'
        if not csv_path.exists():
            csv_path = Path(__file__).with_name('university_info.csv')
        with csv_path.open(encoding='utf-8-sig',newline='') as stream:
            self.info = {row['name']:row for row in csv.DictReader(stream)}

    def predict(self, image, top_k=5):
        if self.model is None:
            return {'candidates':[], 'uncertain':True, 'reason':self.error}
        if not isinstance(top_k,int) or top_k < 1:
            raise ValueError('top_k must be positive')
        from dataset import preprocess_image
        array = np.asarray(preprocess_image(image),dtype=np.float32)[None] / 255.0
        scores = np.asarray(self.model(array,training=False))[0]
        if scores.shape != (len(self.labels),) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1) or not np.isclose(scores.sum(),1,atol=.01):
            return {'candidates':[],'uncertain':True,'reason':'Invalid model scores'}
        order = np.argsort(-scores)
        margin = float(scores[order[0]] - scores[order[1]]) if len(order)>1 else 1.
        uncertain = float(scores[order[0]]) < .6 or margin < .15
        weak_validation = self.metadata.get('validation_accuracy', 1.0) < .7
        uncertain = uncertain or weak_validation
        candidates = []
        for index in order[:top_k]:
            name = self.labels[int(index)]
            info = self.info.get(name,{})
            candidates.append({'name':name,'name_th':info.get('name_th',name),'url':info.get('url',''),'score':float(scores[index])})
        reason = ('โมเดลรุ่นนี้ยังมีผลตรวจที่ไม่เพียงพอสำหรับการยืนยันคำตอบ กรุณาใช้รายการนี้เป็นตัวเลือกให้ตรวจทาน'
                  if weak_validation else
                  'คะแนนยังต่ำหรืออันดับแรกใกล้เคียงกัน เกณฑ์นี้ยังไม่ได้ปรับเทียบจากชุดทดสอบอิสระ'
                  if uncertain else 'ตัวเลือกที่ได้คะแนนสูงสุด กรุณาตรวจสอบผลก่อนนำไปใช้')
        return {'candidates':candidates,'uncertain':uncertain,'reason':reason}
