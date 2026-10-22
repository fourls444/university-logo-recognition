"""Local, reviewed correction storage. No automatic training."""
import csv
import hashlib
import io
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from PIL import Image

class FeedbackStore:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / 'feedback'
        (self.directory / 'images').mkdir(parents=True, exist_ok=True)
        csv_path = self.root / 'university_info.csv'
        if not csv_path.exists():
            csv_path = Path(__file__).with_name('university_info.csv')
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            self.names = {row['name'] for row in csv.DictReader(stream)}
        self.db = self.directory / 'corrections.sqlite3'
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute('CREATE TABLE IF NOT EXISTS corrections (id TEXT PRIMARY KEY, image_path TEXT, correct_name TEXT, prediction TEXT, source_group TEXT, status TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')

    def add(self, image, prediction, correct_name):
        if correct_name not in self.names | {'__unknown__'}:
            raise ValueError('Unknown university label')
        if not isinstance(image, Image.Image):
            raise TypeError('A Pillow image is required')
        from dataset import preprocess_image
        canonical = preprocess_image(image)
        buffer = io.BytesIO()
        canonical.save(buffer, format='PNG')
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
        relative = f'feedback/images/{digest}.png'
        (self.root / relative).write_bytes(buffer.getvalue())
        key = uuid.uuid4().hex
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute('INSERT INTO corrections (id,image_path,correct_name,prediction,source_group,status) VALUES (?,?,?,?,?,?)', (key,relative,correct_name,json.dumps(prediction,ensure_ascii=False),f'feedback:{digest}','pending'))
        return key

    def _rows(self, status):
        with closing(sqlite3.connect(self.db)) as con, con:
            con.row_factory = sqlite3.Row
            rows = [dict(row) for row in con.execute('SELECT * FROM corrections WHERE status=? ORDER BY created_at,id', (status,))]
        for row in rows:
            row['prediction'] = json.loads(row['prediction'])
        return rows

    def list_pending(self):
        return self._rows('pending')

    def review(self, id, approved):
        if not isinstance(approved, bool):
            raise TypeError('approved must be boolean')
        with closing(sqlite3.connect(self.db)) as con, con:
            result = con.execute('UPDATE corrections SET status=? WHERE id=?', ('approved' if approved else 'rejected',id))
            if result.rowcount == 0:
                raise KeyError(id)

    def export_approved(self):
        target = self.directory / 'approved_manifest.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps({'schema_version':1,'rows':self._rows('approved')},ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(target)
        return target
