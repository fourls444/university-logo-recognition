# บันทึกการกู้ไฟล์โมเดลเดิม — 21 ตุลาคม 2026

ระหว่างเพิ่ม regression test มีการ import `train.py` รุ่นเดิม ซึ่งเริ่มฝึกและเขียนทับ `model.h5` กับ `label.npy` ทันทีโดยไม่ได้ตั้งใจ

- กู้ `label.npy` จาก Git ได้ตรงกับไฟล์ก่อนเริ่มงาน SHA-256: `4b5a02119d2e46f24ec7329d7cba36ff8c1f72fbb83b3c6a848f86b8ccd06250`
- ยังหาสำเนาที่ตรงกับ `model.h5` ก่อนเริ่มงานไม่พบ (hash เดิมขึ้นต้น `3A56BF6CA5B3EC2AAF2C2E00B6A326`) สำเนาในโฟลเดอร์รายวิชาเป็นคนละไฟล์ จึงไม่ได้นำมาอ้างว่าเป็นไฟล์เดิม
- เก็บโมเดลที่เกิดจากการเขียนทับไว้ที่ `artifacts/recovery/model-written-by-accidental-import.h5` เพื่อไม่ให้ใช้คู่กับ label เก่าที่กู้กลับมาโดยเข้าใจผิด
- รุ่นใหม่ฝึกและทำนายผ่าน `artifacts/versions/` กับ `artifacts/current.json` เท่านั้น ทั้ง Notebook สคริปต์ และหน้าใช้งานใช้ขั้นตอนเดียวกัน
- `train.py` และ `test.py` รุ่นใหม่ไม่เริ่มฝึกหรือโหลดโมเดลเมื่อ import และมี regression test ตรวจการเรียก help โดยไม่เปิดโมเดล

ไฟล์โมเดลเดิมที่ถูกเขียนทับไม่ได้ถูกกู้คืน การฝึกโมเดลรุ่นใหม่ไม่ใช่การกู้โมเดลนั้น


Historical paths above describe the original layout. On 2026-10-22, legacy logos moved to `data/logos/` and examples to `data/examples/`; see `data/provenance/dataset-path-mapping.json`.
