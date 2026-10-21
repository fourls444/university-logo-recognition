import numpy as np
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# กำหนดพาธของชุดข้อมูล
PROJECT_ROOT = Path(__file__).resolve().parent
data_path = PROJECT_ROOT / 'Logo'
csv_file = PROJECT_ROOT / 'university_info.csv'
model_path = PROJECT_ROOT / 'model.h5'
label_path = PROJECT_ROOT / 'label.npy'

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv(csv_file)

# สร้าง dictionary สำหรับชื่อมหาวิทยาลัยภาษาอังกฤษ, ภาษาไทย และ URL
university_info = df.set_index('name').to_dict()['url']
university_info_th = df.set_index('name').to_dict()['name_th']
reverse_label_map = {i: name for i, name in enumerate(df['name'])}

# เตรียมข้อมูล
labels = []
images = []
label_map = {}  # ใช้เพื่อจัดการกับชื่อโฟลเดอร์เป็น label

for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    if os.path.isdir(folder_path):
        # เพิ่ม label ใหม่ถ้ายังไม่มี
        if folder_name not in label_map:
            label_map[folder_name] = len(label_map)

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):  # ตรวจสอบว่าคือไฟล์ไม่ใช่โฟลเดอร์
                try:
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')  # แปลงภาพให้เป็น RGB
                        img = img.resize((256, 256))  # ใช้ขนาดเดียวกับ Notebook
                        images.append(np.array(img))

                        # ใช้ชื่อโฟลเดอร์เป็น label
                        label = label_map[folder_name]
                        labels.append(label)
                except Exception as e:
                    print(f"Error opening image {image_name}: {e}")

# ตรวจสอบว่ามีข้อมูลใน images และ labels
if not images or not labels:
    raise ValueError("No images or labels were loaded. Please check the file paths and image formats.")

# แปลงเป็นอาร์เรย์
images = np.array(images)
labels = np.array(labels)

# ทำการ Normalize ข้อมูล
images = images / 255.0

# แปลง labels เป็น categorical
num_classes = len(set(labels))  # นับจำนวนคลาสที่มีใน labels
labels = to_categorical(labels, num_classes=num_classes)

# สร้างโมเดล
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ฝึกอบรมโมเดล
model.fit(images, labels, epochs=10)

# บันทึกโมเดลและ label map
model.save(model_path)
np.save(label_path, reverse_label_map)
