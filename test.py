import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# โหลดโมเดลที่บันทึกไว้
PROJECT_ROOT = Path(__file__).resolve().parent
model_path = PROJECT_ROOT / 'model.h5'
label_path = PROJECT_ROOT / 'label.npy'
model = load_model(model_path)

# โหลด label map ที่บันทึกไว้
reverse_label_map = np.load(label_path, allow_pickle=True).item()

# อ่านข้อมูลจากไฟล์ CSV
csv_file = PROJECT_ROOT / 'university_info.csv'
df = pd.read_csv(csv_file)

# สร้าง dictionary สำหรับชื่อมหาวิทยาลัยภาษาอังกฤษ, ภาษาไทย และ URL
university_info = df.set_index('name').to_dict()['url']
university_info_th = df.set_index('name').to_dict()['name_th']

def predict_image(image_path):
    # เปิดและปรับขนาดภาพ
    img = Image.open(image_path)
    img = img.convert('RGB')  # แปลงภาพให้เป็น RGB
    img = img.resize((256, 256))  # ใช้ขนาดเดียวกับ Notebook และ train.py
    img_array = np.array(img) / 255.0  # Normalize ข้อมูล
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติ batch

    # ทำการคาดการณ์
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions[0])
    
    # ใช้ label map เพื่อแปลง label เป็นชื่อ
    predicted_name_en = reverse_label_map.get(predicted_label, 'Unknown')
    predicted_name_th = university_info_th.get(predicted_name_en, 'Unknown')
    website_url = university_info.get(predicted_name_en, 'No URL available')
    
    return img, predicted_name_en, predicted_name_th, website_url

# ตัวอย่างการทดสอบ
test_image_path = PROJECT_ROOT / 'Image' / 'test30.png'
image, predicted_name_en, predicted_name_th, website_url = predict_image(test_image_path)

# แสดงผลลัพธ์
print(f"University (English) : {predicted_name_en}")
print(f"University (Thai) : {predicted_name_th}")
print(f"Website : {website_url}")
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.title(f"University (English): {predicted_name_en}")
plt.show()
