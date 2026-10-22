"""ลองทำนายภาพด้วยโมเดลที่ฝึกจาก train.py โดยไม่เริ่มงานตอน import"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Predict university logo candidates')
    parser.add_argument('--image', type=Path, default=Path(__file__).resolve().parent / 'data/examples/king-mongkut-s-institute-of-technology-ladkrabang-example-01.png')
    args = parser.parse_args()
    from documents import load_image
    from recognition import LogoRecognizer
    recognizer = LogoRecognizer(Path(__file__).resolve().parent)
    if recognizer.error:
        parser.exit(1, f'{recognizer.error}\nRun: python train.py --epochs 10\n')
    try:
        prediction = recognizer.predict(load_image(args.image.read_bytes()))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    print('คะแนนโมเดลใช้จัดอันดับ ไม่ใช่เปอร์เซ็นต์ความคล้ายของภาพ')
    if prediction['uncertain']:
        print('ยังไม่มั่นใจ:', prediction['reason'])
    for i, candidate in enumerate(prediction['candidates'], 1):
        print(f"{i}. {candidate['name_th']} ({candidate['name']}): {candidate['score']:.4f}")


if __name__ == '__main__':
    main()
