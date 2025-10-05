import json
import os

import requests

# JSON 데이터를 파일에 저장해놨다고 가정 (animals.json)
with open("animals.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 저장 폴더
save_dir = "animal_images"
os.makedirs(save_dir, exist_ok=True)

for item in data:
    notice_no = item.get("noticeNo")
    img_url = item.get("popfile1")

    if not notice_no or not img_url:
        continue  # 값이 없는 경우 스킵

    try:
        # 이미지 다운로드
        resp = requests.get(img_url, timeout=10)
        resp.raise_for_status()

        # 확장자 추출 (jpg/png 등)
        ext = os.path.splitext(img_url)[-1]
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"

        # 저장 경로
        filename = f"{notice_no}{ext}"
        filepath = os.path.join(save_dir, filename)

        with open(filepath, "wb") as f:
            f.write(resp.content)

        print(f"✅ Saved: {filepath}")
    except Exception as e:
        print(f"❌ Failed: {notice_no} ({img_url}) → {e}")
