import csv
import json

with open("firebase_image_urls.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    data = [{"noticeNo": row[0], "createdImg": row[1]} for row in reader]

with open("urls.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
