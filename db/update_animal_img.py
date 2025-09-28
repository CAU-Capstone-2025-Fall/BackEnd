import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient

load_dotenv()

def update_animal_img(desertion_no: str):
    # DB 연결
    try:
        uri = os.getenv("MONGODB_URI")
        client = MongoClient(uri)
        db = client["testdb"]
        collection = db["abandoned_animals"]
        print ("db connected")
    except Exception as e:
        print("db connection error:", e)

    animal = collection.find_one({"desertionNo": desertion_no})
    if not animal:
        raise HTTPException(status_code=404, detail="Animal not found")
    img = animal.get("popfile1")

    url = "http://3.38.48.153:8000/image/edit"
    # url = "http://localhost:8000/image/edit" # 로컬 테스트용
    params = {
        #프롬프트
        "prompt" : "Edit this photo of an animal. If the animal is dirty, gently clean the animal by removing any mud, stains, or foreign substances — without altering the fur color, body shape, breed, or facial structure in any way. Preserve the animal's original appearance as much as possible, and do not hide or remove its unique traits or medical conditions. For example, if there is a small lipoma on the right side, or any visible condition, these must remain clearly visible and true to reality. Maintain the existing background and setting, but enhance it by improving lighting, sharpness, and cleanliness. Do not generate a different animal. Do not recolor the fur, eyes, or ears. Do not remove or hide unique characteristics, injuries, disabilities, or medical features. Do not add accessories, collars, or extra limbs. Do not apply any cartoon, painted, or unrealistic styles. The final image should remain photorealistic and true to the original scene, while clearly representing the animal's real features.", 
        "image_url" : img,
    }
    response = requests.post(url, json=params)
    if response.status_code != 200:
        print("API 응답:", response.text)
        raise HTTPException(status_code=500, detail="External API error")
    data = response.json()
    created_img = data.get("edited_image_url")
    if not created_img:
        print("API 응답:", response.text)
        raise HTTPException(status_code=500, detail="No image returned")

    collection.update_one(
        {"desertionNo": desertion_no},
        {"$set": {"createdImg": created_img}}
    )
    print("Image updated successfully")

if __name__ == "__main__":
    update_animal_img("444457202500537")  # 테스트용 유기동물 번호