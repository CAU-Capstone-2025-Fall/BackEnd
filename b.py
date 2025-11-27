import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)

db = client["testdb"]
collection = db["abandoned_animals"]

target_no = "441394202501660"

result = collection.update_one(
    {"desertionNo": target_no},
    {"$set": {"createdImg": None}}
)

print("matched:", result.matched_count)
print("modified:", result.modified_count)
