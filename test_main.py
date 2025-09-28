from fastapi import FastAPI
from db import crud_api
from routers import img_edit, parse
from db.update_animal_img import update_animal_img

app = FastAPI()

# API 등록
app.include_router(crud_api.router)
app.include_router(parse.router)
app.include_router(img_edit.router)

# update_animal_img("429361202500596")

@app.get("/")
def root():
    return {"msg": "서버 동작 중"}