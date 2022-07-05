import io
import json
import torchvision.transforms as transforms
from torchvision import models
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from time import sleep
from typing import Optional
import httpx

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
model.to(device)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()
imagenet_class_index = json.load(open('static/imagenet_class_index.json'))


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))

    return my_transforms(image).unsqueeze(0)


def get_prediction(model, image_bytes, imagenet_class_index):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor.to(device))
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]



@app.get('/')
def index():
    return "hello"

@app.post("/link/")
async def create_file(link: Optional[str] = None):
    print(link)
    if not link:
        return {"message": "No link sent"}
    else:
        async with httpx.AsyncClient() as client:
            r = await client.get(link)
        prediction: list[str] = get_prediction(model=model, image_bytes=r.content, imagenet_class_index=imagenet_class_index)
        return {"prediction": prediction[1]}


@app.post("/files/")
async def create_file(file: Optional[bytes] = File(None)):
    if not file:
        return {"message": "No file sent"}
    else:
        
        prediction: list[str] = get_prediction(model=model, image_bytes=file, imagenet_class_index=imagenet_class_index)
        return {"prediction": prediction[1]}


@app.post("/uploadfile/")
async def create_upload_file(file: Optional[UploadFile] = None):

    if not file:
        return {"message": "No upload file sent"}
    else:
        prediction: list[str] = get_prediction(model=model, image_bytes=file, imagenet_class_index=imagenet_class_index)
        return {"prediction": prediction[1]}
