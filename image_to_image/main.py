import io
import json
import os
from re import A
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model import Unet_plus_plus
import numpy as np
import cv2
import base64
import datetime
from datetime import timedelta
from loguru import logger

app = Flask(__name__)
CORS(app)  # 解决跨域问题
app.send_file_max_age_default = timedelta(seconds=1)

#记录日志
logger.add('./log/log.log')

# 加载模型
model_path = './detect.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet_plus_plus().to(device)   # Unet++
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512,512)),
        ])

def get_predict(image_bytes):
    try:
        # Image读取
        image = Image.open(io.BytesIO(image_bytes))
        # # cv2读取
        # image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        # image = Image.fromarray(image)
        
        if image.mode == "RGBA":
            image=image.convert('RGB')
        
        # 保存到 temp文件中，作为临时文件
        curr_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H-%M-%S')
        image.save(f"./temp/{curr_time}.jpg")
        log_str = f"user ip ({request.remote_addr})({request.environ['REMOTE_ADDR']}) save a image!!! "
        logger.info(log_str)

        if image.mode != "RGB":
            raise ValueError(f"input file does not RGB image... but {image.mode} image!!!")
        image = trans(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image).squeeze(0)
        prediction = outputs.cpu().numpy()[0]
        prediction = prediction > 0.5
        # cv2.imshow("1",(prediction * 255).astype(np.uint8))
        # cv2.waitKey()
        pil_img = Image.fromarray((prediction * 255).astype(np.uint8))
        pil_img = pil_img.convert("1")

        buffer=io.BytesIO()
        pil_img.save(buffer,"PNG") # 将Image对象转为二进制存入buffer。因BytesIO()是在内存中操作，所以实际是存入内存
        buf_bytes=buffer.getvalue() # 从内存中取出bytes类型的图片
        base64_data='data:image/png;base64,'+str(base64.b64encode(buf_bytes),'utf-8') # 1.txt中的内容和base64_data2一样，也可以用base64_data3的写法
        # base64_data3=(b'data:image/png;base64,'+base64_data).decode('utf-8')
        return_info = {"result":base64_data}

    except Exception as e:
        log_str = f"user ip ({request.remote_addr})({request.environ['REMOTE_ADDR']}) produce a erro:{e}!!! "
        logger.error(log_str)
        return None
    return return_info


@app.route("/dirpre", methods=["POST"])
def dirpre():
    info = {"result":''}
    # 通过request获得图片
    print("****************************")
    dir = request.form["dir"]
    if not os.path.exists(dir):
        info["result"] = 'The input dir is not exits, please check again !'
        return jsonify(info)
    dir = os.path.join(dir)
    print(os.listdir(dir))
    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    log_str = f"user ip ({request.remote_addr})({request.environ['REMOTE_ADDR']}) access the predict!!! "
    logger.info(log_str)
    # 通过request获得图片
    image = request.files["file"]
    # 通过read 将图片转为二进制文件
    img_bytes = image.read()
    # 调用predict,获得分割/增强后的结果
    info = get_predict(img_bytes)
    # return render_template('main.html',result=info)
    return jsonify(info)

@app.route("/trans_img", methods=["POST"])
def trans_img():
    log_str = f"user ip ({request.remote_addr})({request.environ['REMOTE_ADDR']}) input a image!!! "
    logger.info(log_str)
    # 通过request获得图片
    image = request.files["file"]
    # 通过read 将图片转为二进制文件
    img_bytes = image.read()
    # 调用predict,获得分割/增强后的结果
    image = Image.open(io.BytesIO(img_bytes))
    buffer=io.BytesIO()
    image.save(buffer,"PNG") # 将Image对象转为二进制存入buffer。因BytesIO()是在内存中操作，所以实际是存入内存
    buf_bytes=buffer.getvalue() # 从内存中取出bytes类型的图片
    base64_data='data:image/png;base64,'+str(base64.b64encode(buf_bytes),'utf-8') # 1.txt中的内容和base64_data2一样，也可以用base64_data3的写法
    info = {"result":base64_data}
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    # print("************************")
    # print(request.user_agent)
    # print("************************")
    log_str = f"user ip ({request.remote_addr})({request.environ['REMOTE_ADDR']}) access the server!!! "
    logger.info(log_str)
    return render_template("main.html")

@app.route("/snake")
def snake():
    log_str = f"user ip ({request.remote_addr})-({request.environ['REMOTE_ADDR']}) access the snake game!!! "
    logger.info(log_str)
    return render_template("snake.html")


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)  


