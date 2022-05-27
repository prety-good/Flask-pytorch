import io
import json
from traceback import print_tb
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model import Unet_plus_plus
import numpy as np
import cv2
import base64

from datetime import timedelta
app = Flask(__name__)
CORS(app)  # 解决跨域问题
app.send_file_max_age_default = timedelta(seconds=1)
# 加载模型
model_path = './detect.pth'
device = 'cuda:0'
model = Unet_plus_plus().to(device)   # Unet++
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
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
        
        if image.mode != "RGB":
            raise ValueError("input file does not RGB image...")
        image = trans(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image).squeeze(0)
        prediction = outputs.cpu().numpy()[0]
        prediction = prediction > 0.1
        # cv2.imshow("1",(prediction * 255).astype(np.uint8))
        # cv2.waitKey()
        pil_img = Image.fromarray((prediction * 255).astype(np.uint8))
        pil_img = pil_img.convert("1")
        # pil_img.save("./static/pre.jpg")

        buffer=io.BytesIO()
        pil_img.save(buffer,"PNG") # 将Image对象转为二进制存入buffer。因BytesIO()是在内存中操作，所以实际是存入内存
        buf_bytes=buffer.getvalue() # 从内存中取出bytes类型的图片
        base64_data='data:image/png;base64,'+str(base64.b64encode(buf_bytes),'utf-8') # 1.txt中的内容和base64_data2一样，也可以用base64_data3的写法
        # base64_data3=(b'data:image/png;base64,'+base64_data).decode('utf-8')
        return_info = {"result":base64_data}

    except Exception as e:
        print(e)
        return None
    return return_info


@app.route("/predict", methods=["POST"])
def predict():
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
    return render_template("main.html", result = './test.jpg')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)  




