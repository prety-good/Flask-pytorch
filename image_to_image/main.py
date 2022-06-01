import io
import json
import os
from re import A
import torch
import torchvision.transforms as transforms
from efficientnet import efficientnetv2_l as create_model
from detect import onehot_to_mask
import albumentations as A
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from unetplus import Unet_plus_plus
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



# 模型准备相关工作 预先读入内存********************************************************************
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 模型路径
model1_weight_path = "./modelfile/unetplus.pth"
model2_weight_path = "./modelfile/efficientnet.pth"
# 加载模型
model1 = Unet_plus_plus(in_ch=1, out_ch=4).to(device)
model1.load_state_dict(torch.load(model1_weight_path, map_location=device))
model1.eval()
model2 = create_model(num_classes= 3).to(device)
model2.load_state_dict(torch.load(model2_weight_path, map_location=device))
model2.eval()
transform1 = A.Compose([
        A.Resize(height=224, width=224, p=1.0),
        # A.CenterCrop(height=224, width=224, p=1.0),
        A.CLAHE(p=1.0),
])
transform2 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
dict = {
    "0": "DCM",
    "1": "HCM",
    "2": "NOR"
}
#******************************************************************************************

@torch.no_grad()
def get_predict(image_bytes):
    '''
        获得单张图片的推理结果
    '''
    return_info = {}
    try:
        # Image读取二进制图片
        img = Image.open(io.BytesIO(image_bytes))
        # # cv2读取
        # image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        # image = Image.fromarray(image)

        # 土图片保存到 temp文件中，作为临时文件
        curr_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H-%M-%S')
        img.save(f"./temp/{curr_time}.jpg")
        log_str = f"user ip ({request.remote_addr})({request.environ['REMOTE_ADDR']}) 保存了一张临时图片 !!! "
        logger.info(log_str)

        if img.mode != "L":
            img=img.convert('L')
        if img.mode != "L":
            raise ValueError(f"input file does not L image... but {img.mode} image!!!")
        img1 = np.array(img)
        img1 = transform1(image=img1)["image"]
        img1 = transforms.ToTensor()(img1)
        img1 = img1.unsqueeze(0).to(device)
        with torch.no_grad():
            segement_image = model1(img1)
            segement_image = torch.sigmoid(segement_image)
            segement_image = onehot_to_mask(np.transpose(segement_image[0].detach().cpu().numpy(), (1, 2, 0)))

            final_seg_image = segement_image.copy()
            
            segement_image = segement_image[:,:,0]
            segement_image = Image.fromarray(segement_image).convert('L')
            size = list(img.size)
            size.reverse()
            segement_image = transforms.Resize(size)(segement_image)
            # print("segement_image.shape",segement_image.size)
            mask = np.array(segement_image)
            # print("mask.shape",mask.shape)
            mask[mask > 0] = 1
            mask_image = np.array(img)
            mask_image = mask_image * mask
            # print("mask_image.shape",mask_image.shape)
            mask_image = Image.fromarray(mask_image).convert('L')
            # print(mask_image.size)

            img = Image.merge('RGB', (img, segement_image, mask_image))
            img = transforms.ToTensor()(img).unsqueeze(0).to(device)
            predict = model2(img)

            pred_classes = torch.max(predict, dim=1)[1][0].item()
            print(f"预测结果为：{pred_classes}")

            outputs = torch.softmax(predict.squeeze(), dim=-1)
            prediction = outputs.detach().cpu().numpy()
            template = "类别:{}   置信度:{:.3f}"
            index_pre = [(dict[str(index)], float(p)) for index, p in enumerate(prediction)]
            # sort probability
            index_pre.sort(key=lambda x: x[1], reverse=True)
            text = [template.format(k, v) for k, v in index_pre]
            return_info["classification"] = text

        final_seg_image = Image.fromarray(final_seg_image[:,:,0].astype(np.uint8))
        buffer=io.BytesIO()
        final_seg_image.save(buffer,"PNG") # 将Image对象转为二进制存入buffer。因BytesIO()是在内存中操作，所以实际是存入内存
        buf_bytes=buffer.getvalue() # 从内存中取出bytes类型的图片
        base64_data='data:image/png;base64,'+str(base64.b64encode(buf_bytes),'utf-8') # 1.txt中的内容和base64_data2一样，也可以用base64_data3的写法
        # base64_data3=(b'data:image/png;base64,'+base64_data).decode('utf-8')
        return_info["result"] = base64_data

    except Exception as e:
        log_str = f"user ip ({request.remote_addr})({request.environ['REMOTE_ADDR']}) produce a erro:{e}!!! "
        logger.error(log_str)
        return None
    return return_info


@app.route("/dirpre", methods=["POST"])
def dirpre():
    '''
        获得目标路径的批量处理结果
    '''
    log_str = f"user ip ({request.remote_addr})({request.environ['REMOTE_ADDR']}) 使用了文件夹批量推理功能 !!!"
    logger.info(log_str)
    info = {"result":''}
    # 通过request获得图片
    test_dir = request.form["dir"]
    if not os.path.exists(test_dir):
        info["result"] = 'The input dir is not exits, Please check again !'
        return jsonify(info)
    label_dir = os.path.join(test_dir,'..','Label')
    if os.path.exists(label_dir) is False:
        os.makedirs(label_dir)

    try:
        # 预测的主体代码
        patient_list = os.listdir(test_dir)
        for patient in patient_list:
            path = os.path.join(test_dir, patient)
            img_name_list = os.listdir(path)
            
            for img_name in img_name_list:
                
                img_num = img_name.split(".")[0]
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path)

                imag1 = np.array(img)
                # img1 = transforms.Resize([224, 256])(img)
                img1 = transform1(image=imag1)["image"]
                img1 = transforms.ToTensor()(img1)
                img1 = img1.unsqueeze(0).to(device)

                with torch.no_grad():
                    
                    segement_image = model1(img1)
                    segement_image = torch.sigmoid(segement_image)
                    segement_image = onehot_to_mask(np.transpose(segement_image[0].detach().cpu().numpy(), (1, 2, 0)))
                    segement_image = segement_image[:,:,0]
                    segement_image = Image.fromarray(segement_image).convert('L')
                    # print(img.size)
                    size = list(img.size)
                    size.reverse()
                    # print(size)
                    segement_image = transforms.Resize(size)(segement_image)
                    # print("segement_image.shape",segement_image.size)

                    mask = np.array(segement_image)
                    # print("mask.shape",mask.shape)
                    mask[mask > 0] = 1
                    mask_image = np.array(img)
                    mask_image = mask_image * mask
                    # print("mask_image.shape",mask_image.shape)

                    mask_image = Image.fromarray(mask_image).convert('L')
                    # print(mask_image.size)


                    img = Image.merge('RGB', (img, segement_image, mask_image))
                    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
                    predict = model2(img)
                    pred_classes = torch.max(predict, dim=1)[1][0].item()
                    img_name = img_num + '_' + dict[str(pred_classes)] + ".png"
                    final_path = label_dir +'/' +patient +'/' 
                    if os.path.exists(final_path) is False:
                        os.makedirs(final_path)
                    segement_image.save(final_path+ img_name)
    except Exception as e:
        print(f'erro! {e}')
        info["result"] = e
        log_str = f"user ip ({request.remote_addr})({request.environ['REMOTE_ADDR']})触发了一个BUG：{e}!!! "
        logger.error(log_str)
        return jsonify(info)
    info["result"] = "Predict over ！！! "
    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    '''
        通过调用get_predict()获得单张图片的推理结果
    '''
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
    '''
        将图片转化为PNG格式以在前端显示
    '''
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
    '''
        贪吃蛇
    '''
    log_str = f"user ip ({request.remote_addr})-({request.environ['REMOTE_ADDR']}) access the snake game!!! "
    logger.info(log_str)
    return render_template("snake.html")


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)  


