import os
import torch
from PIL import Image
from model import Unet_plus_plus
from torchvision import transforms
import cv2
import numpy as np

model_path = '../files/detect.pth'
device = 'cuda:0'
model = Unet_plus_plus().to(device)   # Unet++
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
model.eval()

root = "../../test/images"
root = "../test/images"
img_name_list = os.listdir(root)
for img_name in img_name_list:
    img_path = os.path.join(root, img_name)

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512,512)),
        ])
    img = trans(image)

    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        seg_img = model(img)
    seg_img = seg_img.squeeze(0)
    seg_img = transforms.Resize((584,565))(seg_img)

    seg_img = seg_img.cpu().numpy()[0]
    seg_img = seg_img > 0.1

    pil_img = Image.fromarray((seg_img * 255).astype(np.uint8))


    pil_img = pil_img.convert("1")
    if not os.path.exists("./pre"):
        os.makedirs("pre")
    pil_img.save(os.path.join("./pre", img_name))