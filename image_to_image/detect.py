import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from efficientnet import efficientnetv2_l as create_model
import albumentations as A
from unetplus import Unet_plus_plus

model1_weight_path = "./modelfile/unetplus.pth"
model2_weight_path = "./modelfile/efficientnet.pth"

# 测试文件夹路径
test_dir = ""
label_dir = os.path.join(test_dir,'..','Label')
dict = {
    "0": "DCM",
    "1": "HCM",
    "2": "NOR"
}
if os.path.exists(label_dir) is False:
    os.makedirs(label_dir)


def onehot_to_mask(mask, palette = [[0], [85],[170], [255]]):
    """
    输入为numpy格式，(H, W, K)，K为背景的类别数
    输出为numpy格式的图片，(H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    img_size = 224
    transform1 = A.Compose([
        A.Resize(height=224, width=224, p=1.0),
        # A.CenterCrop(height=224, width=224, p=1.0),
        A.CLAHE(p=1.0),
    ])
    transform2 = transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor(),
                                     ])

    model1 = Unet_plus_plus(in_ch=1, out_ch=num_classes + 1).to(device)
    model1.load_state_dict(torch.load(model1_weight_path, map_location=device))
    model1.eval()
    model2 = create_model(num_classes=num_classes).to(device)
    model2.load_state_dict(torch.load(model2_weight_path, map_location=device))
    model2.eval()

    patient_list = os.listdir(test_dir)
    for patient in patient_list:
        path = os.path.join(test_dir, patient)
        img_name_list = os.listdir(path)
        
        ans = []
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
                print(img.size)
                size = list(img.size)
                size.reverse()
                print(size)
                segement_image = transforms.Resize(size)(segement_image)
                print("segement_image.shape",segement_image.size)

                mask = np.array(segement_image)
                print("mask.shape",mask.shape)
                mask[mask > 0] = 1
                mask_image = np.array(img)
                mask_image = mask_image * mask
                print("mask_image.shape",mask_image.shape)

                mask_image = Image.fromarray(mask_image).convert('L')
                print(mask_image.size)


                img = Image.merge('RGB', (img, segement_image, mask_image))
                img = transforms.ToTensor()(img).unsqueeze(0).to(device)
                predict = model2(img)
                pred_classes = torch.max(predict, dim=1)[1][0].item()


                img_name = img_num + '_' + dict[str(pred_classes)] + ".png"
                final_path = label_dir +'/' +patient +'/' 
                if os.path.exists(final_path) is False:
                    os.makedirs(final_path)
                segement_image.save(final_path+ img_name)
