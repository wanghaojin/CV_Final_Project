import cv2
import os
import glob
import torch
from framework import Stylization

## -------------------
##  Parameters
for num in range(1,5):
    style_img = '/home/sem/Videos/ReReVST-Code/test/styles/kandinsky.jpg'  
    content_dir = f"/home/sem/Videos/ReReVST-Code/MAE/stitch_75fox" 
    checkpoint_path = "/home/sem/Videos/ReReVST-Code/test/Model/style_net-TIP-final.pth"
    cuda = torch.cuda.is_available()
    result_path = f'/home/sem/Videos/ReReVST-Code/MAE/stitch_75fox_transfer/'

    ## -------------------
    ##  Tools


    def read_img(img_path, scale=1.0):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return img

    class ReshapeTool():
        def __init__(self):
            self.record_H = 0
            self.record_W = 0

        def process(self, img):
            H, W, C = img.shape

            if self.record_H == 0 and self.record_W == 0:
                new_H = H + 128
                if new_H % 64 != 0:
                    new_H += 64 - new_H % 64

                new_W = W + 128
                if new_W % 64 != 0:
                    new_W += 64 - new_W % 64

                self.record_H = new_H
                self.record_W = new_W

            new_img = cv2.copyMakeBorder(img, 64, self.record_H-64-H,
                                            64, self.record_W-64-W, cv2.BORDER_REFLECT)
            return new_img

    def img_transform(content_img_path, style_img_path, checkpoint_path, cuda, result_path):
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        
        style = cv2.imread(style_img_path)
        framework = Stylization(checkpoint_path, cuda, False)
        framework.prepare_style(style)

        reshape = ReshapeTool()

        content_image = read_img(content_img_path)
        
        new_content_image = reshape.process(content_image)# 补齐处理
        styled_image = framework.transfer(new_content_image)
        
        H, W, C = content_image.shape
        styled_image = styled_image[64:64+H, 64:64+W, :]

        result_img_path = os.path.join(result_path, os.path.basename(content_img_path))
        cv2.imwrite(result_img_path, styled_image)

    ## -------------------
    ##  Inference

    content_images = glob.glob(os.path.join(content_dir, "*.jpg")) 

    for content_img_path in content_images:
        img_transform(content_img_path, style_img, checkpoint_path, cuda, result_path)

    print("Styling of all images are complete.")