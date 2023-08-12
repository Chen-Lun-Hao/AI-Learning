import gradio as gr
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from cyclegan import Cycle_Gan_G


#图片转换
transforms = transform.Compose([
        transform.ToTensor(),
        transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def generate_image(input_image):

    input_tensor = transforms(input_image.copy())
    input_tensor = input_tensor[None].to('cuda')

    # 实例化网络
    Gb = Cycle_Gan_G().to('cuda')
    # 加载预训练权重
    ckpt = torch.load('cycle_monent2photo.pth')
    Gb.load_state_dict(ckpt['Gb_model'], strict=False)

    Gb.eval()
    out = Gb(input_tensor)[0]
    out = out.permute(1, 2, 0)

    out = (0.5 * (out + 1)).cpu().detach().numpy()

    return out


inputs = gr.inputs.Image()
outputs = gr.outputs.Image(type='numpy')
interface = gr.Interface(fn=generate_image, inputs=inputs, outputs=outputs)

interface.launch(share=True)
