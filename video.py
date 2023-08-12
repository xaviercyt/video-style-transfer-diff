import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
# from torchvision import transforms
# from torchvision.utils import save_image
# from function import calc_mean_std, normal, coral
# import net as net
import numpy as np
import cv2
import yaml


#%%
def load_video(content_path,style_path, outfile):
    video = cv2.VideoCapture(content_path)

    rate = video.get(5)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获得帧宽和帧高
    fps = int(rate)

    video_name = outfile + '/{:s}_stylized_{:s}{:s}'.format(
        splitext(basename(content_path))[0], splitext(basename(style_path))[0], '.avi')

    videoWriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps,(512,512))
                                  #(int(width), int(height)))

    return video,videoWriter

def save_frame(output, videoWriter):
  # output = [int(item) for item in output]

  output = torch.from_numpy(np.array(output))
  output = output * 255 + 0.5
  output = (torch.clamp(output, 0, 255).permute(1, 2, 0)).numpy().astype(np.uint8)
  output = np.transpose(output, (2, 0, 1))
  output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
  videoWriter.write(output)


#%%

def main(content_path,style_path,outfile):
    video,videoWriter=load_video(content_path,style_path,outfile)
    j = 11
    path_to_check = './img2img-samples'
    target_file = 'tmp-_-00' + str(j) + '.png'
    if video.isOpened():
        while(os.path.exists(os.path.join(path_to_check, target_file))):
            #save_frame(target_file,videoWriter)
            img=cv2.imread(os.path.join(path_to_check, target_file))
            #save_frame(img, videoWriter)
            videoWriter.write(img)
            j = j+1
            target_file = 'tmp-_-00' + str(j) + '.png'
    else:
        print('it doesnt opened')

content_path = './room.mp4'
style_path = ''
outfile = './outputs'
main(content_path,style_path,outfile)

