import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
#from XGHNet import *
from PIL import Image
from PIL import ImageOps
from model.networks import DerainNet
from torch.autograd import Variable
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage
import torchvision.transforms as transform
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Datacrop(Num, ps=128):
    inp_path = "C:/Users/Xiong/Documents/paper/Derain/dataset/rain_data_train_Light/rain/" + "rain-{}.png".format(Num)
    tar_path = "C:/Users/Xiong/Documents/paper/Derain/dataset/rain_data_train_Light/norain/" + "norain-{}.png".format(Num)
    inp_img = Image.open(inp_path)
    inp_img = ImageOps.exif_transpose(inp_img)  # 恢复正常角度的图像
    tar_img = Image.open(tar_path)
    tar_img = ImageOps.exif_transpose(tar_img)  # 恢复正常角度的图像

    w, h = tar_img.size
    padw = ps - w if w < ps else 0
    padh = ps - h if h < ps else 0

    # Reflect Pad in case image is smaller than patch_size
    if padw != 0 or padh != 0:
        inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
        tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

    inp_img = TF.to_tensor(inp_img)
    tar_img = TF.to_tensor(tar_img)

    hh, ww = tar_img.shape[1], tar_img.shape[2]

    # 生成裁剪与随机数据增强
    rr = random.randint(0, hh - ps)
    cc = random.randint(0, ww - ps)
    aug = random.randint(0, 4)
    # Crop patch
    inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
    tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

    return inp_img, tar_img


if __name__ == '__main__':
    """**************************************************************"""
    save_infor = torch.load('../result/model_epoch_500.pth',  map_location=torch.device('cpu'))
    model = DerainNet().to(device)
    model.load_state_dict(save_infor['state_dict'])
    print(model)

    """**************************************************************"""

    Num = 5
    PatchSize = 256

    img_path = "C:/Users/Xiong/Documents/paper/Derain/dataset/rain_data_train_Light/" + "rain/rain-{}.png".format(Num)
    img_org = cv2.imread(img_path)

    # img = cv2.resize(img_org, (224, 224))
    # img = np.swapaxes(img, 0, 2)
    # img = np.swapaxes(img, 1, 2)
    # img_tensor = torch.from_numpy(img).float()
    # img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False)
    """**************************************************************"""
    img_tensor = Datacrop(Num, PatchSize)
    inp_img = img_tensor[0]
    inp_img = torch.unsqueeze(inp_img, dim=0)
    feature = model.feature_extract(inp_img)
    show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
    show(torch.squeeze(inp_img, dim=0)).show()

    """**************************************************************"""
    if not os.path.exists('./featureMap'):
        os.makedirs('./featureMap')

    """**************************************************************"""
    heat = feature[0].data.numpy()	     # 将tensor格式的feature map转为numpy格式
    heat = np.squeeze(heat, 0)	         # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
    heatmap = np.maximum(heat, 0)        # heatmap与0比较
    heatmap = np.mean(heatmap, axis=0)   # 多通道时，取均值
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.show()
    img = img_org
    heatmap = cv2.resize(heatmap, (PatchSize, PatchSize))  # 特征图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)              # 将特征图转换为uint8格式
    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)  # 将特征图转为伪彩色图
    # heat_img = cv2.addWeighted(img_org, 1, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
    #heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
    cv2.imwrite('./featureMap/'+str(Num)+'_heatmap_0.jpg', heat_img)               # 将图像保存

    """**************************************************************"""
    heat = feature[1].data.numpy()  # 将tensor格式的feature map转为numpy格式
    heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
    heatmap = np.maximum(heat, 0)  # heatmap与0比较
    heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()
    img = img_org
    heatmap = cv2.resize(heatmap, (PatchSize, PatchSize))  # 特征图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将特征图转换为uint8格式
    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)  # 将特征图转为伪彩色图
    # heat_img = cv2.addWeighted(img_org, 1, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
    # heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
    cv2.imwrite('./featureMap/' + str(Num) + '_heatmap_1.jpg', heat_img)  # 将图像保存

    """**************************************************************"""
    heat = feature[2].data.numpy()	     # 将tensor格式的feature map转为numpy格式
    heat = np.squeeze(heat, 0)	         # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
    heatmap = np.maximum(heat, 0)        # heatmap与0比较
    heatmap = np.mean(heatmap, axis=0)   # 多通道时，取均值
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.show()
    img = img_org
    heatmap = cv2.resize(heatmap, (PatchSize, PatchSize))  # 特征图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)              # 将特征图转换为uint8格式
    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)  # 将特征图转为伪彩色图
    # heat_img = cv2.addWeighted(img_org, 1, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
    #heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
    cv2.imwrite('./featureMap/'+str(Num)+'_heatmap_2.jpg', heat_img)               # 将图像保存

    """**************************************************************"""
    heat = feature[3].data.numpy()	     # 将tensor格式的feature map转为numpy格式
    heat = np.squeeze(heat, 0)	         # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
    heatmap = np.maximum(heat, 0)        # heatmap与0比较
    heatmap = np.mean(heatmap, axis=0)   # 多通道时，取均值
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.show()
    img = img_org
    heatmap = cv2.resize(heatmap, (PatchSize, PatchSize))  # 特征图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)              # 将特征图转换为uint8格式
    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)  # 将特征图转为伪彩色图
    # heat_img = cv2.addWeighted(img_org, 1, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
    #heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
    cv2.imwrite('./featureMap/'+str(Num)+'_heatmap_3.jpg', heat_img)               # 将图像保存

    """**************************************************************"""
    heat = feature[4].data.numpy()  # 将tensor格式的feature map转为numpy格式
    heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
    heatmap = np.maximum(heat, 0)  # heatmap与0比较
    heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()
    img = img_org
    heatmap = cv2.resize(heatmap, (PatchSize, PatchSize))  # 特征图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将特征图转换为uint8格式
    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE)  # 将特征图转为伪彩色图
    # heat_img = cv2.addWeighted(img_org, 1, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
    # heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
    cv2.imwrite('./featureMap/' + str(Num) + '_heatmap_4.jpg', heat_img)  # 将图像保存

    # """**************************************************************"""
    # heat = feature[5].data.numpy()	     # 将tensor格式的feature map转为numpy格式
    # heat = np.squeeze(heat, 0)	         # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
    # heatmap = np.maximum(heat, 0)        # heatmap与0比较
    # heatmap = np.mean(heatmap, axis=0)   # 多通道时，取均值
    # heatmap /= np.max(heatmap)
    # #plt.matshow(heatmap)
    # #plt.show()
    # img = img_org
    # heatmap = cv2.resize(heatmap, (PatchSize, PatchSize))  # 特征图的大小调整为与原始图像相同
    # heatmap = np.uint8(255 * heatmap)              # 将特征图转换为uint8格式
    # heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
    # # heat_img = cv2.addWeighted(img_org, 1, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
    # #heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
    # cv2.imwrite('./featureMap/'+str(Num)+'_heatmap_5.jpg', heat_img)               # 将图像保存