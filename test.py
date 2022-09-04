from model.networks import DerainNet
import torch
from torch import nn
from utils.PSNR import torchPSNR as PSNR
from utils.SSIM import SSIM
from utils.data_RGB import get_test_data
from utils.dataset_RGB import DataLoader_Test
from tqdm import tqdm
from PIL import Image
from PIL import ImageOps
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    # 'state_dict': model.state_dict()
    save = torch.load("./result/model_best.pth", map_location=torch.device('cpu'))
    model = DerainNet().to(device)
    model.load_state_dict(save['state_dict'])
    test_dir = "./dataset/rain_data_test_Light/"
    inp_fileList, tar_fileList = DataLoader_Test(test_dir)
    SSIM_loss = SSIM()
    PSNR_TEST = 0.0
    SSIM_TEST = 0.0
    show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

    model.eval()
    for i in range(5): #len(inp_fileList)
        index = i+1
        inp_img = Image.open(inp_fileList[index])
        inp_img = ImageOps.exif_transpose(inp_img)  # 恢复正常角度的图像
        tar_img = Image.open(tar_fileList[index])
        tar_img = ImageOps.exif_transpose(tar_img)  #

        input_ = TF.to_tensor(inp_img)
        # print(input_.shape)
        input_ = TF.resize(input_, [(input_.shape[1]-1), (input_.shape[2]-1)])
        target = TF.to_tensor(tar_img)
        target = TF.resize(target, [(target.shape[1] - 1), (target.shape[2] - 1)])


        input_ = torch.unsqueeze(input_, dim=0)
        restored = model(input_)
        # restored = torch.squeeze(restored)
        # target = torch.squeeze(target)


        print("psnr: ", PSNR(restored, target))
        print("ssim: ", SSIM_loss(restored, target))
        PSNR_TEST += PSNR(restored, target)
        SSIM_TEST += SSIM_loss(restored, target)

        show(torch.squeeze(input_, dim=0)).show()
        show(torch.squeeze(restored, dim=0)).show()
        show(torch.squeeze(target, dim=0)).show()
        print("pic1")

    PSNR_TEST /= (i+1)
    SSIM_TEST /= (i+1)
    print("all_psnr: ", PSNR_TEST)
    print("all_ssim: ", SSIM_TEST)
















