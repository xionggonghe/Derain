from model.networks import DerainNet, Discriminator
from model.CR import ContrastLoss
import numpy as np
import time
import datetime
import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import torch.optim as optim
import utils
from utils.PSNR import torchPSNR as PSNR

from utils.data_RGB import get_training_data, get_validation_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params

def Count_DisAdversarialLoss(discriminator, gen_imgs, Rain_img, norain_img):
    # Loss function
    criterion = {}
    criterion["CosSimilarity"] = nn.CosineSimilarity(dim=1).to(device)
    criterion["Classfier"] = nn.CrossEntropyLoss(weight=None, reduction='mean').to(device)
    criterion["MatriSimilarity"] = nn.L1Loss().to(device)
    Weight = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
    gen_Vector, gen_classfier, gen_fea = discriminator(gen_imgs)
    Rain_Vector, Rain_classfier, Rain_fea = discriminator(Rain_img)
    norain_Vector, norain_classfier, norain_fea = discriminator(norain_img)

    Loss_VectorPN = criterion["CosSimilarity"](Rain_Vector, norain_Vector).mean(0)  # CosineSimilarity
    # Loss_VectorN = criterion["CosSimilarity"](gen_Vector, norain_Vector).mean(0)  # CosineSimilarity
    Loss_Vector = 1 / Loss_VectorPN

    labels_Norain = torch.zeros([gen_classfier.shape[0]], dtype=torch.int64).to(device)
    labels_Rain = torch.ones([gen_classfier.shape[0]], dtype=torch.int64).to(device)
    Loss_Classfier = criterion["Classfier"](gen_classfier, labels_Rain) + \
                     criterion["Classfier"](Rain_classfier, labels_Rain) + \
                     criterion["Classfier"](norain_classfier, labels_Norain)

    Loss_feas = 0.0
    for i in range(0, len(gen_fea)-1):
        Loss_feas = Loss_feas + Weight[i] * criterion["MatriSimilarity"](gen_fea[i], norain_fea[i])  # L1Loss

    Loss = Loss_Classfier #+ Loss_pic + Loss_feas + Loss_pic

    return Loss


if __name__ == '__main__':
    model = DerainNet().to(device)
    Dis = Discriminator().to(device)
    cnn_paras_count(model)
    cnn_paras_count(Dis)
    # print(model)

    '''##############################################加载数据#########################################################################'''
    NUM_EPOCHS = 500    # 训练周期
    BATCH_SIZE = 32     # 每批次训练数量
    patch_size = 128    # 训练图像裁剪大小
    train_dir = "./dataset/rain_data_train_Light/"     # 训练数据集目录
    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir, {'patch_size':patch_size})
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_dir = "./dataset/rain_data_test_Light/"
    val_dataset = get_validation_data(val_dir, {'patch_size':patch_size})
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    '''##############################################优化器#########################################################################'''
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
    optimizer_D = optim.Adam(Dis.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
    # loss
    criterion = []
    criterion.append(nn.L1Loss().to(device))
    criterion.append(ContrastLoss())

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=0.0001, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer_D = optim.Adam(params=filter(lambda x: x.requires_grad, Dis.parameters()), lr=0.0001, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()

    best_psnr = 0
    best_epoch = 0
    model_dir = "./result"
    w_loss_l1 = 1
    w_loss_vgg7 = 0.5

    # Tensorboard
    writer = SummaryWriter('runs/experiment_1')

    images = torch.rand([1, 3, 128, 128]).to(device)
    net = Discriminator().to(device)
    writer.add_graph(net, images)
    writer.close()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # zero_grad
            for param in model.parameters():
                param.grad = None

            target = data[0].to(device)
            input_ = data[1].to(device)

            restored = model(input_)

            loss_rec = criterion[0](restored, target)
            loss_vgg7 = criterion[1](Dis, restored, target, input_)
            loss = w_loss_l1 * loss_rec + w_loss_vgg7 * loss_vgg7
            loss.backward(retain_graph=True)
            psnr = PSNR(target, restored.detach()).mean().item()

            # print("psnr: ", psnr)

            # optimizer.step()
            # optimizer.zero_grad()
            epoch_loss += loss.item()
            # -----------------------------------------------------------------------------------------
            #  Train Discriminator
            # -----------------------------------------------------------------------------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            gen_img = restored.clone()
            d_loss = Count_DisAdversarialLoss(Dis, gen_img, input_, target)
            d_loss.backward()
            optimizer.step()
            optimizer_D.step()
            with torch.no_grad():
                writer.add_scalar('loss Gen',
                                  loss,
                                  epoch * len(train_loader) + i)
                writer.add_scalar('loss Dis',
                                  d_loss,
                                  epoch * len(train_loader) + i)

                writer.add_scalar('PSNR: ',
                                  psnr,
                                  epoch * len(train_loader) + i)

        #### Evaluation ####
        if epoch % 1 == 0:
            model.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)

                with torch.no_grad():
                    restored = model(input_)
                restored = restored[0]

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(PSNR(res, tar))
            print("PSNR: ", PSNR(res, tar))
            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_best.pth"))

        scheduler.step()
        scheduler_D.step()

        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))







