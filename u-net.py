import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

## Dataset part
class BratsDataset_seg(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.data_list = list(sorted(os.listdir(os.path.join(root))))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ###### fill the codes below #####

        # 1. Define paths for an image and a label
        image_path = os.path.join(self.root, self.data_list[idx], 'img.npy')
        label_path = os.path.join(self.root, self.data_list[idx], 'label.npy')

        # 2. Load arrays for the image and label
        img = np.load(image_path)
        label = np.load(label_path)

        # 3. Process image array (e.g., normalization, numpy array to tensor)
        # Normalize the image
        img = img.astype(np.float32)
        img = (img - img.mean()) / img.std()

        # 4. Process label array [0,1,2,4] => [0,1,2,3]
        # Map label 4 to 3
        label[label == 4] = 3
        label = torch.from_numpy(label).long()

        #################################
        output = {'img': img, 'label': label}
        return output

## Add DilatedConv 
class DilatedConv(nn.Module):
    """Dilated Convolution"""

    def __init__(self, in_channels, out_channels, dilation_rate=2):
        super().__init__()
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.dilated_conv(x)

## Model part
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_dilation=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_dilation:
            self.double_conv = nn.Sequential(
                DilatedConv(in_channels, mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_dilation=False):
        super().__init__()
        if use_dilation:
            self.maxpool_conv = nn.Sequential(
                DoubleConv(in_channels, out_channels, use_dilation=use_dilation)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512, use_dilation=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_dilation=True)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        ###### fill the codes below #####

        # process of U-Net using predefined modules.
        # Encoder part
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder part
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output layer
        logits = self.outc(x)

        #################################
        return logits


def train_one_epoch(model, optimizer, criterion, train_data_loader, valid_data_loader, device, epoch, lr_scheduler, print_freq=10, min_valid_loss=100):
    for train_iter, pack in enumerate(train_data_loader):
        train_loss = 0
        valid_loss = 0
        img = pack['img'].to(device)
        label = pack['label'].to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if (train_iter+1) % print_freq == 0:
            with torch.no_grad():
                model.eval()
                for valid_iter, pack in enumerate(valid_data_loader):
                    img = pack['img'].to(device)
                    label = pack['label'].to(device)
                    pred = model(img)

                    loss = criterion(pred, label)
                    valid_loss += loss.item()

                if min_valid_loss >= valid_loss/len(valid_data_loader):

                    # set file fname for model save 'best_model_{your name}.pth'
                    torch.save(model.state_dict(), 'best_model_v1.pth')

                    min_valid_loss = valid_loss/len(valid_data_loader)
                    print('{}th epoch {}/{} iter: train loss={}, valid loss={}, lr={}'
                          .format(epoch+1, train_iter+1, len(train_data_loader), train_loss/print_freq, valid_loss/len(valid_data_loader), lr_scheduler.get_last_lr()),
                          " => model saved")
                else:
                    print('{}th epoch {}/{} iter: train loss={}, valid loss={}, lr={}'
                          .format(epoch+1, train_iter+1, len(train_data_loader), train_loss/print_freq, valid_loss/len(valid_data_loader), lr_scheduler.get_last_lr()))
        lr_scheduler.step()
    return min_valid_loss

def dice_similarity_coefficient(input, target):
    smooth = 1.0  # To avoid division by zero
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def evaluate(model, test_data_loader, device, classes):
    ###### fill the codes below #####
    model.eval()
    total_dsc = 0.0
    count = 0
    n_cases = 5
    good_cases = []
    bad_cases = []

    with torch.no_grad():
        for pack in test_data_loader:
            img = pack['img'].to(device)
            true_mask = pack['label'].to(device)

            pred_mask = model(img)
            pred_mask = torch.sigmoid(pred_mask)  # Apply sigmoid or softmax depending on output layer
            pred_mask = (pred_mask > 0.5).float()  # Thresholding to get binary mask

            for class_idx in range(classes):
                dsc = dice_similarity_coefficient(pred_mask[:, class_idx, :, :], true_mask == class_idx)
                if len(good_cases) < n_cases and dsc > 0.8:  # Threshold for good cases
                    good_cases.append((img.cpu().numpy(), pred_mask[:, class_idx, :, :], true_mask))
                elif len(bad_cases) < n_cases and dsc < 0.5:  # Threshold for bad cases
                    bad_cases.append((img.cpu().numpy(), pred_mask[:, class_idx, :, :], true_mask))
                total_dsc += dsc
                count += 1

    class_DSC = total_dsc / count
    #################################
    
    return class_DSC, good_cases, bad_cases

def visualize_cases(good_cases, bad_cases):
    # Visualization
    for i, (img, pred, true) in enumerate(good_cases + bad_cases):
        case_type = "Good Case" if i < len(good_cases) else "Bad Case"
        for img_id, img_one in enumerate(img[0]):
            plt.subplot(1, 6, img_id+1)
            plt.imshow(img_one, cmap='gray')
            plt.title(f'{case_type} - Input Image')
            plt.axis('off')
        pred_cpu = pred[0].cpu()
        plt.subplot(1, 6, 5)
        plt.imshow(pred_cpu, cmap='gray')
        plt.title(f'{case_type} - Predicted Mask')
        plt.axis('off')
        true_cpu = true[0].cpu()
        plt.subplot(1, 6, 6)
        plt.imshow(true_cpu, cmap='gray')
        plt.title(f'{case_type} - True Mask')
        plt.axis('off')
        
        plt.show()

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # gpu가 사용가능하다면 gpu를 사용하고, 아니라면 cpu를 사용함

    ## Hyper-parameters
    num_epochs = 30
    n_channels = 4 # number of modalities
    n_classes = 4

    model_channel = UNet(n_channels=n_channels, n_classes=n_classes)
    model_channel.to(device)

    optimizer = torch.optim.Adam(model_channel.parameters(), lr=0.0005)#weight_decay=0.0001
    criterion = nn.CrossEntropyLoss()

    # step_size 이후 learning rate에 gamma만큼을 곱해줌 ex) 111번 스텝 뒤에 lr에 gamma를 곱해줌
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=111,
                                                gamma=1) #0.9 ## learning rate decay

    ## data loader
    train_dataset = BratsDataset_seg('./seg_dataset/train')
    valid_dataset = BratsDataset_seg('./seg_dataset/valid')
    test_dataset = BratsDataset_seg('./seg_dataset/test')


    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, shuffle=True, num_workers=0)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=0)
    
    min_val_loss = 100
    best_model_path = 'best_model_v1.pth'  # Path to save the best model
    
    for epoch in range(num_epochs):
        min_val_loss = train_one_epoch(model_channel, optimizer, criterion, train_data_loader, \
            valid_data_loader, device, epoch, lr_scheduler, print_freq=30, min_valid_loss=min_val_loss)
    
    ###### fill the codes below #####
    # 1. Load the model which has the minimum validation loss
    model_channel.load_state_dict(torch.load(best_model_path))

    # 2. Evaluate the model using DSC
    dsc, good_cases, bad_cases = evaluate(model_channel, test_data_loader, device=device, classes=n_classes)
    print(f'Dice Similarity Coefficient: {dsc}')

    visualize_cases(good_cases, bad_cases)
    #################################

    
if __name__ == '__main__':
    main()
