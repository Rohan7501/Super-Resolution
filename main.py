import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
from math import log10
from time import time

EDSR_B = 32
EDSR_F = 256
EDSR_scale = 2
EDSR_PS = 3 * (EDSR_scale * EDSR_scale)
EDSR_scaling_factor = 0.1
Epochs = 100

threads = 4
batchSize = 8
testBatchSize = 10

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(EDSR_F, EDSR_F, 3, padding=1)
        self.conv2 = nn.Conv2d(EDSR_F, EDSR_F, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * EDSR_scaling_factor
        out = out + residual
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, EDSR_F, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(EDSR_B)])
        self.conv2 = nn.Conv2d(EDSR_F, EDSR_F, 3, padding=1)
        self.conv3 = nn.Conv2d(EDSR_F, EDSR_PS, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(EDSR_scale)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = x + residual
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    psnr = 10 * log10(1 / mse.item())
    return psnr

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('===> Loading datasets')
    train_set = get_training_set(EDSR_scale)
    test_set = get_test_set(EDSR_scale)
    training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=testBatchSize, shuffle=False)

    print('===> Building model')
    model = Net().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    def train(epoch):
        epoch_loss = 0
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            input, target = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

    def test():
        model.eval()
        avg_psnr = 0
        with torch.no_grad():
            for batch in testing_data_loader:
                input, target = batch[0].to(device), batch[1].to(device)
                prediction = model(input)
                psnr = calc_psnr(prediction, target)
                avg_psnr += psnr
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    def checkpoint(epoch):
        model_out_path = "models_iter_6/model_epoch_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    for epoch in range(1, Epochs + 1):
        start = time()
        train(epoch)
        end = time()
        print(f"Time for required epoch {epoch}: {(end - start)/60} minutes")
        test()
        checkpoint(epoch)
        scheduler.step()
