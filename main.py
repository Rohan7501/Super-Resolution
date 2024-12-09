import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
from math import log10
from time import time
from torchvision.models import vgg19

torch.cuda.empty_cache()

EDSR_B = 32
EDSR_F = 256
EDSR_scale = 2
EDSR_PS = 3 * (EDSR_scale * EDSR_scale)
EDSR_scaling_factor = 0.1
Epochs = 100

threads = 4
batchSize = 5
testBatchSize = 10

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # Squeeze: Global average pooling
        y = x.mean(dim=(2, 3))

        # Excitation: Fully connected layers
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Reshape and scale
        y = y.view(batch, channels, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Compute average and max pooling along the channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along the channel axis and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(out))
        return x * out

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(EDSR_F, EDSR_F, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(EDSR_F, EDSR_F, kernel_size=3, padding=1)
        self.se_block = SEBlock(EDSR_F)  # Channel attention
        self.spatial_attention = SpatialAttention()  # Spatial attention

    def forward(self, x):
        residual = x

        # Main convolutional path
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply SE block (Channel attention)
        out = self.se_block(out)

        # Apply Spatial attention
        out = self.spatial_attention(out)

        # Add the residual connection
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
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features  # Load VGG19 features
        self.feature_extractor = nn.Sequential(*list(vgg)[:36])  # Use up to layer conv5_4
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG parameters

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return F.mse_loss(pred_features, target_features)

def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    psnr = 10 * log10(1 / mse.item())
    return psnr

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    perceptual_criterion = PerceptualLoss().to(device)


    def train(epoch):
        epoch_loss = 0
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            input, target = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            output = model(input)

            # Compute the losses
            pixel_loss = criterion(output, target)
            perceptual_loss = perceptual_criterion(output, target)
            combined_loss = pixel_loss + 0.01 * perceptual_loss 
            epoch_loss += combined_loss.item()
            combined_loss.backward()
            optimizer.step()

            print(f"===> Epoch[{epoch}]({iteration}/{len(training_data_loader)}): "
              f"Pixel Loss: {pixel_loss.item():.4f}, "
              f"Perceptual Loss: {perceptual_loss.item():.4f}, "
              f"Combined Loss: {combined_loss.item():.4f}")

        print(f"===> Epoch {epoch} Complete: Avg. Loss: {epoch_loss / len(training_data_loader):.4f}")

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
        model_out_path = "models_iter/attn_percep_256/model_epoch_{}.pth".format(epoch)
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
