import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from matplotlib import pyplot as plt
from VMDataset import VideoSoundDataset
from torch.utils.data import Subset, ConcatDataset
import argparse
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import numpy as np
import torchaudio
import time

class SiameseConvNet(nn.Module):
    def __init__(self):
        super(SiameseConvNet, self).__init__()
        # Shared convolutional layers
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),  # reduces size to 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),  # reduces size to 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # reduces size to 4x4

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # reduces size to 2x2
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 2 * 2 * 2, 1024),  # *2 because of concatenation
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256,1),
            nn.Tanh()
        )

    def forward(self, x):
        
        # Split input image
        x1 = x[:,:,:,:32]
        x2 = x[:,:,:,32:]
        # Each branch processes an image
        out1 = self.convnet(x1)
        out2 = self.convnet(x2)

        # Flatten and concatenate features from both images
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out = torch.cat((out1, out2), dim=1)  # Concatenate along the feature dimension

        # Fully connected layers
        out = self.fc_layers(out)
        
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Use a pre-trained ResNet model
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the last layer (fully connected layer) of the ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.requires_grad_(False)
        # Define fully connected layers
        self.fc_layers = self._make_fc_layers()

    def forward(self, x):
        # Forward pass through ResNet
        x1 = x[:,:,:,:224]
        x2 = x[:,:,:,224:]

        x1 = self.resnet(x1)
        x2 = self.resnet(x2)

        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)
        
        # Concatenate x1 and x2
        x3 = torch.cat((x1, x2), dim=1)      
        # Forward pass through fully connected layers
        x3 = self.fc_layers(x3)
        
        return x3

    def _make_fc_layers(self):
        # Define sequential layers for fully connected layers
        return nn.Sequential(
            nn.Linear(4096, 2048),  # Adjust input dimension based on concatenated feature size
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # Shared convolutional layers
        self.conv1 = FPNBlock(3, 16)
        self.conv2 = FPNBlock(16, 32)
        self.conv3 = FPNBlock(32, 128)
        
        # Top-down pathway
        self.top_layer = nn.Conv2d(128, 128, kernel_size=1)
        
        # Lateral connections
        self.lateral_layer1 = nn.Conv2d(32, 128, kernel_size=1)
        self.lateral_layer2 = nn.Conv2d(16, 128, kernel_size=1)

        # Smooth layers
        self.smooth_layer1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.smooth_layer2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)


        # Fully connected layers
        self.fc_layers = nn.Sequential(
            # nn.Linear(256 * 2 * 2 * 2, 1024),  # *2 because of concatenation
            nn.Linear(81920,10240),
            nn.Tanh(),
            nn.Linear(10240,2560),
            nn.Tanh(),
            nn.Linear(2560,1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Bottom-up pathway
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        # Top-down pathway
        p3 = self.top_layer(c3)
        p2 = self.lateral_layer1(c2) + nn.functional.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lateral_layer2(c1) + nn.functional.interpolate(p2, scale_factor=2, mode='nearest')

        # Smooth
        p2 = self.smooth_layer1(p2)
        p1 = self.smooth_layer2(p1)


        # Flatten and concatenate features from both images
        out1 = p1.view(p1.size(0), -1)
        out2 = p2.view(p2.size(0), -1)
        out = torch.cat((out1, out2), dim=1)  # Concatenate along the feature dimension

        # Fully connected layers
        out = self.fc_layers(out)
        
        return out

# Create an instance of the model
# net = Net()
#print(net)


def eval_model(model, data_loader, criterion, device,save=False):
    total_rmse = 0
    total_samples = 0
    val_loss = 0
    model.eval()
    outputsCon = list()

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = data['image']
            labels = data['label']
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if save:
                outputsCon.append(outputs)
            outputs1 = torch.squeeze(outputs)
            loss = criterion(outputs1, labels.float())
            val_loss += loss.item()
            total_rmse += rmse(outputs1, labels) * labels.size(0)
            total_samples += labels.size(0)

    average_rmse = total_rmse / total_samples
    if save:
        return val_loss / len(data_loader), average_rmse, outputsCon
    return val_loss / len(data_loader), average_rmse

def compute_mean_std(loader):
    # Variances and means
    mean = 0.
    std = 0.
    total_images_count = 0

    for sample in loader:
        images = sample['image']
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples
        if total_images_count%1024 == 0:
            print(total_images_count)

    # Final mean and std
    mean /= total_images_count
    std /= total_images_count

    return mean, std

def rmse(outputs, labels):
    return torch.sqrt(torch.mean((outputs - labels) ** 2))

def main():
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # transform_load = transforms.Compose([
    #     transforms.ToTensor()])

    #Computed once to save Processing Time
    # print("Loading Dataset to Normalize")
    # dataset = VideoSoundDataset('./Dataset1','../Audio/SampledData.csv',transform=transform_load)
    # train_dataset = Subset(dataset, range(0, 77069))
    # ptrainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # mean, std = compute_mean_std(ptrainloader)
    # print(mean, std)

    mean = torch.tensor([0.3454, 0.3406, 0.2473])
    std = torch.tensor([0.2112, 0.187, 0.1789])

    transform_with_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
    
    print("Loading Dataset with Normalization")
    dataset = VideoSoundDataset('./Dataset1', '../Audio/SampledData.csv', transform=transform_with_norm)
    train_dataset_1 = Subset(dataset, range(0, 38083))
    train_dataset_2 = Subset(dataset, range(77069,len(dataset)))
    train_dataset = ConcatDataset([train_dataset_1,train_dataset_2])
    testset = Subset(dataset, range(38083, 77069))

    print(len(train_dataset))
    trainset, valset = torch.utils.data.random_split(train_dataset, [55000,5942], generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    print(f"Size of trainset: {len(trainset)}")
    print(f"Size of testset: {len(testset)}")
    print(f"Size of valset: {len(valset)}")

    learning_rate = 0.01
    l2_regularization = 0.0001
    epochs = 10

    # net = Net().to(device)
    # net = SiameseConvNet().to(device)
    net = FPN().to(device)
    print(net)

    criterion = nn.MSELoss(reduction='sum')  # Use Mean Squared Error loss for regression
    optimizer_adam = optim.Adam(net.parameters(), lr = learning_rate, weight_decay= l2_regularization)
    # optimizer_sgd = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    optimizer = optimizer_adam

    print("Started Training")
    try:
        for epoch in range(epochs):
            running_loss = 0.0
            total_rmse = 0.0
            total_samples = 0
            net.train()
            start = time.time()
            for data in trainloader:
                inputs = data['image']
                labels = data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                outputs1 = torch.squeeze(outputs)
                loss = criterion(outputs1, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_rmse += rmse(outputs1, labels) * labels.size(0)
                total_samples += labels.size(0)
                if total_samples%10240 == 0:
                    print(f"Finished Training for {total_samples}")
                
            end = time.time()
            print(f"Time last epoch: {end-start}")
            epoch_rmse = total_rmse / total_samples
            if epoch % 10 == 0:
                val_loss, val_rmse = eval_model(net, valloader, criterion, device)
                print(f'Epoch - {epoch} Loss: {running_loss / len(trainloader):.3f} RMSE: {epoch_rmse:.3f} Val Loss: {val_loss:.3f} Val RMSE: {val_rmse:.3f}')
            else:
                print(f'Epoch - {epoch} Loss: {running_loss / len(trainloader):.3f} RMSE: {epoch_rmse:.3f}')

        print('Finished training')
    except KeyboardInterrupt:
        pass

    try:
        net.eval()
        # Evaluate the model on the test set
        test_loss, test_acc, outputs = eval_model(net, testloader, criterion, device,save=True)
        concatenated_tensor = torch.cat(outputs, dim=0)
        print(len(concatenated_tensor))
        torch.save(concatenated_tensor, 'tensor_data_s.pt')
        
    except:
        with open('tensor_data_f.txt', 'w') as f:
            f.write(str(concatenated_tensor.cpu().numpy()))
            torch.save(concatenated_tensor, 'tensor_data_f.pt')
    
    print('Test loss: %.3f Average_RMSE: %.3f' % (test_loss, test_acc))



main()