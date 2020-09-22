import torch
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from model import Unet
from dataset import MyDataset, make_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把多个步骤整合到一起, channel=（channel-mean）/std, 因为是分别对三个通道处理
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # ->[0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1, 1] 3通道
    # transforms.Normalize([0.5], [0.5])  # 一通道灰度图
])
# mask只需要转换为tensor
y_transforms = transforms.ToTensor()


def train(args):
    my_dataset = MyDataset("../data/train", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(my_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    model = Unet(3, 1).to(device)
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        data_size = len(dataloaders.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataloaders:
            step += 1
            inputs = x.to(device)
            lables = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d, train_loss:%0.3f" % (step, (data_size - 1) // dataloaders.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    torch.save(model.state_dict(), 'model_weights.pth')
    return model


def test(args):
    my_dataset = MyDataset("../data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(my_dataset, batch_size=1)
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    plt.ion()
    with torch.no_grad():
        count = 0
        for x, _ in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            predict_path = os.path.join("../data/predict/", "%03d_predict.png" % count)
            plt.imsave(predict_path, img_y)
            plt.imshow(img_y)
            plt.pause(0.1)
            count += 1
        plt.show()


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--epochs", type=int, default=10)
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--model_path", type=str, help="the path of model weight file")
    args = parse.parse_args()

    args.model_path = "weights_19.pth"
    print(args)

    # train and test
    # train(args)
    test(args)


if __name__ == '__main__':
    main()