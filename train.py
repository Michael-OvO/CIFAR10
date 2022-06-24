import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from time import sleep
from tqdm import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)

gpu = torch.cuda.is_available()
learning_rate = 0.01
epoch = 10
log_dir = './logs'


train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

writer = SummaryWriter(log_dir)

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print("---------------------准备开始训练---------------------")
print("训练数据量:", train_data_size)
print("测试数据量:", test_data_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        return self.model(x)

# 实例化
if gpu:
    net = Net().cuda()
else:
    net = Net()

# 损失函数
if gpu:
    loss_fn = nn.CrossEntropyLoss().cuda()
else:
    loss_fn = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.SGD(net.parameters(),lr=learning_rate)

total_train_cnt = 0
total_test_cnt = 0

# 训练
def train(epoch):
    total_loss = 0 
    total_train_step = 0
    global total_train_cnt
    print("---------------------第{}轮训练开始---------------------".format(epoch))
    net.train()
    pbar = tqdm(train_loader)
    for data in train_loader:
        img,target = data
        if gpu:
            img = img.cuda()
            target = target.cuda()
        outputs = net(img)
        loss = loss_fn(outputs,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_train_step += 1
        total_train_cnt += 1
        writer.add_scalar('loss',loss.item(),total_train_cnt)
        pbar.set_description("Epoch {}: loss = {:.6f}".format(epoch,total_loss/total_train_step))   
        pbar.update()
        
def test(epoch):
    net.eval()
    correct = 0
    total = 0
    global total_test_cnt
    with torch.no_grad():
        pbar_val = tqdm(test_loader)
        for data in test_loader:
            img,target = data
            if gpu:
                img = img.cuda()
                target = target.cuda()
            outputs = net(img)
            _,pred = torch.max(outputs,1)
            total += target.size(0)
            correct += (pred == target).sum().item()
            pbar_val.set_description("测试中: acc = {:.6f}".format(correct/total))   
            pbar_val.update()
            total_test_cnt += 1
            writer.add_scalar('acc',correct/total,total_test_cnt)
    sleep(1)
    print("---------------------测试结果:---------------------")
    print("测试数据量:", total)
    print("测试准确率:", correct/total)
    print("---------------------测试结束---------------------")
    torch.save(net,"./model/net_{}.pth".format(epoch))
    print("Model saved to net_{}.pth".format(epoch))


for i in range(epoch):
    train(i+1)
    test(i+1)


mod = torch.load("./model/net_10.pth",map_location=torch.device('cpu'))
print(mod)
mod.eval()



fig = plt.figure()
for i in range(10):
    data = test_loader.dataset[i]
    img,label = data
    img = torch.reshape(img,(1,3,32,32))
    with torch.no_grad():
        output = mod(img)
    fig.add_subplot(2,5,i+1)
    plt.imshow(img.numpy().squeeze().transpose(1,2,0))
    plt.title("Prediction: {} Original:{}".format(classes[torch.argmax(output)],label))
plt.show()


img_path = "./imgs/plane2.png"
image = Image.open(img_path)
image = image.convert('RGB')
trans = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                        torchvision.transforms.ToTensor()])
image = trans(image)
print(image.shape)
image = torch.reshape(image,(1,3,32,32))
with torch.no_grad():
    output = mod(image)
print(classes[output.argmax(1).item()])
