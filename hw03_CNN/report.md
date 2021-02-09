

## 一、数据预处理

本次实验的主要任务是将图片分类，总共为11类，图片格式是为类别_图片序号，所以只要字符串截取一下就行。

<img src="/Users/zhanghailin/Library/Application Support/typora-user-images/截屏2021-02-09 上午10.15.54.png" alt="截屏2021-02-09 上午10.15.54" style="zoom:50%;" />

主要用到DataSet和DataLoader。

Dataset是一个包装类，用来将数据包装为Dataset类，我们只需要自己写一个类然后继承Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。

当我们集成了一个Dataset类之后，我们需要重写 **len** 方法，该方法提供了dataset的大小； **getitem** 方法， 该方法支持从 0 到 len(self)的索引。

```python
def read_file(path, flag):
    """
    读取文件目录里的内容
    :param path: 文件夹位置
    :param flag: 1训练集或验证集 0测试集
    """
    image_dir = os.listdir(path)
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros(len(image_dir))
    
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :, :] = cv2.resize(img, (128, 128)) # 将图片大小变为128*128
        if flag:
            y[i] = file.split('_')[0]
    
    if flag:
        return x, y
    else:
        return x
   
```

```python
class ImgDataset(Dataset):
    """
    实现对数据的封装
    """
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        res_x = self.x[index]
        if self.transform is not None:
            res_x = self.transform(res_x)
        if self.y is not None:
            res_y = self.y[index]
            return res_x, res_y
        else:
            return res_x
        

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # 随机水平翻转图片
    transforms.RandomRotation(15), # 随机旋转图片15度
    transforms.ToTensor() # 将图片变为Tensor [H, W, C]-->[C, H, W]
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
```



## 二、模型建立



### 2.1 5层卷积层+5层池化层+3层全连接层（原模型）

本次实验采用CNN卷积神经网络，网络结构为5层卷积层+5层池化层+3层全连接层。

注意连到全连接层时，需要将tensor展开，从[n, 512, 4, 4]->[n, 512 * 4 * 4]，n为batch_size。

```python
class Classifier1(nn.Module):
    """
    构建神经网络1：5层卷积+5层池化+3层全连接
    """
    def __init__(self):
        super(Classifier1, self).__init__()
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.cnn = nn.Sequential(           # input: 3 * 128 * 128
            # 卷积层1
            nn.Conv2d(3, 64, 3, 1, 1),       # output: 64 * 128 * 128
            nn.BatchNorm2d(64), # 归一化处理，可以使每一个batch的分布都在高斯分布附近，这样可以使用更大的学习率，加快训练速度
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),           # output: 64 * 64 * 64
            
            # 卷积层2
            nn.Conv2d(64, 128, 3, 1, 1),     # output: 128 * 64 * 64
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),           # output: 128 * 32 * 32
            
             # 卷积层3
            nn.Conv2d(128, 256, 3, 1, 1),    # output: 256 * 32 * 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),           # output: 256 * 16 * 16
            
            # 卷积层4
            nn.Conv2d(256, 512, 3, 1, 1),    # output: 512 * 16 * 16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),           # output: 512 * 8 * 8
            
            # 卷积层5
            nn.Conv2d(512, 512, 3, 1, 1),    # output: 512 * 8 * 8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)            # output: 512 * 4 * 4
        ) 
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024), # 全连接层
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )
        
    def forward(self, x):
        cnn_out = self.cnn(x)
        flatten = cnn_out.view(cnn_out.size()[0], -1) # 将Tensor展开
        return self.fc(flatten)
        
```



### 2.2 3层卷积层+3层池化层+3层全连接层（深度减半、参数量与原模型相当的模型）

由于作业说明中有该要求，所以还定义了另外两种网络结构。

<img src="/Users/zhanghailin/Library/Application Support/typora-user-images/截屏2021-02-09 上午10.26.17.png" alt="截屏2021-02-09 上午10.26.17" style="zoom:50%;" />

```python
class Classifier2(nn.Module):
    """
    构建神经网络2：3层卷积+3层池化+3层全连接
    """
    def __init__(self):
        super(Classifier2, self).__init__()
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.cnn = nn.Sequential(           # input: 3 * 128 * 128
            # 卷积层1
            nn.Conv2d(3, 64, 3, 1, 1),       # output: 64 * 128 * 128
            nn.BatchNorm2d(64), # 归一化处理
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),           # output: 64 * 32 * 32
            
            # 卷积层2
            nn.Conv2d(64, 512, 3, 1, 1),     # output: 512 * 32 * 32
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),           # output: 512 * 8 * 8
            
             # 卷积层3
            nn.Conv2d(512, 512, 3, 1, 1),    # output: 512 * 8 * 8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),           # output: 512 * 4 * 4
        ) 
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024), # 全连接层
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )
        
    def forward(self, x):
        cnn_out = self.cnn(x)
        flatten = cnn_out.view(cnn_out.size()[0], -1) # 将Tensor展开
        return self.fc(flatten)
        
```



### 2.3  5层卷积层+5层池化层+2层全连接层（简单DNN）

```python
class Classifier3(nn.Module):
    """
    构建神经网络3：5层卷积+5层池化+2层全连接
    """
    def __init__(self):
        super(Classifier3, self).__init__()
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.cnn = nn.Sequential(           # input: 3 * 128 * 128
            # 卷积层1
            nn.Conv2d(3, 64, 3, 1, 1),       # output: 64 * 128 * 128
            nn.BatchNorm2d(64), # 归一化处理
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),           # output: 64 * 64 * 64
            
            # 卷积层2
            nn.Conv2d(64, 128, 3, 1, 1),     # output: 128 * 64 * 64
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),           # output: 128 * 32 * 32
            
             # 卷积层3
            nn.Conv2d(128, 256, 3, 1, 1),    # output: 256 * 32 * 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),           # output: 256 * 16 * 16
            
            # 卷积层4
            nn.Conv2d(256, 512, 3, 1, 1),    # output: 512 * 16 * 16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),           # output: 512 * 8 * 8
            
            # 卷积层5
            nn.Conv2d(512, 512, 3, 1, 1),    # output: 512 * 8 * 8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)            # output: 512 * 4 * 4
        ) 
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024), # 全连接层
            nn.ReLU(),
            nn.Linear(1024, 11)
        )
        
    def forward(self, x):
        cnn_out = self.cnn(x)
        flatten = cnn_out.view(cnn_out.size()[0], -1) # 将Tensor展开
        return self.fc(flatten)
        
```



## 三、模型训练

本模型采用交叉熵作为损失函数，Adam为优化器，总共训练了30epoch。

```python
def train_model(train_loader, val_loader, train_len, val_len):
    """
    模型训练
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 构建神经网络1：5层卷积+5层池化+3层全连接
    # model = Classifier1().to(device)
    
    # 构建神经网络2：3层卷积+3层池化+3层全连接
    model = Classifier2().to(device)
    
    # 构建神经网络3：5层卷积+5层池化+2层全连接
    # model = Classifier3().to(device)
    
    loss = nn.CrossEntropyLoss() # 使用交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 30
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        
        # 保证BN层(Batch Normalization)用每一批数据的均值和方差，而对于Dropout层，随机取一部分网络连接来训练更新参数
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad() # 清空梯度，否则会一直累加
            train_pred = model(data[0].to(device)) # data[0]：x data[1]：y
            batch_loss = loss(train_pred, data[1].to(device))
            batch_loss.backward()
            optimizer.step() # 更新参数
            
            # .data表示将Variable中的Tensor取出来
            # train_pred是(50，11)的数据，np.argmax()返回最大值的索引，axis=1则是对行进行，返回的索引正好就对应了标签，然后和y真实标签比较，则可得到分类正确的数量
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
         
        
        # 保证BN用全部训练数据的均值和方差，而对于Dropout层，利用到了所有网络连接
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].to(device))
                batch_loss = loss(val_pred, data[1].to(device))
                
                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()
                
         
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, epochs, time.time() - epoch_start_time, \
               train_acc / train_len, train_loss / train_len, val_acc / val_len,
               val_loss / val_len))
        
    return model
    
```

模型1训练结果：

![336481612615714_.pic_hd](/Users/zhanghailin/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b3e5f8fbb028468c0aaf2a572e71971e/Message/MessageTemp/f487012e881f6b026ef2707a37707f8e/Image/336481612615714_.pic_hd.jpg)

模型2训练结果：

![337911612664090_.pic_hd](/Users/zhanghailin/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b3e5f8fbb028468c0aaf2a572e71971e/Message/MessageTemp/f487012e881f6b026ef2707a37707f8e/Image/337911612664090_.pic_hd.jpg)

模型3训练结果：

![337961612666075_.pic_hd](/Users/zhanghailin/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/b3e5f8fbb028468c0aaf2a572e71971e/Message/MessageTemp/f487012e881f6b026ef2707a37707f8e/Image/337961612666075_.pic_hd.jpg)



## 四、模型测试

```python
def predict_model(test_loader, model):
    """
    模型预测
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    result = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                result.append(y)
    return result

def write_file(result):
    with open('result.csv', mode='w') as f:
        f.write('Id,Category\n')
        for i, label in enumerate(result):
            f.write('{},{}\n'.format(i, label))
```





## 部分API解析



**torch.nn.Conv2d：**

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

- in_channels(`int`) – 输入信号的通道
- out_channels(`int`) – 卷积产生的通道
- kerner_size(`int` or `tuple`) - 卷积核的尺寸
- stride(`int` or `tuple`, `optional`) - 卷积步长
- padding(`int` or `tuple`, `optional`) - 输入的每一条边补充0的层数
- dilation(`int` or `tuple`, `optional`) – 卷积核元素之间的间距
- groups(`int`, `optional`) – 从输入通道到输出通道的阻塞连接数
- bias(`bool`, `optional`) - 如果`bias=True`，添加偏置



**torch.nn.BatchNorm2d：**

在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
$$
y=\frac{x-mean(x)}{\sqrt{Var(x)} +\epsilon}*\gamma+\beta
$$


```python
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
```

- num_features： 来自期望输入的特征数，该期望输入的大小为`C来自输入大小(N,C,H,W)`
- eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
- momentum： 动态均值和动态方差所使用的动量。默认为0.1。
- affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数。



**torch.nn.MaxPool2d：**

```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

- kernel_size(`int` or `tuple`) - max pooling的窗口大小
- stride(`int` or `tuple`, `optional`) - max pooling的窗口移动的步长。默认值是`kernel_size`
- padding(`int` or `tuple`, `optional`) - 输入的每一条边补充0的层数
- dilation(`int` or `tuple`, `optional`) – 一个控制窗口中元素步幅的参数
- return_indices - 如果等于`True`，会返回输出最大值的序号，对于上采样操作会有帮助
- ceil_mode - 如果等于`True`，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作



**torch.nn.Linear：**

对输入数据做线性变换$$ y=Ax+b $$，即全连接层。

```python
torch.nn.Linear(in_features, out_features, bias=True)
```

- in_features - 每个输入样本的大小
- out_features - 每个输出样本的大小
- bias - 若设置为False，这层不会学习偏置。默认值：True