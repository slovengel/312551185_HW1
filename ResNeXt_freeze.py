import random
import os
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, \
    RandomHorizontalFlip, RandomAffine, RandomPerspective, \
    ToTensor, Normalize, Resize
from tqdm.auto import tqdm
from PIL import Image
from torchvision.models import resnext101_32x8d

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

NUM_CLASSES = 100
resume = False
# hyper-parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCH = 50


class HW1Dataset(Dataset):

    def __init__(self, data, transform):
        self.root_dir = \
            os.path.join("C:\\Users\\Documents\\VRDL\\hw1-data\\data", data)
        self.inputs = []
        self.targets = []
        for label in os.listdir(self.root_dir):
            for name in os.listdir(os.path.join(self.root_dir, label)):
                self.inputs.append(name)
                self.targets.append(label)
        self.transform = transform
        self.len = len(self.inputs)

    def __getitem__(self, index):
        name = self.inputs[index]
        label = self.targets[index]
        img = Image.open(self.root_dir + '/' + label + '/' + name)
        img = img.convert("RGB")
        img = self.transform(img)
        label = int(label)
        return img, label

    def __len__(self):
        return self.len


transform_train = Compose([RandomResizedCrop(224),
                           RandomHorizontalFlip(),
                           RandomAffine(degrees=(-30, 30)),
                           RandomPerspective(),
                           ToTensor(),
                           Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

transform_val = Compose([Resize([224, 224]),
                         ToTensor(),
                         Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))])

train_dataset = HW1Dataset('train', transform_train)
val_dataset = HW1Dataset('val', transform_val)

train_dataflow = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_dataflow = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

for inputs, targets in train_dataflow:
    print(f"[inputs] dtype: {inputs.dtype}, shape: {inputs.shape}")
    print(f"[targets] dtype: {targets.dtype}, shape: {targets.shape}")
    break

model = resnext101_32x8d(pretrained=True)

model.fc = nn.Linear(in_features=2048, out_features=NUM_CLASSES)
print(model)

for name, param in model.named_parameters():
    if name in ['conv1.weight', 'bn1.weight', 'bn1.bias'] \
            or 'layer1' in name or 'layer2' in name \
            or 'layer3' in name or 'layer4' in name:
        param.requires_grad = False

for name, param in model.named_parameters():
    print(name)
    print("requires_grad: ", param.requires_grad)

num_params = 0
for param in model.parameters():
    if param.requires_grad:
        num_params += param.numel()
print("#Params:", num_params)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def train_one_batch(
  model: nn.Module,
  criterion: nn.Module,
  optimizer: Optimizer,
  inputs: torch.Tensor,
  targets: torch.Tensor,
  scheduler: LRScheduler,
) -> None:

    num_correct_batch = 0

    optimizer.zero_grad()
    targets_pred = model(inputs)
    loss = criterion(targets_pred, targets)
    loss.backward()
    optimizer.step()

    targets_pred = torch.argmax(targets_pred, dim=1)

    num_correct_batch += (targets_pred == targets).sum()

    return loss.item(), num_correct_batch


def train(
    model: nn.Module,
    dataflow: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
):

    model.train()

    num_samples = 0
    num_correct = 0
    loss_sum = 0

    for inputs, targets in tqdm(train_dataflow, desc='train', leave=False):
        inputs = inputs.cuda()
        targets = targets.cuda()

        loss, num_correct_batch = train_one_batch(
            model, criterion, optimizer, inputs, targets, scheduler)
        loss_sum += loss

        num_samples += targets.size(0)
        num_correct += num_correct_batch

    scheduler.step()

    return (num_correct / num_samples * 100).item(), \
        (loss_sum / len(train_dataflow)), (optimizer.param_groups[0]['lr'])


def evaluate(
  model: nn.Module,
  dataflow: DataLoader
) -> float:

    model.eval()

    num_samples = 0
    num_correct = 0
    loss_sum = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataflow, desc="eval", leave=False):
            inputs = inputs.cuda()
            targets = targets.cuda()
            predicts = model(inputs)
            loss = criterion(predicts, targets)
            loss_sum += loss
            predicts = torch.argmax(predicts, dim=1)
            num_samples += targets.size(0)
            num_correct += (predicts == targets).sum()

    return (num_correct / num_samples * 100).item(), \
        (loss_sum / len(val_dataflow))


def unfreeze(unfreeze_name):
    for name, param in model.named_parameters():
        if unfreeze_name in name:
            param.requires_grad = True
            optimizer.add_param_group({'params': param})


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=10,
    eta_min=1e-6)

last_epoch = 0
if resume:
    checkpoint = torch.load('train_epoch{}.pth'.format(last_epoch))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    last_epoch = checkpoint['epoch']

for epoch_num in tqdm(range(last_epoch + 1, NUM_EPOCH + 1)):

    acc_train, loss_train, lr = train(model, train_dataflow, criterion,
                                      optimizer, scheduler)
    acc_val, loss_val = evaluate(model, val_dataflow)
    print("epoch %d:\ttrain acc:%.4f\tval acc:%.4f\ttrain loss:%.4f\t\
        val loss:%.4f\tlr:%.8f"
          % (epoch_num, acc_train, acc_val, loss_train, loss_val, lr))

    torch.save({'epoch': epoch_num,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
               'train_epoch{}.pth'.format(epoch_num))

    if epoch_num == 10:
        unfreeze('layer4.')
        new_lr = 1e-3
        T_max = 10
        eta_min = 1e-6
    elif epoch_num == 20:
        unfreeze('layer3.')
        new_lr = 1e-3
        T_max = 10
        eta_min = 1e-6
    elif epoch_num == 30:
        unfreeze('layer2.')
        unfreeze('layer1.')
        new_lr = 1e-3
        T_max = 20
        eta_min = 1e-6
    else:
        continue

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min)
