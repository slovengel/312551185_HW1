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

# hyper-parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCH = 200


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
print(model)

model.fc = nn.Linear(in_features=2048, out_features=NUM_CLASSES)
print(model)

num_params = 0
for param in model.parameters():
    if param.requires_grad:
        num_params += param.numel()
print("#Params:", num_params)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=NUM_EPOCH * len(train_dataflow),
    eta_min=1e-6)


def train_one_batch(
  model: nn.Module,
  criterion: nn.Module,
  optimizer: Optimizer,
  inputs: torch.Tensor,
  targets: torch.Tensor,
  scheduler: LRScheduler,
) -> None:

    optimizer.zero_grad()
    targets_pred = model(inputs)
    loss = criterion(targets_pred, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()


def train(
    model: nn.Module,
    dataflow: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
):

    model.train()

    for inputs, targets in tqdm(train_dataflow, desc='train', leave=False):
        inputs = inputs.cuda()
        targets = targets.cuda()

        train_one_batch(
            model, criterion, optimizer, inputs, targets, scheduler)


def evaluate(
  model: nn.Module,
  dataflow: DataLoader
) -> float:

    model.eval()
    num_samples = 0
    num_correct = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataflow, desc="eval", leave=False):
            inputs = inputs.cuda()
            targets = targets.cuda()
            predicts = model(inputs)
            predicts = torch.argmax(predicts, dim=1)
            num_samples += targets.size(0)
            num_correct += (predicts == targets).sum()

    return (num_correct / num_samples * 100).item()


for epoch_num in tqdm(range(1, NUM_EPOCH + 1)):
    train(model, train_dataflow, criterion, optimizer, scheduler)
    acc = evaluate(model, val_dataflow)
    print(f"epoch {epoch_num}:", acc)
    torch.save(model.state_dict(), 'model' + str(epoch_num) + '.pt')

print(f"final accuracy: {acc}")

torch.save(model.state_dict(), 'model.pt')
