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
from transformers import ViTModel
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
NUM_EPOCH = 20


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


class ResNeXtViTBlock(nn.Module):
    def __init__(self, original_block,
                 vit_model_name="WinKawaks/vit-tiny-patch16-224"):
        super().__init__()
        self.original_block = original_block
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.reduce_dim = nn.Conv2d(original_block.conv3.out_channels, 3,
                                    kernel_size=1)
        self.expand_dim = nn.Linear(192, original_block.conv3.out_channels)

    def forward(self, x):
        residual = x
        x = self.original_block(x)
        x = self.reduce_dim(x)
        x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                      align_corners=False)
        vit_out = self.vit(pixel_values=x).last_hidden_state[:, 0, :]
        vit_out = self.expand_dim(vit_out)

        return residual + vit_out.unsqueeze(-1).unsqueeze(-1)


class ResNeXtWithViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnext = resnext101_32x8d(pretrained=True)
        for name, module in self.resnext.named_children():
            if name in ["layer4"]:
                for i, bottleneck in enumerate(module):
                    if (i == 2):
                        module[i] = ResNeXtViTBlock(bottleneck)
        self.resnext.fc = nn.Linear(2048, NUM_CLASSES)

    def forward(self, x):
        return self.resnext(x)


model_1 = resnext101_32x8d(pretrained=True)  # scheduler
model_2 = resnext101_32x8d(pretrained=True)  # scheduler + freeze
model_3 = ResNeXtWithViT()  # ResNeXtWithViT + scheduler

model_1.fc = nn.Linear(in_features=2048, out_features=NUM_CLASSES)
model_2.fc = nn.Linear(in_features=2048, out_features=NUM_CLASSES)

model_1.load_state_dict(torch.load('ensemble_model1.pt', weights_only=True))
model_2.load_state_dict(torch.load('ensemble_model2.pth', weights_only=True)
                        ['model'])
model_3.load_state_dict(torch.load('ensemble_model3.pt', weights_only=True))


class EnsembleModel(nn.Module):
    def __init__(self, model_1, model_2, model_3):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3
        self.classifier = nn.Linear(NUM_CLASSES * 3, NUM_CLASSES)

    def forward(self, x):
        x1 = self.model_1(x)
        x2 = self.model_2(x)
        x3 = self.model_3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        output = self.classifier(x)
        return output


ensemble_model = EnsembleModel(model_1, model_2, model_3)

for param in ensemble_model.parameters():
    param.requires_grad = False

for param in ensemble_model.classifier.parameters():
    param.requires_grad = True

ensemble_model = ensemble_model.cuda()

num_params = 0
for param in ensemble_model.parameters():
    if param.requires_grad:
        num_params += param.numel()
print("#Params:", num_params)


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


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                   ensemble_model.parameters()),
                            lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=NUM_EPOCH,
    eta_min=1e-6)

last_epoch = 0

for epoch_num in tqdm(range(last_epoch + 1, NUM_EPOCH + 1)):

    acc_train, loss_train, lr = train(ensemble_model, train_dataflow,
                                      criterion, optimizer, scheduler)
    acc_val, loss_val = evaluate(ensemble_model, val_dataflow)
    print("epoch %d:\ttrain acc:%.4f\tval acc:%.4f\ttrain loss:%.4f\t\
        val loss:%.4f\tlr:%.8f"
          % (epoch_num, acc_train, acc_val, loss_train, loss_val, lr))

    torch.save({'epoch': epoch_num,
                'model': ensemble_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
               'train_epoch{}.pth'.format(epoch_num))
