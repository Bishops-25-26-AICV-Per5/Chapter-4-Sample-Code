"""
Author: TBSDrJ
Date: Spring 2025
Purpose: A PyTorch solution to the Cat-Dog-Panda dataset,
using an AlexNet-style CNN.
"""
import atexit
import pathlib
import time

import cv2
import torch, torchvision
import torchvision.transforms.v2 as v2

@atexit.register
def clean_up() -> None:
    torch.save(model, 
        f"saves/model_torch_{TIME_STAMP}_{epoch:03d}.pt")

BATCH_SIZE = 32
INPUT_SHAPE = (3, 224, 224)
TIME_STAMP = time.strftime("%Y_%m_%d_%H_%M")

def my_print(*args, **kwargs) -> None:
    with open("saves/printout_" + TIME_STAMP + ".txt", "a") as f:
        print(*args, **kwargs)
        if "end" in kwargs and kwargs["end"] == "\r":
            del kwargs["end"]
        print(*args, **kwargs, file=f)

# If you have a Mac laptop, this will make it use the GPU power.
torch.set_default_device(torch.device("mps"))
# Set random seed so that results are reproducible.
torch.manual_seed(37)

class Dataset(torch.utils.data.Dataset):
    """Class for loading data."""
    def __init__(self, image_path: str):
        my_print("Loading images...")
        image_path = pathlib.Path(image_path)
        self.cls_names = []
        self.image_set = []
        for folder in image_path.iterdir():
            if folder.is_dir():
                self.cls_names.append(folder.name)
        self.cls_names.sort()
        self.cls_counts = [0] * len(self.cls_names)
        for folder in image_path.iterdir():
            if folder.is_dir():
                for image in folder.iterdir():
                    img = cv2.imread(image)
                    if img is not None:                    
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img / 255.0
                        img = cv2.resize(img, INPUT_SHAPE[1:3])
                        img = img.transpose([2, 0, 1])
                        cls_idx = self.cls_names.index(folder.name)
                        self.image_set.append((
                            torch.tensor(img, dtype=torch.float32).to('mps'), 
                            torch.tensor(cls_idx, dtype=torch.float32).to('mps')
                        ))
                        self.cls_counts[cls_idx] += 1
    
    def __len__(self) -> int:
        return len(self.image_set)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.image_set[idx]

class Model(torch.nn.Module):
    def __init__(self, input_shape: (int, int, int)):
        super().__init__()
        self.input_shape = input_shape
        # Reusable layers
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3, stride=2)

        # Now into the architecture
        # 3 x 224 x 224
        self.zp0 = torch.nn.ZeroPad2d((1, 2, 1, 2))
        # 3 x 227 x 227
        self.conv0 = torch.nn.Conv2d(3, 24, 11, stride=4)
        # (227 - 11)/4 + 1 = 216/4 + 1 = 54 + 1
        # 24 x 55 x 55
        # Maxpool, 3x3 stride=2: (55 - 3)/2 + 1 = 52/2 + 1 = 27
        # 24 x 27 x 27
        self.bn0 = torch.nn.BatchNorm2d(24)
        self.zp1 = torch.nn.ZeroPad2d((1, 1, 1, 1))
        # 24 x 29 x 29
        self.conv1 = torch.nn.Conv2d(24, 32, 3, stride=1)
        # (29 - 3)/1 + 1 = 27
        # 32 x 27 x 27
        # Maxpool, 3x3 stride=2: (27 - 3)/2 + 1 = 13
        # 32 x 13 x 13
        self.bn1 = torch.nn.BatchNorm2d(32)
        # re-use self.zp1
        # 32 x 15 x 15
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1)
        # (15 - 3)/1 + 1 = 13
        # 32 x 13 x 13
        # No maxpool this time around
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.flatten = torch.nn.Flatten()
        self.linear0 = torch.nn.Linear(5408, 1024)
        self.linear1 = torch.nn.Linear(1024, 256)
        self.linear2 = torch.nn.Linear(256, 64)
        self.linear3 = torch.nn.Linear(64, 3)
        self.softmax = torch.nn.Softmax(dim=1)

        # Re-assign the weight of conv0, bounded away from 0:
        t = torch.rand((24, 3, 11, 11)) / 0.1 + 0.05
        t.to('mps')
        t = torch.nn.parameter.Parameter(data=t)
        self.conv0.weight = t

    def forward(self, x: torch.Tensor):
        # Block 0
        y = self.bn0(self.maxpool(self.relu(self.conv0(self.zp0(x)))))
        # Block 1
        y = self.bn1(self.maxpool(self.relu(self.conv1(self.zp1(y)))))
        # Block 2
        y = self.bn2(self.relu(self.conv2(self.zp1(y))))
        y = self.relu(self.linear0(self.flatten(y)))
        y = self.relu(self.linear1(y))
        y = self.relu(self.linear2(y))
        y = self.softmax(self.linear3(y))
        return y

def get_dataloaders(
            full_dataset: Dataset, batch_size: int, train_proportion: float
    ) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """Set up dataloaders for train and validation sets."""
    path_to_images = "animals"

    generator = torch.Generator(device="mps").manual_seed(37)
    train_set, valid_set = torch.utils.data.random_split(
            full_dataset,
            [train_proportion, 1-train_proportion],
            generator = generator,
    )
    flipped = v2.RandomHorizontalFlip(1.0)(train_set)
    train_set = torch.utils.data.ConcatDataset([train_set, flipped])
    # Notice brightness transform applies to both base train set + flipped
    brightness = v2.ColorJitter(brightness=0.25)(train_set)
    hue = v2.ColorJitter(hue=0.25)(train_set)
    train_set = torch.utils.data.ConcatDataset([train_set, brightness, hue])

    train = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    valid = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    return train, valid

def main():
    global model, epoch

    # save code
    with open(__file__, "r") as f:
        this_code = f.read()
    with open("saves/code_" + TIME_STAMP + ".py", "w") as f:
        print(this_code, file=f)

    dataset = Dataset('animals')
    train, valid = get_dataloaders(
        dataset, batch_size=BATCH_SIZE, train_proportion=0.8)

    # Number of batches
    my_print(f"Train batches = {len(train)}")
    my_print(f"Validation batches = {len(valid)}")

    model = Model(INPUT_SHAPE)

    lr = .001
    my_print(f"Initial learning rate: {lr:.8f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sch = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
            lambda epoch: 0.95)
    loss_fn = torch.nn.CrossEntropyLoss()
    torch.save(model, f"saves/model_torch_{TIME_STAMP}_000.pt")

    for epoch in range(100):
        batch_losses = []
        batch_accuracies = []
        data_load_time = 0
        forward_time = 0
        backprop_time = 0
        model.train()
        my_print(f"=== Epoch {epoch+1} ===")
        for (image_batch, label_batch) in train:
            preds = model(image_batch)
            loss = loss_fn(preds, label_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(float(loss.detach()))
            cur_loss = sum(batch_losses)/len(batch_losses)
            batch_accuracies.append(
                int(sum(preds.argmax(1) == label_batch))/len(label_batch))
            cur_acc = sum(batch_accuracies)/len(batch_accuracies)
            my_print("Train:", end="\t\t")
            my_print(f"Batch: {len(batch_losses)}", end="\t")
            my_print(f"Loss: {round(cur_loss, 4)}", end="\t")
            my_print(f"Accuracy: {round(cur_acc, 4)}", end="\r")
        my_print()
        lr_sch.step()
        batch_losses = []
        batch_accuracies = []
        model.eval()
        for (image_batch, label_batch) in valid:
            with torch.no_grad():
                preds = model(image_batch)
                loss = loss_fn(preds, label_batch)
                batch_losses.append(float(loss.detach()))
                cur_loss = sum(batch_losses)/len(batch_losses)
                batch_accuracies.append(
                    int(sum(preds.argmax(1) == label_batch))/len(label_batch))
                cur_acc = sum(batch_accuracies)/len(batch_accuracies)
                my_print("Validation:", end="\t")
                my_print(f"Batch: {len(batch_losses)}", end="\t")
                my_print(f"Loss: {round(cur_loss, 4)}", end="\t")
                my_print(f"Accuracy: {round(cur_acc, 4)}", end="\r")
        my_print()
        my_print(f"Current Learning Rate: {lr_sch.get_lr()[0]}")
        if (epoch+1) % 5 == 0:
            torch.save(model, 
                f"saves/model_torch_{TIME_STAMP}_{epoch+1:03}.pt")

if __name__ == "__main__":
    main()