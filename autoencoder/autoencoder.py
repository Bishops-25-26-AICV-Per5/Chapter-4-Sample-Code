"""
Author: TBSDrJ
Date: Spring 2025
Purpose: An example of an Autoencoder to detect pandas in images.
"""
import pathlib, time
import torch, torchvision
import torchvision.transforms.v2 as v2
import cv2
import numpy as np

BATCH_SIZE = 32

# If you have a Mac laptop, this will make it use the GPU power.
torch.set_default_device(torch.device("mps"))
# Set random seed so that results are reproducible.
torch.manual_seed(37)

class MyZipDataset(torch.utils.data.Dataset):
    """Build Torch Dataset from the loaded EEG data"""
    def __init__(self, path_str: str):
        dtype = torch.float32
        base_path = pathlib.Path(path_str)
        self.inputs = []
        for path in base_path.iterdir():
            if not path.is_dir():
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    img = np.moveaxis(img, -1, 0)
                    img = torch.tensor(img, dtype=dtype)
                    self.inputs.append(img)
        self.length = len(self.inputs)
    def __len__(self) -> int:
        return self.length
    def __getitem__(self, i: int) -> torch.Tensor:
        inp = self.inputs[i]
        return inp


def get_dataloaders(
            *, batch_size: int, train_proportion: float
    ) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """Set up dataloaders for train and validation sets."""
    path_to_images = "animals/panda"
    full_dataset = MyZipDataset(path_to_images)
    generator = torch.Generator(device="mps").manual_seed(37)
    train_set, valid_set = torch.utils.data.random_split(
            full_dataset,
            [train_proportion, 1-train_proportion],
            generator = generator,
    )
    train = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    valid = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    return train, valid

def main():
    train, valid = get_dataloaders(batch_size=BATCH_SIZE, train_proportion=0.8)

    # Number of batches
    print(f"Train batches = {len(train)}")
    print(f"Validation batches = {len(valid)}")

    # Set up model with named sections so we can optimize separately
    encoder = torch.nn.Sequential( 
            # Input: 3 x 224 x 224
            torch.nn.ZeroPad2d((1, 2, 1, 2)),
            # 227 x 227 x 3
            torch.nn.Conv2d(3, 24, 11, stride=4),
            # 55 x 55 x 24
            torch.nn.ZeroPad2d((1, 1, 1, 1)),
            # 57 x 57 x 24
            torch.nn.Conv2d(24, 32, 3, stride=2),
            # 28 x 28 x 32
            torch.nn.ZeroPad2d((0, 1, 0, 1)),
            # 29 x 29 x 32
            torch.nn.Conv2d(32, 32, 3, stride=2),
            # 14 x 14 x 32
    )
    decoder = torch.nn.Sequential(
        # Input: 14 x 14 x 32
        torch.nn.ConvTranspose2d(32, 32, 3, stride=2),
        # 29 x 29 x 32
        v2.CenterCrop((28, 28)),
        # 28 x 28 x 32
        torch.nn.ConvTranspose2d(32, 24, 3, stride=2),
        # 57 x 57 x 24
        v2.CenterCrop((55, 55)),
        # 55 x 55 x 24
        torch.nn.ConvTranspose2d(24, 3, 11, stride=4),
        # 227 x 227 x 3
        v2.CenterCrop((224, 224)),
        # Output: 224 x 224 x 3
    )
    lr = .001
    print(f"Initial learning rate: {lr:.8f}")
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    encoder_lr_sch = torch.optim.lr_scheduler.MultiplicativeLR(encoder_optimizer, 
            lambda epoch: 0.975)
    encoder.train()
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    decoder_lr_sch = torch.optim.lr_scheduler.MultiplicativeLR(decoder_optimizer, 
            lambda epoch: 0.975)
    decoder.train()
    loss_fn = torch.nn.MSELoss()
    torch.save(encoder, 'saves/encoder_000.pt')
    torch.save(decoder, 'saves/decoder_000.pt')

    for i in range(150):
        batch_losses = []
        batch_accuracies = []
        data_load_time = 0
        forward_time = 0
        backprop_time = 0
        print(f"=== Epoch {i+1} ===")
        for image_batch in train:
            fwd = encoder(image_batch)
            preds = decoder(fwd)
            loss = loss_fn(preds, image_batch)
            loss.backward()
            encoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.step()
            decoder_optimizer.zero_grad()
            batch_losses.append(float(loss))
            cur_loss = sum(batch_losses)/len(batch_losses)
            print("Train:", end="\t\t")
            print(f"Batch: {len(batch_losses)}", end="\t")
            print(f"Loss: {round(cur_loss, 4)}", end="\r")
        print()
        encoder_lr_sch.step()
        decoder_lr_sch.step()
        batch_losses = []
        batch_accuracies = []
        for image_batch in valid:
            with torch.no_grad():
                fwd = encoder(image_batch)
                preds = decoder(fwd)
                loss = loss_fn(preds, image_batch)
                batch_losses.append(float(loss))
                cur_loss = sum(batch_losses)/len(batch_losses)
                print("Validation:", end="\t")
                print(f"Batch: {len(batch_losses)}", end="\t")
                print(f"Loss: {round(cur_loss, 4)}", end="\r")
        print()
        if (i+1) % 5 == 0:
            torch.save(encoder, f"saves/encoder_{i+1:03}.pt")
            torch.save(decoder, f"saves/decoder_{i+1:03}.pt")

if __name__ == "__main__":
    main()