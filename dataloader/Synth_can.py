import os
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Synth_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train, test_size=0.1, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        # Read the file with image locations and labels
        labels_file = os.path.join(root, "train.txt")
        with open(labels_file, "r") as fp:
            content = fp.readlines()

        # Create a list of tuples containing image paths and labels
        self.mapping = list(map(lambda x: (x.split()[0], int(x.split()[1])), [c.strip() for c in content]))

        # Split the data into train and test sets
        if self.train:
            self.mapping, _ = train_test_split(self.mapping, test_size=test_size, random_state=42)
        else:
            _, self.mapping = train_test_split(self.mapping, test_size=test_size, random_state=42)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image_path, label = self.mapping[idx]
        image_path = os.path.join(self.root, image_path)
        image = self.transform(Image.open(image_path).convert('RGB'))

        return image, label


if __name__ == '__main__':
    root_dir = 'data/Synthetic_Can_Data'
    batch_size = 10

    # Define the transformations
    composed_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    # Create train and test datasets
    train_dataset = Synth_Dataset(root=root_dir, train=True, test_size=0.1, transform=composed_transform)
    test_dataset = Synth_Dataset(root=root_dir, train=False, test_size=0.1, transform=composed_transform)

    print('Size of train dataset: %d' % len(train_dataset))
    print('Size of test dataset: %d' % len(test_dataset))

    # Create data loaders for train and test datasets
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    def imshow(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()


        # Retrieve a batch of data using an iterator
    for train_images, train_labels in train_loader:
        break

    print("Train images", train_images)
    print("Train labels", train_labels)

    # Display the batch of train images
    imshow(torchvision.utils.make_grid(train_images))

    # Retrieve a batch of test data
    for test_images, test_labels in test_loader:
        break

    print("Test images", test_labels)
    imshow(torchvision.utils.make_grid(test_images))