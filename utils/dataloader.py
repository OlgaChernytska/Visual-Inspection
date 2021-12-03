import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils.constants import GOOD_CLASS_FOLDER, DATASET_SETS, INPUT_IMG_SIZE, ClASSES, IMG_FORMAT


class MVTEC_AD_DATASET(Dataset):
    def __init__(self, root):
        self.classes = ["Good", "Anomaly"]
        self.img_filenames, self.img_labels = self._get_images_and_labels(root)
        self.transform = transforms.Compose(
            [
                transforms.Resize(INPUT_IMG_SIZE),
                transforms.ToTensor(),
            ]
        )

    def _get_images_and_labels(self, root):
        image_names = []
        labels = []

        for folder in DATASET_SETS:
            folder = os.path.join(root, folder)

            for class_folder in os.listdir(folder):
                label = 0 if class_folder == GOOD_CLASS_FOLDER else 1

                class_folder = os.path.join(folder, class_folder)
                class_images = os.listdir(class_folder)
                class_images = [
                    os.path.join(class_folder, image)
                    for image in class_images
                    if image.find(IMG_FORMAT) > -1
                ]

                image_names.extend(class_images)
                labels.extend([label] * len(class_images))

        print(
            "Dataset {}: N Images = {}, Share of anomalies = {:.3f}".format(
                root, len(labels), np.sum(labels) / len(labels)
            )
        )
        return image_names, labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_fn = self.img_filenames[idx]
        label = self.img_labels[idx]

        img = Image.open(img_fn)
        img = self.transform(img)

        label = torch.as_tensor(label, dtype=torch.long)

        return img, label


def get_train_test_loaders(root, batch_size, test_size=0.2, random_state=42):

    dataset = MVTEC_AD_DATASET(root=root)

    train_idx, test_idx = train_test_split(
        np.arange(dataset.__len__()),
        test_size=test_size,
        shuffle=True,
        stratify=dataset.img_labels,
        random_state=random_state,
    )

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True
    )
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False
    )
    return train_loader, test_loader


def get_cv_train_test_loaders(root, batch_size, n_folds=5):

    dataset = MVTEC_AD_DATASET(root=root)

    kf = StratifiedKFold(n_splits=n_folds)
    kf_loader = []

    for train_idx, test_idx in kf.split(np.arange(dataset.__len__()), dataset.targets):
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True
        )
        test_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False
        )

        kf_loader.append((train_loader, test_loader))

    return kf_loader