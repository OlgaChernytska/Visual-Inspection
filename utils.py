import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms, datasets, models
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score


def get_train_test_loaders(
    root, batch_size, img_size, test_size=0.2, random_state=42
):
    
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(root=root, transform=transform)

    train_idx, test_idx = train_test_split(
        np.arange(dataset.__len__()),
        test_size=test_size,
        shuffle=True,
        stratify=dataset.targets,
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


class CustomVGG(nn.Module):
    """
    Returns class scores when in train mode.
    Returns class probs and feature maps when in eval mode.
    """

    def __init__(self, input_size):
        super(CustomVGG, self).__init__()
        self.input_size = input_size
        self.feature_extractor = models.vgg16(pretrained=True).features[:-1]
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(
                kernel_size=(input_size[0] // 2 ** 5, input_size[1] // 2 ** 5)
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=self.feature_extractor[-2].out_channels, out_features=2
            ),
        )
        self._freeze_params()

    def _freeze_params(self):
        for param in self.feature_extractor[:23].parameters():
            param.requires_grad = False

    def forward(self, x):
        feature_maps = self.feature_extractor(x)
        scores = self.classification_head(feature_maps)

        if self.training:
            return scores

        else:
            probs = nn.functional.softmax(scores, dim=-1)

            weights = self.classification_head[3].weight
            weights = (
                weights.unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(0)
                .repeat((x.size(0), 1, 1, 14, 14))
            )
            feature_maps = feature_maps.unsqueeze(1).repeat((1, 2, 1, 1, 1))
            location = torch.mul(weights, feature_maps).sum(axis=2)
            location = F.interpolate(location, size=self.input_size, mode="bilinear")

            return probs, location
    
    
def train(dataloader, model, optimizer, criterion, epochs, device):
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}:", end=" ")
        running_loss = 0
        running_corrects = 0
        n_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds_scores = model(inputs)
            preds_class = torch.argmax(preds_scores, dim=-1)
            loss = criterion(preds_scores, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds_class == labels)
            n_samples += inputs.size(0)

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double() / n_samples
        print("Loss = {:.4f}, Accuracy = {:.4f}".format(epoch_loss, epoch_acc))

    return model


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    class_names = dataloader.dataset.classes
    
    running_corrects = 0
    y_true = np.empty(shape=(0,))
    y_pred = np.empty(shape=(0,))

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds_probs = model(inputs)[0]
        preds_class = torch.argmax(preds_probs, dim=-1)

        labels = labels.to("cpu").numpy()
        preds_class = preds_class.detach().to("cpu").numpy()

        y_true = np.concatenate((y_true, labels))
        y_pred = np.concatenate((y_pred, preds_class))

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))

    print()
    plot_confusion_matrix(y_true, y_pred, class_names=class_names)


def plot_confusion_matrix(y_true, y_pred, class_names="auto"):
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=[5, 5])
    sns.heatmap(
        confusion,
        annot=True,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.title("Confusion Matrix")
    plt.show()
    
    
def get_bbox_from_heatmap(heatmap, thres):
    """Return upper left and lower right corner locations."""
    binary_map = heatmap > thres

    if binary_map.sum() == 0:
        return 0, 0, 0, 0

    x_dim = np.max(binary_map, axis=0) * np.arange(0, binary_map.shape[1])
    x_0 = int(x_dim[x_dim > 0].min())
    x_1 = int(x_dim.max())

    y_dim = np.max(binary_map, axis=1) * np.arange(0, binary_map.shape[0])
    y_0 = int(y_dim[y_dim > 0].min())
    y_1 = int(y_dim.max())

    return x_0, y_0, x_1, y_1


def predict_localize(model, dataloader, device, thres, neg_class, n_samples=9):
    model.to(device)
    model.eval()

    class_names = dataloader.dataset.classes
    transform_to_PIL = transforms.ToPILImage()

    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))
    plt.figure(figsize=[n_cols * 5, n_rows * 5])

    counter = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        out = model(inputs)
        probs, class_preds = torch.max(out[0], dim=-1)
        feature_maps = out[1].to("cpu")

        for img_i in range(inputs.size(0)):
            img = transform_to_PIL(inputs[img_i])
            class_pred = class_preds[img_i]
            prob = probs[img_i]
            label = labels[img_i]
            heatmap = feature_maps[img_i][neg_class].detach().numpy()

            counter += 1
            plt.subplot(n_rows, n_cols, counter)
            plt.imshow(img)
            plt.axis("off")
            plt.title(
                "Predicted: {}, Prob: {:.3f}, True Label: {}".format(
                    class_names[class_pred], prob, class_names[label]
                )
            )

            if class_pred == neg_class:
                x_0, y_0, x_1, y_1 = get_bbox_from_heatmap(heatmap, thres)
                rectangle = Rectangle(
                    (x_0, y_0),
                    x_1 - x_0,
                    y_1 - y_0,
                    edgecolor="red",
                    facecolor="none",
                    lw=3,
                )
                plt.gca().add_patch(rectangle)

                plt.imshow(heatmap, cmap="Reds", alpha=0.3)

            if counter == n_samples:
                plt.tight_layout()
                plt.show()
                return
            
            
def get_cv_train_test_loaders(root, batch_size, img_size, n_folds=5):

    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(root=root, transform=transform)

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