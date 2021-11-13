import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms, datasets, models
from sklearn.model_selection import train_test_split
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
    def __init__(self, input_size):
        super(CustomVGG, self).__init__()
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
        probs = nn.functional.softmax(scores, dim=-1)
        return scores, probs, feature_maps
    
    
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
            preds_scores = model(inputs)[0]
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

        preds_scores = model(inputs)[0]
        preds_class = torch.argmax(preds_scores, dim=-1)

        labels = labels.cpu().numpy()
        preds_class = preds_class.detach().cpu().numpy()

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