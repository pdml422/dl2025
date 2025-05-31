import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

VGG_architecture = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512,
                    "M"]


class VGG19(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG19, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_architecture)

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Assumes 224x224 input after 5 pooling layers (224 / 2^5 = 7)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 5


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1] range
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = VGG19(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Starting training on {DEVICE}")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:  # Print progress every 100 mini-batches
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    print(
        f"Epoch {epoch + 1} completed. Average loss: {loss.item():.4f}")  # Note: this is last batch loss, not avg for epoch

model.eval()
correct = 0
total = 0
with torch.no_grad():  # No need to track gradients during testing
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTesting finished.")
print(f"Accuracy of the network on the {total} test images: {accuracy:.2f} %")

# model.eval()
# # Assuming 'test_dataset' is loaded, get a sample image
# image, label = test_dataset[0]
# image_tensor = image.unsqueeze(0).to(DEVICE) # Add batch dimension and send to device
# with torch.no_grad():
#     output = model(image_tensor)
#     _, predicted_idx = torch.max(output, 1)
# class_names = train_dataset.classes # ['airplane', 'automobile', ..., 'truck'] for CIFAR10
# print(f"Sample image true label: {class_names[label]}, Predicted class: {class_names[predicted_idx.item()]}")