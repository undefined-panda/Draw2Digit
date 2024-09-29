from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
import torch
import torch.optim as optim

from model import MNIST_Net

current_dir = os.path.dirname(__file__)
root_path = os.path.join(current_dir, "root")

train_data = datasets.MNIST(root=root_path, train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root=root_path, train=False, download=True, transform=ToTensor())

def save_model(model, file_name):
    model_path = os.path.join(current_dir, "saved_models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    torch.save(model.state_dict(), f"{model_path}/{file_name}.pth")
    print(f"Model saved in '{model_path}'.")

def test(model, device, loss_func, test_data):
    model.to(device)
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for sample, label in test_data:
            sample, label = sample.to(device), label.to(device)
            output = model(sample)
            test_loss += loss_func(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item() # hab ich nicht ganz verstanden
        
        test_loss = test_loss / len(test_data.dataset)
        print(f"Test Error: \n Accuracy: {correct}/{len(test_data.dataset)} ({100. * correct / len(test_data.dataset):.0f}%), Avg Loss: {test_loss:.4f}")

def fit(epochs, model, loss_func, opt, train_data, test_data=None, testing=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    checkpoint = len(train_data) // 5

    loss_per_epoch = []
    accuracy_per_epoch = []

    model.train()
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}\n"+35*'-')
        accuracy = 0
        for batch_idx, (sample, label) in enumerate(train_data):
            sample, label = sample.to(device), label.to(device)
            output = model(sample)
            loss = loss_func(output, label)                
            loss.backward()
            opt.step()
            opt.zero_grad()

            if batch_idx % checkpoint == 0:
                batch_part, batch_number, percentage = batch_idx * len(sample), len(train_data.dataset), 100. * batch_idx / len(train_data)
                print(f"Loss: {loss.item():.6f} [{batch_part}/{batch_number} ({percentage:.0f}%)]")
        
        loss_per_epoch.append(loss.item())
        accuracy_per_epoch.append(accuracy)
    
        if testing:
            test(model, device, loss_func, test_data)
    
    print("Done!")
    return loss_per_epoch, accuracy_per_epoch

if __name__ == "__main__":
    bs = 10

    loaders = {
        "train":DataLoader(train_data, batch_size=bs, shuffle=True), 
        "test":DataLoader(test_data, batch_size=bs, shuffle=True)
    }

    model = MNIST_Net()

    lr = 0.01
    epochs = 2

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = F.cross_entropy
    nn_loss, nn_accuracy = fit(epochs, model, loss_fn, optimizer, loaders["train"], loaders["test"], testing=True)

    save_model(model, file_name="digit_recognition_model_1")