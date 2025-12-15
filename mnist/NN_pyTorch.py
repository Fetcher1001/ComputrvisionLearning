import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PlotGraph import PlotGraph
from sklearn.metrics import confusion_matrix, accuracy_score


class NeuralNet_Pytorch(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 128, output_size = 10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu =nn.ReLU
        self.layer2 = (hidden_size, output_size)

    def forward(self, input_1):
        out = self.layer1(input_1)
        out = self.relu(out)
        out = self.layer2(out)
        return out


def main_test():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform= transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    nn_model = NeuralNet_Pytorch()
    optimizer = optim.SGD(nn_model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    epochs =12
    losses = []
    accuracies = []

    for epoch in range(epochs):
        nn_model.train()
        epoch_loss = 0
        batches = 0

        for x_batch, y_batch in train_loader:
            preds = nn_model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        losses.append(avg_loss)


        nn_model.eval()

        with torch.no_grad():
            y_pred = []
            y_test = []

            for x_test_batch, y_test_batch in test_loader:

                preds = nn_model(x_test_batch)
                y_pred.extend(torch.argmax(preds, dim = 1).cpu().numpy())
                y_test.extend(y_test_batch.cpu().numpy())

        acc = np.mean(np.array(y_pred) == np.array(y_test))
        accuracies.append(acc)

        print(f"Epoch {epoch+1}/{epochs}  |   Loss: {avg_loss:.4f}    |   Acc: {acc:.4%}")


    final_acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy: {final_acc:.4%}")

    PlotGraph.plot_training_history(losses, accuracies)

    PlotGraph.plot_confusion(y_test, y_pred, normalize= True)

    idx = np.random.randint(0, 10000)

    x_single = test_dataset[idx][0]
    y_true = test_dataset[idx][1]

    with torch.no_grad():
        probs = torch.softmax(nn_model(x_single.unsqueeze(0)),dim = 1).cpu().numpy()
        pred_lable = y_pred[idx]

        PlotGraph.show_single(test_dataset[idx][0].cpu().numpy(), y_true, pred_lable, probs.max() * 100)
        PlotGraph.plot_probabilities(probs, y_true)

        x_test_np = np.concatenate([batch[0].cpu().numpy() for batch in test_loader])
        PlotGraph.show_grid_pred(x_test_np, y_test, y_pred, only_errors= True)
        PlotGraph.show_grid_pred(x_test_np, y_test, y_pred, only_errors= False)



main_test()
