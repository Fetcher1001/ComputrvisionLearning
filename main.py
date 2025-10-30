from Requierments import np, tf, plt, sns, confusion_matrix, accuracy_score
import Mnist
import Nerual_Net
from PlotGraph import PlotGraph

def main():
    database = Mnist.MNIST_Dataset()

    x_train, y_train = database.get_train_data()
    x_test, y_test = database.get_test_data()

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    nn = Nerual_Net.NeuralNet()

    epochs = 5
    batch_size= 32

    losses = []
    accuracies = []

    for epoch in range(epochs):
        indices = np.random.permutation(len(x_train))
        X_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        batches = 0

        for i in range(0, len(x_train), batch_size):
            x_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            loss = nn.train_step(x_batch,y_batch)
            epoch_loss += loss
            batches += 1

        avg_loss = epoch_loss/ batches
        losses.append(avg_loss)

        y_pred = nn.predict(x_test)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)

        print(f"Epoch {epoch+1, epochs}  |   Loss: {avg_loss:.4f}    |   Acc: {acc:.4%}")

    y_pred_final = nn.predict(x_test)
    final_acc = accuracy_score(y_test, y_pred_final)
    print(f"\nFinal Test Accuracy: {final_acc:.4%}")

    PlotGraph.plot_training_history(losses, accuracies)

    PlotGraph.plot_confusion(y_test, y_pred_final, normalize=True)

    idx = 0
    probs = nn.forward_prop(x_test[idx:idx+1])
    PlotGraph.show_single(x_test[idx], y_test[idx], y_pred_final[idx], prob = probs.max() *100)
    PlotGraph.plot_probabilities(probs, y_test[idx])

    PlotGraph.show_grid_pred(x_test,y_test, y_pred_final, only_errors=True)

    PlotGraph.show_grid_pred(x_test,y_test,y_pred_final, only_errors= False )


if __name__ == "__main__":
    main()
