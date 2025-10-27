from Requierments import *


class PlotGraph:
    def __init__(self):
        plt.figure(figsize=(6,6))
        sns.set_style("whitegrid")


    @staticmethod
    def show_single(image, true_label, pred_label = None, prob = None):
        plt.figure(figsize=(6,6))
        plt.imshow(image.reshape(28,28), cmap='gray')
        title = f"True: {true_label}"
        color = 'black'
        if pred_label is not None:
            correct = true_label == pred_label
            color = 'green' if correct else 'red'
            title += f"     | Prediction = {pred_label}"
            if prob is not None:
                title += f"{prob:.1f}%"
            plt.title(title, color= color, fontsize= 16, pad= 20)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def show_grid_pred(images, true_labels, pred_labels, cols=5, rows=3, only_errors = False):
        num_pics = cols * rows
        if only_errors:
            errors_idx = np.where(true_labels != pred_labels)[0]
            if len(errors_idx)  == 0:
                print("No errors to show")
                return
            idx = errors_idx[:num_pics]
            title = f"Error Classifications({len(errors_idx)})"
        else:
            idx = np.random.choice(len(images), num_pics, replace = False)
            title = "Random prediction"

        plt.figure(figsize=(12,8))
        for i,idx_i in enumerate(idx):
            plt.subplot(rows, cols, i+1)
            plt.imshow(images[idx_i].reshape(28,28), cmap = 'gray')
            correct = true_labels[idx_i]
            pred = pred_labels[idx_i]
            color = 'green' if correct == pred else 'red'
            plt.title(f"True: {correct} to prediction: {pred}",color= color, fontsize= 12)
            plt.axis('off')
        plt.suptitle(title, fontsize = 16, y= 0.95)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion(true_labels, pred_labels,normalize = False):
        cm = confusion_matrix(true_labels, pred_labels)
        if normalize:
            cm = cm.astype('float')/ cm.sum(axis =1)[:,np.newaxis]
            fmt = '.2f'
            title = 'Confusion Matrix( normalized)'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',xticklabels=range(10), yticklabels= range(10))
        plt.title(title, fontsize= 16)
        plt.ylabel('true label')
        plt.xlabel('Prediction Label')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_training_history(losses, accuracies = None):
        epochs = range(1, len(losses) +1)

        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.plot(epochs, losses, 'b-o', label = 'loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.grid(True)

        if accuracies:
            plt.subplot(1,2,2)
            plt.plot(epochs,accuracies, 'g-o', label = 'Accuracy')
            plt.title('Training Accuaracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_probailities(probabilities, true_label, top_k=5):
        probs = probabilities[0]
        pred_label = np.argmax(probs)

        top_idx = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_idx]
        top_labels = top_idx

        plt.figure(figsize=(8,5))
        bars = plt.bar(range(top_k), top_probs, color='skyblue', edgecolor= 'black')

        if true_label in top_labels:
            idx = list(top_labels).index(true_label)
            bars[idx].set_color('green')
        if pred_label in top_labels:
            idx = list(top_labels).index(pred_label)
            bars[idx].set_edgecolor('red')
            bars[idx].set_linewidth(2)

        plt.xticks(range(top_k), top_labels)
        plt.ylabel('Probabilities')
        plt.title(f'Prediction: {pred_label}    | Truth: {true_label}')
        plt.ylim(0,1)

        for i,v in enumerate(top_probs):
            plt.text(i,v+0.02, f'{v:.3f}', ha= 'center')

        plt.tight_layout()
        plt.show()
