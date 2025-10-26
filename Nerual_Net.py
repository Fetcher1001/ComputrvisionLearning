class NeuralNet:
    def __init__(self, input_size = 784, hidden_size = 28, output_size = 10):
        
        # weights and bias layer one
        self.W1 = np.random.randn(input_size, hidden_size) *np.sqrt(2.0/ input_size)
        self.B1 = np.zeros(1, hidden_size)

        # weights and bias layer one
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/ hidden_size)
        self.B2 = np.zeros(1, output_size)

        # Layer 1 
        self.h1 = None
        self.out1 = None

        #layer 2 
        self.h2 = None
        self.out2 = None

        self.learning_rate = 0.01

    def relu(self, layer):
        return np.maximum(0, layer)

    def relu_deriv(self, layer):
        return (layer > 0).astype(float)

    def softmax(self, layer):
        exp_layer = np.exp(layer- np.max(layer, axis=1, keepdims=True))
        return exp_layer/ np.sum(exp_layer,axis=1,keepdims=True)

    def foward_prop(self):
        self.h1 = (self.W1 @ self.W2) + self.B1
        self.out1 = self.relu(self.d1)

        self.h2 = (self.h1 @ self.W2) + self.B2
        self.out2 = self.softmax(self.h2)

        return self.out2
    
    def loss(self, prediction, truth):
        m = truth.shape[0]  # num sample

        one_hot = np.eye(10)[truth] # reshaping label to matrix 

        loss = -np.sum(one_hot * np.log(prediction+ 1e-8)) /m # loss funktion 1e-8 prventing -infinety in log 

        return loss
    

    def backprob(self, input_l, prediction, truth):
        m = input.shape[0]
        one_hot = np.eye(10)[truth]

        #output layer

        delta_a2 = prediction - one_hot
        delta_W2 = (self.out1.T @ delta_a2) / m # .T transposition of matrix 
        delta_B2 = np.sum(delta_a2,axis=0, keepdims=0)

        #hidden layer

        delta_a1 = delta_a2 @ self.W2.T
        loss_grad = delta_a1 * self.relu_deriv(self.h1)
        delta_W1 = (input_l.T @ loss_grad) / m
        delta_B1 = np.sum(loss_grad, axis=0, keepdims= True) / m

        #Update weights 

        self.W2 -= self.learning_rate * delta_W2
        self.B2 -= self.learning_rate * delta_B2
        self.W1 -= self.learning_rate * delta_W1
        self.B1 -= self.learning_rate * delta_B1

    def train_step(self, input, label):
        pred = self.foward_prop(input)
        loss = self.loss(pred, label)
        self.backprob(input, pred, label)
        return loss
    

    def predict(self, input):
        pred = self.foward(input)
        return np.argmax(pred, axis= 1)



