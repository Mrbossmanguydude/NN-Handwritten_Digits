import numpy as np
import pygame

pygame.init()

# Define the Layer class
class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.randn(num_inputs, num_outputs) * np.sqrt(2. / num_inputs)  # He initialization for ReLU
        self.biases = np.zeros(num_outputs)
        self.activations = []

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.activations = self.relu(np.dot(self.inputs, self.weights) + self.biases)
        #self.activations = self.tanh(np.dot(self.inputs, self.weights) + self.biases)
        return self.activations

    def backward(self, gradient_loss_output):
        gradient_loss_input = np.multiply(gradient_loss_output, self.d_relu(self.activations))
        #gradient_loss_input = np.multiply(gradient_loss_output, self.d_tanh(self.activations))

        gradient_input_weights = self.inputs
        gradient_input_inputs = self.weights

        gradient_loss_weights = np.dot(gradient_input_weights.T, gradient_loss_input)
        gradient_loss_bias = np.sum(gradient_loss_input, axis=0)
        gradient_loss_inputs = np.dot(gradient_loss_input, gradient_input_inputs.T)

        self.gradients = {"weights": gradient_loss_weights, "biases": gradient_loss_bias}

        return gradient_loss_inputs

    def update_weights(self, gradients, learning_rate):
        self.weights -= learning_rate * gradients["weights"]
        self.biases -= learning_rate * gradients["biases"]

    def relu(self, x):
        return np.maximum(0, x)
    
    def d_relu(self, x):
        return (x > 0).astype(float)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def d_tanh(self, x):
        return 1 - np.square(np.tanh(x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        sig = self.sigmoid(x)  # Reuse the sigmoid function
        return sig * (1 - sig)
    
# Define the Neural_Network class
class Neural_Network:
    def __init__(self, inputs, len_layers, num_outputs):
        self.inputs = inputs
        self.len_layers = len_layers
        self.num_outputs = num_outputs
        self.layers = [Layer(784, len_layers), Layer(len_layers, len_layers - 50), Layer(len_layers - 50, 64), Layer(64, 16), Layer(16, num_outputs)]

    def forward_pass(self, batch_images):
        batch_images = batch_images.reshape(batch_images.shape[0], -1)
        inputs = batch_images
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward_pass(self, labels):
        nn_predictions = self.layers[-1].activations
        gradient_loss_output = nn_predictions - labels
        for layer in reversed(self.layers):
            gradient_loss_output = layer.backward(gradient_loss_output)

    def cost(self, nn_predictions, labels):
        return -np.sum(labels * np.log(nn_predictions + 1e-7))
    
    def train(self, train_images, train_labels, num_epochs, batch_size, learning_rate):
        num_samples = len(train_images)
        num_batches = num_samples // batch_size

        for epoch in range(num_epochs):
            correct_predictions = 0

            for i in range(num_batches):
                batch_images = train_images[i*batch_size:(i+1)*batch_size]
                batch_labels = train_labels[i*batch_size:(i+1)*batch_size]
                
                predictions = self.forward_pass(batch_images)

                loss = self.cost(predictions, batch_labels)

                self.backward_pass(batch_labels)
                for layer in self.layers:
                    layer.update_weights(layer.gradients, learning_rate)
                    
                correct_predictions += np.sum(np.argmax(predictions, axis=1) == np.argmax(batch_labels, axis=1))
            accuracy = correct_predictions / num_samples
            print(f'Epoch {epoch+1}/{num_epochs}, Accuracy = {accuracy}')
        print(f"Accuracy = {accuracy}, Epochs = {num_epochs}")

# Function to load data
def get_data():
    with np.load('mnist.npz') as file:
        images, labels = file['x_train'], file['y_train']

    images = images.astype('float32') / 255
    labels = np.eye(10)[labels]

    print("Unique label values:", np.unique(labels))
    print("Label data type:", labels.dtype)

    return images, labels

def calculate_percentages(activations):
    non_zero_mask = activations != 0
    exp_activations = np.exp(activations[non_zero_mask])
    probabilities = exp_activations / np.sum(exp_activations)
    percentages = probabilities
    full_percentages = np.zeros_like(activations)
    full_percentages[non_zero_mask] = percentages
    return full_percentages

# Function to initialize Pygame and run the main loop
def main():
    pygame.init()

    # Set up some constants
    WIDTH, HEIGHT = 560, 560
    ROWS, COLS = 28, 28
    SQUARE_SIZE = WIDTH // COLS

    # Create a 2D numpy array to store the pixel data
    pixels = np.zeros((ROWS, COLS))

    # Create the Pygame window
    win = pygame.display.set_mode((WIDTH, HEIGHT))

    state = "Drawing"

    # Function to draw the grid
    def draw_window():
        win.fill((255, 255, 255))
        for j in range(ROWS):
            for i in range(COLS):
                pygame.draw.rect(win, (pixels[i, j], pixels[i, j], pixels[i, j]), (j*SQUARE_SIZE, i*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        pygame.display.update()

    agent = Neural_Network([], 128, 10)
    images, labels = get_data()
    agent.train(images, labels, 5, 100, 0.01)

    # Main loop
    run = True
    drawing = False  # Variable to track if the left mouse button is being held down
    while run:
        draw_window()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            try:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and state == "Drawing":  # Left mouse button
                        drawing = True
                    elif event.button == 3 and state == "Drawing":  # Right mouse button
                        x, y = pygame.mouse.get_pos()
                        i, j = y // SQUARE_SIZE, x // SQUARE_SIZE
                        pixels[i, j] = 0  # Erase the pixel
                elif event.type == pygame.MOUSEBUTTONUP and state == "Drawing":
                    if event.button == 1:  # Left mouse button
                        drawing = False
                elif event.type == pygame.MOUSEMOTION and state == "Drawing":
                    if drawing:
                        x, y = pygame.mouse.get_pos()
                        i, j = y // SQUARE_SIZE, x // SQUARE_SIZE
                        pixels[i, j] = 255  # Draw the pixel
                elif event.type == pygame.KEYDOWN:
                    if state == "Drawing":
                        if event.key == pygame.K_SPACE:
                            user_image = pixels.reshape(1, 784) / 255

                            prediction = agent.forward_pass(user_image)
                            print(f'Predicted number: {np.argmax(prediction)}')
                            print(prediction)
                            percentages = list(calculate_percentages(prediction[0]))
                            for percent in range(len(percentages)):
                                print(f"Number = {percent}, Probability = {percentages[percent]*100}")
                            
                        elif event.key == pygame.K_r:  # Reset the board when 'R' is pressed
                            pixels.fill(0)
            except IndexError:
                pass
    pygame.quit()

if __name__ == "__main__":
    main()