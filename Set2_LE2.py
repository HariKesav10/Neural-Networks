import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# 1. Load and preprocess the Fashion-MNIST dataset
def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Reshape and normalize the input data
    X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    X_test = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_fashion_mnist()

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 2. Build a basic CNN architecture (conv filter 3 x 3 with ReLU activation, max pooling with filter size 2x2, and a fully connected layer with 64 neurons)
def create_base_model():
    model = Sequential([
        Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)),
        MaxPooling2D(),
        Conv2D(64,(3,3),activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(64,activation="relu"),
        Dense(10,activation="softmax")
    ])
    return model

# 3. Compile and train the model
base_model = create_base_model()
print("\nBase Model Summary:")
base_model.summary()

# Fill your code here to compile and train the model (use necessary hyperparameters)
base_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

history = base_model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val))

# 4. Evaluate the model's performance
train_loss, train_accuracy = base_model.evaluate(X_train, y_train)
val_loss, val_accuracy = base_model.evaluate(X_val, y_val)
test_loss, test_accuracy = base_model.evaluate(X_test, y_test)

print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accurcay")
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()

plot_history(history)

# 5. Optimize the model (conv filter 3 x 3 with ReLU activation, batch normalization, max pooling with filter size 2x2, and a fully connected layer with 128 neurons)
def create_optimized_model():
    model = Sequential([
        Conv2D(32,(3,3), activation="relu", input_shape=(28,28,1)),
        BatchNormalization(),
        Conv2D(32,(3,3),activation="relu"),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Conv2D(64,(3,3),activation="relu"),
        Conv2D(64,(3,3),activation="relu"),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Flatten(),
        Dense(128,activation="relu"),
        Dropout(0.5),
        Dense(64,activation="relu"),
        Dropout(0.5),
        Dense(10,activation="softmax")
    ])
    return model

optimized_model = create_optimized_model()
print("\nOptimized Model Summary:")
optimized_model.summary()

optimized_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

optimized_history = optimized_model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_val, y_val))

# 6. Compare the performance of the optimized model
opt_train_loss, opt_train_accuracy = optimized_model.evaluate(X_train, y_train)
opt_val_loss, opt_val_accuracy = optimized_model.evaluate(X_val, y_val)
opt_test_loss, opt_test_accuracy = optimized_model.evaluate(X_test, y_test)

print("\nOptimized Model Results:")
print(f"Train accuracy: {opt_train_accuracy:.4f}")
print(f"Validation accuracy: {opt_val_accuracy:.4f}")
print(f"Test accuracy: {opt_test_accuracy:.4f}")

plot_history(optimized_history)

# Compare base and optimized model performance
print("\nPerformance Comparison:")
print(f"Base Model Test Accuracy: {test_accuracy:.4f}")
print(f"Optimized Model Test Accuracy: {opt_test_accuracy:.4f}")
print(f"Improvement: {(opt_test_accuracy - test_accuracy) * 100:.2f}%")

# Function to plot sample images
def plot_sample_images(X, y, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Class: {np.argmax(y[i])}")
        plt.axis('off')
    plt.show()

# Plot some sample images
plot_sample_images(X_test, y_test)