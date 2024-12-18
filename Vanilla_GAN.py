import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

# Hyperparameters
latent_dim = 100  # Size of the noise vector
num_epochs = 50  # Number of training epochs
batch_size = 128  # Batch size
sample_interval = 500  # Interval for saving generated images


# Create directory for saving images
output_dir = './generated_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 127.5 - 1.0  # Normalize to [-1, 1]
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension

def build_generator():
    model = Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=latent_dim))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))  # Output layer
    model.add(layers.Reshape((28, 28, 1)))  # Reshape to image format
    return model

def build_discriminator():
    model = Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output probability
    return model

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build the generator
generator = build_generator()

# Create the combined model
z = layers.Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False  # Freeze the discriminator during generator training
validity = discriminator(img)

combined = tf.keras.Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer='adam')


# Training the GAN
for epoch in range(num_epochs):
    # Train Discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))  # We want the generator to fool the discriminator

    # Print losses and save generated images
    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
        save_image(epoch)


def save_image(epoch):
    noise = np.random.normal(0, 1, (25, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(5, 5)
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig(f"{output_dir}/mnist_{epoch}.png")
    plt.close()
