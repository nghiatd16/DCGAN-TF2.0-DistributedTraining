import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import model
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import os

tf.config.set_soft_device_placement(True)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy()

# Load data
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

train_images = (train_images-127.5)/127.5 # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
# Model

# Distributed training
with mirrored_strategy.scope():
    generator = model.build_generator()
    discriminator = model.build_discriminator()
    print(generator.summary())
    print(discriminator.summary())


    # Define loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    def step_fn(inputs):
        noise = tf.random.normal([inputs.shape[0], noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(inputs, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return gen_loss, disc_loss
    gen_loss, disc_loss = mirrored_strategy.experimental_run_v2(
        step_fn, args=(images,))
    gen_mean_loss = mirrored_strategy.reduce(
        tf.distribute.ReduceOp.MEAN, gen_loss, axis=0)
    disc_mean_loss = mirrored_strategy.reduce(
        tf.distribute.ReduceOp.MEAN, disc_loss, axis=0)
    return gen_mean_loss, disc_mean_loss

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False).numpy()

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        sample_img = (predictions[i] * 127.5 + 127.5).astype(np.uint8)
        # sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        plt.imshow(np.squeeze(sample_img), cmap='gray')
        plt.axis('off')

    plt.savefig('samples/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()

def train(dataset, epochs):
    with mirrored_strategy.scope():
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

                # Produce images for the GIF as we go

                

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            generate_and_save_images(generator,
                                        epoch + 1,
                                        seed)
        # Generate after the final epoch
        generate_and_save_images(generator,
                                epochs,
                                seed)
if __name__ == "__main__":
    train(train_dataset, EPOCHS)