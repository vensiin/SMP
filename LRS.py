import tensorflow as tf
import keras

def lr_decay_schedule(epoch, initial_lr, min_lr):
    new_lr = initial_lr * (0.5 ** epoch)
    final_lr = max(new_lr, min_lr)
    print(f"Epoch {epoch + 1}: Learning rate is {final_lr}")
    return final_lr