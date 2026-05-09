import tensorflow as tf
import keras


class LearningRateScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, schedule, min_lr):
        self.schedule = schedule
        self.min_lr = min_lr

    # Call makes the entire class a function
    def __call__(self, step):
        lr = self.schedule(step)
        return tf.maximum(lr, self.min_lr)