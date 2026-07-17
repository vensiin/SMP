import numpy as np
def predict_sequence(model, initial_sequence, n_steps):
    """
    Predict n_steps into the future using autoregressive prediction.
    Each prediction becomes the input for the next prediction.

    Args:
        model: Trained Keras model
        initial_sequence: Starting sequence, shape (1, num_unrollings, 1)
        n_steps: How many steps to predict ahead

    Returns:
        List of predictions
    """
    predictions = []
    current_sequence = initial_sequence.copy()

    # Loops n amount of times
    for _ in range(n_steps):
        # Predict next value
        next_pred = model.predict(current_sequence, verbose=0)[0, 0]
        predictions.append(next_pred)

        # Slide window: remove oldest, add prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred  # ⬅️ Use own prediction!

    return predictions
