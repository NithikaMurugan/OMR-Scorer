import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train bubble classifier CNN from prepared dataset')
    parser.add_argument('--data', default='dataset_bubbles.npy')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--out', default=os.path.join('Models', 'cnn_modal.h5'))
    args = parser.parse_args()

    # Ensure project root on path (not strictly necessary here but for consistency)
    THIS_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    data = np.load(args.data, allow_pickle=True).item()
    X = data['X']
    y = data['y']

    # Filter valid labels (>=0)
    mask = y >= 0
    X = X[mask]
    y = y[mask]

    # Train/val split
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * 0.8)
    tr_idx, va_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    model = build_model()
    # Simple augmentation pipeline
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        shear_range=2,
    )
    datagen.fit(X_tr)
    steps = max(1, len(X_tr) // args.batch)
    model.fit(datagen.flow(X_tr, y_tr, batch_size=args.batch),
              validation_data=(X_va, y_va),
              epochs=args.epochs,
              steps_per_epoch=steps)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == '__main__':
    main()
