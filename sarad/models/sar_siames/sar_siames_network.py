import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from sarad.models.sar_siames.SARPatchPairsDataset import SARPatchPairsDataset


def l2_norm(t):
    return tf.math.l2_normalize(t, axis=1)

def build_base_cnn(input_shape=(128, 128, 2), embedding_dim=64):
    """
    Simple CNN to embed SAR patches into a vector.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(embedding_dim)(x)  # embeddings without activation
    # Normalize embeddings to unit vectors
    outputs = layers.Lambda(l2_norm, output_shape=(embedding_dim,))(x)

    model = models.Model(inputs, outputs, name="base_cnn")
    return model

def euclidean_distance(vectors):
    """
    Compute Euclidean distance between two vectors.
    """
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, 1e-10))  # add epsilon for numerical stability

def build_siamese_network(input_shape=(128, 128, 2), embedding_dim=64):
    """
    Build the full Siamese network that outputs the distance between two patch embeddings.
    """
    base_cnn = build_base_cnn(input_shape, embedding_dim)

    input_a = layers.Input(shape=input_shape, name="patch1")
    input_b = layers.Input(shape=input_shape, name="patch2")

    embed_a = base_cnn(input_a)
    embed_b = base_cnn(input_b)

    distance = layers.Lambda(euclidean_distance, name="distance",
                         output_shape=(1,))([embed_a, embed_b])


    model = models.Model(inputs=[input_a, input_b], outputs=distance, name="siamese_network")
    return model


class ContrastiveLoss(losses.Loss):
    """
    Contrastive loss as defined in Hadsell-et-al.'06
    L = (1 - label) * 0.5 * D^2 + (label) * 0.5 * max(0, margin - D)^2
    where label=0 for similar pairs, label=1 for dissimilar pairs.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(self.margin - y_pred, 0))
        loss = tf.reduce_mean((1 - y_true) * 0.5 * square_pred + y_true * 0.5 * margin_square)
        return loss


# dataset returns tuples like: (p1, p2, labels, p1_names, p2_names)
def map_and_set_shapes(p1, p2, labels, p1n, p2n):
    p1.set_shape([args.batch_size, 128, 128, 2])
    p2.set_shape([args.batch_size, 128, 128, 2])
    labels.set_shape([args.batch_size])
    return (p1, p2), labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_model_path", type=str, default=None,
                    help="Path to save the trained model.")
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()
        # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), "INFO"),
        format="%(asctime)s — %(levelname)s — %(message)s"
    )

    # Load train dataset
    dataset = SARPatchPairsDataset(
        dataset_folder=args.dataset_folder,
        transform=None,
        num_pairs=args.num_pairs,
        batch_size=args.batch_size
    )

    logging.info(f"Total dataset size: {len(dataset.pairs)} pairs, ~{len(dataset.pairs) // args.batch_size} batches")

    full_dataset = dataset.get_dataset(shuffle=True, cache=False)
    num_batches = len(dataset.pairs) // args.batch_size
    val_dataset = full_dataset.take(num_batches // 5)  # 20% validation
    train_dataset = full_dataset.skip(num_batches // 5)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset_for_fit = train_dataset.map(map_and_set_shapes, num_parallel_calls=AUTOTUNE)
    val_dataset_for_fit = val_dataset.map(map_and_set_shapes, num_parallel_calls=AUTOTUNE)
    train_dataset_for_fit = train_dataset_for_fit.prefetch(AUTOTUNE)
    val_dataset_for_fit = val_dataset_for_fit.prefetch(AUTOTUNE)

    # training
    input_shape = (128, 128, 2)
    model = build_siamese_network(input_shape=input_shape)
    model.summary()

    model.compile(optimizer=optimizers.Adam(1e-3), loss=ContrastiveLoss(margin=1.0))
    model.fit(train_dataset_for_fit, epochs=20, validation_data=val_dataset_for_fit)
    
    if args.save_model_path:
        logging.info(f"Saving Siamese model to: {args.save_model_path}")
        model.save(args.save_model_path)

        # Save the base CNN separately for embedding generation
        base_cnn = model.get_layer("base_cnn")
        base_cnn.save(args.save_model_path.replace(".keras", "_base.keras"))
        logging.info(f"Saved base CNN to: {args.save_model_path.replace('.keras', '_base.keras')}")
