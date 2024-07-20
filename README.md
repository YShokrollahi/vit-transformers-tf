
# Vision Transformer (tensorflow package)

This package provides an implementation of the Vision Transformer (ViT) in TensorFlow first version.

## Model Overview

The Vision Transformer (ViT) is a model for image classification that uses a transformer architecture on image patches. This model was introduced in the paper ["AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"](https://arxiv.org/abs/2010.11929).

### Key Components

- **Patches**: Images are split into fixed-size patches.
- **Patch Embedding**: Each patch is linearly embedded.
- **Position Embedding**: Positional information is added to the patch embeddings.
- **Transformer Encoder**: A stack of transformer layers processes the embedded patches.
- **Classification Head**: The output of the transformer encoder is passed to a classification head for final predictions.

### Model Parameters

- **input_shape**: Shape of the input images.
- **patch_size**: Size of the patches extracted from the image.
- **num_patches**: Number of patches per image.
- **projection_dim**: Dimension of the linear projection.
- **transformer_layers**: Number of transformer layers.
- **num_heads**: Number of attention heads in each transformer layer.
- **transformer_units**: List of units in the feed-forward layers within the transformer.
- **mlp_head_units**: List of units in the classification head.
- **num_classes**: Number of output classes.

## Data

For this example, we use the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.

### Data Loading and Preprocessing

- **Normalization**: Images are normalized to the [0, 1] range.
- **One-Hot Encoding**: Labels are converted to one-hot encoding.

## Installation

Clone the repository and install the package using pip:

```bash
git clone https://github.com/YShokrollahi/vit-transformers-tf.git
cd vit-transformers-tf
pip install .
```

## Usage

To train the model, run the example training script:

```python
from vision_transformer.vit import create_vit_classifier
from vision_transformer.utils import load_cifar10_data

input_shape = (32, 32, 3)
patch_size = 4
num_patches = (input_shape[0] // patch_size) ** 2
projection_dim = 64
transformer_layers = 8
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
mlp_head_units = [2048, 1024]
num_classes = 10

(x_train, y_train), (x_test, y_test) = load_cifar10_data()

vit_classifier = create_vit_classifier(
    input_shape=input_shape,
    patch_size=patch_size,
    num_patches=num_patches,
    projection_dim=projection_dim,
    transformer_layers=transformer_layers,
    num_heads=num_heads,
    transformer_units=transformer_units,
    mlp_head_units=mlp_head_units,
    num_classes=num_classes,
)

vit_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

vit_classifier.summary()

history = vit_classifier.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=100,
    validation_split=0.1,
)

test_loss, test_accuracy = vit_classifier.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
```

## Training

The training script loads the dataset, initializes the Vision Transformer model, and trains it for a specified number of epochs. Training progress, including loss and accuracy, will be printed to the console.

## Testing

To run the tests, execute:

```bash
python -m unittest discover tests
```

## Example Usage

You can use the package to train the Vision Transformer model on the CIFAR-10 dataset as shown in the usage section.

## About

This package provides a TensorFlow implementation of the Vision Transformer model for image classification tasks. It includes data loading, model definition, and training scripts for the CIFAR-10 dataset. The repository provides a complete pipeline from preprocessing the data to training and testing the model.

## License

MIT
