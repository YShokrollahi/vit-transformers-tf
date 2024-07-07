# examples/train_vit.py

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
