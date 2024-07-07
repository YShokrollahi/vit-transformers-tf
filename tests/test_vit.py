# tests/test_vit.py

import unittest
from vision_transformer.vit import create_vit_classifier
from vision_transformer.utils import load_cifar10_data

class TestViT(unittest.TestCase):
    def test_model_creation(self):
        input_shape = (32, 32, 3)
        patch_size = 4
        num_patches = (input_shape[0] // patch_size) ** 2
        projection_dim = 64
        transformer_layers = 8
        num_heads = 4
        transformer_units = [projection_dim * 2, projection_dim]
        mlp_head_units = [2048, 1024]
        num_classes = 10

        model = create_vit_classifier(
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

        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
