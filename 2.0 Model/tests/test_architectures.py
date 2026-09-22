"""Tests for architecture instantiation and forward pass shapes."""

import pytest
import torch

# Ensure all architectures are registered
import skynet.architectures.mlp  # noqa: F401
import skynet.architectures.cnn  # noqa: F401
import skynet.architectures.transformer  # noqa: F401
import skynet.architectures.autoencoder  # noqa: F401

from skynet.architectures.registry import (
    get_architecture,
    list_architectures,
    get_architecture_class,
)


class TestRegistry:
    """Tests for the architecture registry."""

    def test_list_architectures(self):
        archs = list_architectures()
        assert isinstance(archs, list)
        assert "mlp" in archs
        assert "cnn" in archs
        assert "transformer" in archs
        assert "autoencoder" in archs

    def test_get_nonexistent_architecture(self):
        with pytest.raises(KeyError, match="not found"):
            get_architecture("nonexistent_model")

    def test_case_insensitive_lookup(self):
        model = get_architecture("MLP", input_dim=10, output_dim=2)
        assert model is not None


class TestMLP:
    """Tests for the MLP architecture."""

    def test_forward_shape(self):
        model = get_architecture("mlp", input_dim=20, output_dim=5)
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 5)

    def test_custom_hidden_dims(self):
        model = get_architecture(
            "mlp", input_dim=10, output_dim=3, hidden_dims=[64, 32, 16]
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_with_dropout_and_batchnorm(self):
        model = get_architecture(
            "mlp", input_dim=10, output_dim=2, dropout=0.5, batch_norm=True
        )
        model.train()
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 2)

    def test_different_activations(self):
        for act in ["relu", "gelu", "silu"]:
            model = get_architecture(
                "mlp", input_dim=10, output_dim=2, activation=act
            )
            x = torch.randn(4, 10)
            out = model(x)
            assert out.shape == (4, 2)


class TestCNN:
    """Tests for the CNN architecture."""

    def test_forward_shape(self):
        model = get_architecture("cnn", in_channels=3, num_classes=10)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4, 10)

    def test_grayscale_input(self):
        model = get_architecture("cnn", in_channels=1, num_classes=5)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        assert out.shape == (2, 5)

    def test_custom_conv_channels(self):
        model = get_architecture(
            "cnn", in_channels=3, num_classes=3, conv_channels=[16, 32]
        )
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 3)


class TestTransformer:
    """Tests for the Transformer architecture."""

    def test_forward_shape(self):
        model = get_architecture(
            "transformer",
            vocab_size=1000,
            d_model=64,
            nhead=4,
            num_layers=2,
            num_classes=3,
        )
        x = torch.randint(0, 1000, (4, 32))
        out = model(x)
        assert out.shape == (4, 3)

    def test_with_padding_mask(self):
        model = get_architecture(
            "transformer",
            vocab_size=500,
            d_model=32,
            nhead=2,
            num_layers=1,
            num_classes=2,
        )
        x = torch.randint(0, 500, (4, 16))
        mask = torch.zeros(4, 16, dtype=torch.bool)
        mask[:, 10:] = True  # Last 6 positions are padding
        out = model(x, src_key_padding_mask=mask)
        assert out.shape == (4, 2)


class TestAutoencoder:
    """Tests for the Autoencoder architecture."""

    def test_forward_shape(self):
        model = get_architecture(
            "autoencoder", input_dim=50, latent_dim=8
        )
        x = torch.randn(4, 50)
        out = model(x)
        assert out.shape == (4, 50)

    def test_encode_shape(self):
        model = get_architecture(
            "autoencoder", input_dim=100, latent_dim=16
        )
        x = torch.randn(4, 100)
        z = model.encode(x)
        assert z.shape == (4, 16)

    def test_decode_shape(self):
        model = get_architecture(
            "autoencoder", input_dim=100, latent_dim=16
        )
        z = torch.randn(4, 16)
        out = model.decode(z)
        assert out.shape == (4, 100)

    def test_reconstruction(self):
        """Verify encode→decode roundtrip has correct shape."""
        model = get_architecture(
            "autoencoder", input_dim=30, latent_dim=5, hidden_dims=[20, 10]
        )
        x = torch.randn(8, 30)
        z = model.encode(x)
        x_hat = model.decode(z)
        assert x_hat.shape == x.shape
