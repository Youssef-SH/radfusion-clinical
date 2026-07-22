"""Define the standard TorchXRayVision CXR encoder and binary model."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from radfusion.training.config import ModelConfig


class StandardCxrEncoder(nn.Module):
    """Expose pooled DenseNet121 features through a stable encoding boundary."""

    def __init__(
        self,
        *,
        weights: str = "densenet121-res224-chex",
        expected_embedding_dimension: int = 1024,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if weights != "densenet121-res224-chex":
            raise ValueError("Standard CXR encoder requires densenet121-res224-chex weights")
        if expected_embedding_dimension != 1024 or image_size != 224:
            raise ValueError("Standard CXR encoder requires dimensions 1024 and 224")
        self.weights = weights
        self.embedding_dimension = expected_embedding_dimension
        self.image_size = image_size
        import torchxrayvision as xrv

        self.backbone = xrv.models.DenseNet(weights=weights)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return one 1024-dimensional pooled embedding per image."""
        _validate_images(images, self.image_size)
        embeddings = self.backbone.features2(images)
        _validate_embeddings(embeddings, len(images), self.embedding_dimension)
        return embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return pooled embeddings."""
        return self.encode(images)


class CxrBinaryClassifier(nn.Module):
    """Combine a reusable CXR encoder with a separate one-logit head."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        embedding_dimension: int = 1024,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("CXR encoder must be a torch.nn.Module")
        if isinstance(embedding_dimension, bool) or not isinstance(embedding_dimension, int):
            raise ValueError("embedding_dimension must be an integer")
        if isinstance(image_size, bool) or not isinstance(image_size, int):
            raise ValueError("image_size must be an integer")
        if embedding_dimension != 1024 or image_size != 224:
            raise ValueError("CXR model requires dimensions 1024 and 224")
        self.encoder = encoder
        self.embedding_dimension = embedding_dimension
        self.image_size = image_size
        self.classifier = nn.Linear(embedding_dimension, 1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return one validated embedding per image."""
        _validate_images(images, self.image_size)
        encode = getattr(self.encoder, "encode", None)
        embeddings = encode(images) if callable(encode) else self.encoder(images)
        _validate_embeddings(embeddings, len(images), self.embedding_dimension)
        return embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one raw binary logit per image."""
        return self.classifier(self.encode(images)).squeeze(1)

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters for head-only warm-up."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Restore encoder parameter trainability."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True


class ImageDenseNetModel:
    """Build the fixed image-only DenseNet121 classifier."""

    def __init__(
        self,
        encoder_factory: Callable[..., nn.Module] = StandardCxrEncoder,
    ) -> None:
        self._encoder_factory = encoder_factory

    def build(self, config: ModelConfig) -> CxrBinaryClassifier:
        """Build an unfitted classifier without selecting a runtime device."""
        if config.modality != "image" or config.registry_key != "image_densenet":
            raise ValueError("Image DenseNet requires the registered image model configuration")
        if config.fit_parameters:
            raise ValueError("Image DenseNet does not accept model.fit_parameters")
        parameters = dict(config.parameters)
        expected = {
            "encoder_name",
            "weights",
            "image_size",
            "embedding_dimension",
            "class_weighting",
        }
        if set(parameters) != expected:
            raise ValueError(f"Image DenseNet parameters must be exactly {sorted(expected)}")
        if parameters["encoder_name"] != "densenet121":
            raise ValueError("Image DenseNet requires encoder_name='densenet121'")
        if parameters["weights"] != "densenet121-res224-chex":
            raise ValueError("Image DenseNet requires the fixed pretrained weights")
        if parameters["image_size"] != 224 or parameters["embedding_dimension"] != 1024:
            raise ValueError("Image DenseNet requires dimensions 224 and 1024")
        if parameters["class_weighting"] != "train_pos_weight":
            raise ValueError("Image DenseNet requires train_pos_weight class weighting")
        encoder = self._encoder_factory(
            weights=parameters["weights"],
            expected_embedding_dimension=parameters["embedding_dimension"],
            image_size=parameters["image_size"],
        )
        if not isinstance(encoder, nn.Module):
            raise TypeError("Image encoder factory must return a torch.nn.Module")
        return CxrBinaryClassifier(
            encoder,
            embedding_dimension=parameters["embedding_dimension"],
            image_size=parameters["image_size"],
        )


def _validate_images(images: object, image_size: int) -> None:
    if not isinstance(images, torch.Tensor):
        raise TypeError("CXR model input must be a torch.Tensor")
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, image_size, image_size):
        raise ValueError(f"CXR model input must have shape N x 1 x {image_size} x {image_size}")
    if images.shape[0] == 0:
        raise ValueError("CXR model input batch must be non-empty")
    if not images.is_floating_point():
        raise ValueError("CXR model input must contain floating-point values")


def _validate_embeddings(embeddings: object, batch_size: int, dimension: int) -> None:
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("CXR encoder output must be a torch.Tensor")
    if not embeddings.is_floating_point():
        raise ValueError("CXR encoder output must contain floating-point values")
    if embeddings.shape != (batch_size, dimension):
        raise ValueError(
            f"CXR encoder output must have shape ({batch_size}, {dimension}), "
            f"received {tuple(embeddings.shape)}"
        )
