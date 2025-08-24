from typing import Any, Dict, List, Optional, Union, Tuple
from typing import Iterable

import einops
import timm
import torch
import torchvision
from torch import nn
import math
from torch.nn.utils.parametrizations import spectral_norm

class PatchDropout(nn.Module):
    def __init__(self, apply_proba, min_patches_dropped, max_patches_dropped, patch_qty):
        super(PatchDropout, self).__init__()
        self.apply_proba = apply_proba
        self.min_patches_dropped, self.max_patches_dropped = min_patches_dropped, max_patches_dropped
        self.patch_qty = patch_qty
        self.do_dropout = True

    def forward(self, batch):
        apply_dropout = torch.rand(1)
        if self.do_dropout and self.max_patches_dropped > 1e-03 and apply_dropout < self.apply_proba:
            dropped_patches = torch.rand(1) * (self.max_patches_dropped - self.min_patches_dropped) + self.min_patches_dropped
            left_patches_qty = self.patch_qty - int(dropped_patches * self.patch_qty)
            batch_qty = batch.shape[0]
            batch_idxs = torch.arange(batch_qty).tolist()
            patch_idxs = [torch.multinomial(torch.ones(self.patch_qty), left_patches_qty, replacement = False).tolist() \
                          for i in range(len(batch_idxs))]
            new_batch = []
            for i in batch_idxs:
                new_batch.append(batch[i, patch_idxs[i]])
            new_batch = torch.stack(new_batch, axis = 0)
            return new_batch
        return batch

    def turn_off_dropout(self):
        self.do_dropout = False

    def turn_on_dropout(self):
        self.do_dropout = True

# MONKEY PATCHING

def patch_timm_for_fx_tracing():
    """Patch timm to allow torch.fx tracing."""

    def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = "bicubic",
        antialias: bool = True,
        verbose: bool = False,
    ):
        """From timm.layers.pos_embed.resample_abs_pose_embed.

        To avoid control flow using dynamic variables, the check returning early for same size
        is not executed.
        """
        # sort out sizes, assume square if old size not provided
        num_pos_tokens = posemb.shape[1]

        # REMOVED because this relies on dynamic variables:
        # num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
        # if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        #    return posemb

        if old_size is None:
            hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
            old_size = hw, hw

        if num_prefix_tokens:
            posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
        else:
            posemb_prefix, posemb = None, posemb

        # do the interpolation
        embed_dim = posemb.shape[-1]
        orig_dtype = posemb.dtype
        posemb = posemb.float()  # interpolate needs float32
        posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
        posemb = nn.functional.interpolate(
            posemb, size=new_size, mode=interpolation, antialias=antialias
        )
        posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        posemb = posemb.to(orig_dtype)

        # add back extra (class, etc) prefix tokens
        if posemb_prefix is not None:
            posemb = torch.cat([posemb_prefix, posemb], dim=1)

        return posemb

    # Monkey patch method in vision transformer
    timm.models.vision_transformer.resample_abs_pos_embed = resample_abs_pos_embed


torch.fx.wrap("int")  # Needed to allow tracing with int()
patch_timm_for_fx_tracing()

# END MONKEY PATCHING

class FrameEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        backbone_kwargs: nn.Module,
        output_transform_kwargs: Optional[nn.Module] = None,
        obs_size = None,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
    ):
        super().__init__()
        
        self.backbone = TimmExtractor(model = backbone_kwargs['model'],
                                      pretrained = backbone_kwargs['pretrained'],
                                      frozen = backbone_kwargs['frozen'],
                                      features = backbone_kwargs['features'],
                                      checkpoint_path = None,
                                      model_kwargs = backbone_kwargs['model_kwargs'])
        self.output_transform = None
        if output_transform_kwargs is not None:
            self.output_transform = MLP(inp_dim = output_transform_kwargs['inp_dim'], 
                                    outp_dim = output_transform_kwargs['outp_dim'], 
                                    hidden_dims = output_transform_kwargs['hidden_dim'],
                                    initial_layer_norm = output_transform_kwargs['layer_norm'])
        
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key
        self.resizer = lambda x: x
        obs_side, kernel_side = obs_size[0], self.backbone.model.patch_embed.proj.kernel_size[0]
        if backbone_kwargs['resize']:
            self.resizer = Resizer(((obs_side + (kernel_side - obs_side % kernel_side)) * 2, 
                                    (obs_side + (kernel_side - obs_side % kernel_side)) * 2), 
                                    patch_inputs = False)
        
        self.patch_dropout = PatchDropout(apply_proba = backbone_kwargs.patch_dropout.apply_dropout_proba, 
                                          min_patches_dropped = backbone_kwargs.patch_dropout.min_patches_dropped, 
                                          max_patches_dropped = backbone_kwargs.patch_dropout.max_patches_dropped, 
                                          patch_qty = (obs_side // kernel_side) ** 2)
        self.backbone.model.patch_drop = self.patch_dropout

    def forward(self, images: torch.Tensor, patch_dropout) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        images = self.resizer(images)
        if patch_dropout:
            self.patch_dropout.turn_on_dropout()
        else:
            self.patch_dropout.turn_off_dropout()
        backbone_features = self.backbone(images)
        if isinstance(backbone_features, dict):
            features = backbone_features[self.main_features_key].clone()
        else:
            features = backbone_features.clone()

        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if self.output_transform:
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            main_backbone_features = backbone_features[self.main_features_key]

            return {
                "features": features,
                "backbone_features": main_backbone_features,
                **backbone_features,
            }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"

            return {
                "features": features,
                "backbone_features": backbone_features,
            }



class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{i + 1}": f"blocks.{i}" for i in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")

        model = TimmExtractor._create_model(model_name, pretrained, checkpoint_path, model_kwargs)

        if self.features is not None:
            nodes = torchvision.models.feature_extraction.get_graph_node_names(model)[0]

            features = []
            for name in self.features:
                if name in TimmExtractor.FEATURE_ALIASES:
                    name = TimmExtractor.FEATURE_ALIASES[name]

                if not any(node.startswith(name) for node in nodes):
                    raise ValueError(
                        f"Requested features under node {name}, but this node does "
                        f"not exist in model {model_name}. Available nodes: {nodes}"
                    )

                features.append(name)

            model = torchvision.models.feature_extraction.create_feature_extractor(model, features)

        self.model = model

        if self.frozen:
            self.requires_grad_(False)

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractor._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            )

        return model

    def forward(self, inp):
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        if self.features is not None:
            if self.is_vit:
                outputs = {k: v[:, 1:] for k, v in outputs.items()}  # Remove CLS token
            outputs = {self.FEATURE_MAPPING[key]: value for key, value in outputs.items()}
            for name in self.features:
                if ("keys" in name) or ("queries" in name) or ("values" in name):
                    feature_name = name.replace("queries", "keys").replace("values", "keys")
                    B, N, C = outputs[feature_name].shape
                    qkv = outputs[feature_name].reshape(
                        B, N, 3, C // 3
                    )  # outp has shape B, N, 3 * H * (C // H)
                    q, k, v = qkv.unbind(2)
                    if "keys" in name:
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v
                    else:
                        raise ValueError(f"Unknown feature name {name}.")

            if len(outputs) == 1:
                # Unpack single output for now
                return next(iter(outputs.values()))
            else:
                return outputs
        else:
            return outputs

class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        initial_layer_norm: bool = False,
        dropout_prob = 0.,
        spectral_normalisation = False
    ):
        super().__init__()

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            linear = nn.Linear(cur_dim, dim)
            if spectral_normalisation:
                linear = spectral_norm(linear)
            layers.append(linear)
            if dropout_prob > 1e-03:
                layers.append(nn.Dropout(p = dropout_prob))
            layers.append(nn.ReLU(inplace=True))
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))

        self.layers = nn.Sequential(*layers)
        init_parameters(self.layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)
        return outp
    

def init_parameters(layers: Union[nn.Module, Iterable[nn.Module]]):
    if isinstance(layers, nn.Module):
        layers = [layers]

    for idx, layer in enumerate(layers):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)


class MLPDecoder(nn.Module):
    """Decoder that reconstructs independently for every position and slot."""

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        n_patches: int,
        dropout_prob: int,
        spectral_normalisation: bool,
        eval_output_size: Optional[Tuple[int]] = None,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        self.n_patches = n_patches
        self.eval_output_size = list(eval_output_size) if eval_output_size else None

        self.mlp = MLP(inp_dim, outp_dim + 1, hidden_dims, dropout_prob = dropout_prob, 
                       spectral_normalisation = spectral_normalisation)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, n_patches, inp_dim) * inp_dim**-0.5)

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs, n_slots, dims = slots.shape

        if not self.training and self.eval_output_size is not None:
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(
                self.pos_emb.squeeze(1),
                new_size=self.eval_output_size,
                num_prefix_tokens=0,
            ).unsqueeze(1)
        else:
            pos_emb = self.pos_emb

        slots = slots.view(bs, n_slots, 1, dims).expand(bs, n_slots, pos_emb.shape[2], dims)
        slots = slots + pos_emb

        recons, alpha = self.mlp(slots).split((self.outp_dim, 1), dim=-1)

        masks = torch.softmax(alpha, dim=1)
        recon = torch.sum(recons * masks, dim=1)

        return {"reconstruction": recon, "masks": masks.squeeze(-1)}
    
class Resizer:
    """Module that takes image-based tensor and resizes it to an appropriate size.

    Args:
        size: Tuple of (height, width) to resize to. If unspecified, assume an additional
            input used to infer the size. The last two dimensions of this input are taken
            as height and width.
        patch_inputs: If true, assumes tensor to resize has format `(batch, [frames],
            channels, n_points)` instead of separate height, width dimensions.
        patch_outputs: If true, flatten spatial dimensions after resizing.
    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        patch_inputs: bool = False,
    ):

        self.size = size
        self.patch_inputs = patch_inputs
        self.n_expected_dims = 4 - (1 if patch_inputs else 0)

    def __call__(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        if inputs.ndim != self.n_expected_dims:
            raise ValueError(
                f"Mask has {inputs.ndim} dimensions, but expected it to "
                f"have {self.n_expected_dims} dimensions."
            )

        assert self.size is not None
        size = list(self.size)

        if self.patch_inputs:
            n_patches = inputs.shape[-1]
            ratio = size[1] / size[0]
            height = int(math.sqrt(n_patches / ratio))
            width = int(math.sqrt(n_patches * ratio))
            if height * width != n_patches:
                if height == width:
                    raise ValueError(
                        f"Can not reshape {n_patches} patches to square aspect ratio as it's not a "
                        "perfect square."
                    )
                raise ValueError(f"Can not reshape {n_patches} patches to aspect ratio {ratio}.")

            inputs = inputs.unflatten(-1, (height, width))

        dtype = inputs.dtype
        if inputs.dtype == torch.bool:
            inputs = inputs.to(torch.uint8)

        outputs = torch.nn.functional.interpolate(inputs, size = size, mode = 'bilinear')

        if inputs.dtype != dtype:
            inputs = inputs.to(dtype)

        return outputs