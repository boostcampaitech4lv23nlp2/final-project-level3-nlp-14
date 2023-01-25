# coding=utf-8
# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SegFormer model."""


import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    ImageClassifierOutput,
    SemanticSegmenterOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_segnext import SegNextConfig

from .bricks import (
    LayerScale,
    StochasticDepth,
    DWConv3x3,
    NormLayer,
    ConvBNRelu,
    ConvRelu,
    resize,
)

from .hamburger import HamBurger


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "SegNextConfig"
_FEAT_EXTRACTOR_FOR_DOC = "SegNextFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "nvidia/mit-b0"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 16, 16]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "nvidia/mit-b0"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    # See all SegFormer models at https://huggingface.co/models?filter=segformer
]


class SegNextImageClassifierOutput(ImageClassifierOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(
    input, drop_prob: float = 0.0, training: bool = False, scale_by_keep=True
):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (
        input.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device
    )
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Segformer
class SegNextDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class StemConv(nn.Module):
    """following ConvNext paper"""

    def __init__(self, in_channels, out_channels, norm_type, bn_momentum=0.99):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            NormLayer(out_channels // 2, norm_type=norm_type["type"]),
            nn.GELU(),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            NormLayer(out_channels, norm_type=norm_type["type"]),
        )

    def forward(self, x):
        embeddings = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1, 2)  # B*C*H*W -> B*C*HW -> B*HW*C
        return (embeddings, H, W)


class DownSample(nn.Module):
    def __init__(self, kernelSize=3, stride=2, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(kernelSize, kernelSize),
            stride=stride,
            padding=(kernelSize // 2, kernelSize // 2),
        )
        # stride 4 => 4x down sample
        # stride 2 => 2x down sample

    def forward(self, x):
        embeddings = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1, 2)
        return embeddings, H, W


class SegformerEfficientSelfAttention(nn.Module):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(
        self, config, hidden_size, num_attention_heads, sequence_reduction_ratio
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=sequence_reduction_ratio,
                stride=sequence_reduction_ratio,
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, hidden_states):
        new_shape = hidden_states.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # Reshape to (batch_size, num_channels, height, width)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(
                batch_size, num_channels, height, width
            )
            # Apply sequence reduction
            hidden_states = self.sr(hidden_states)
            # Reshape back to (batch_size, seq_len, num_channels)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(
                0, 2, 1
            )
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class FFN(nn.Module):
    """following ConvNext paper"""

    def __init__(self, in_channels, out_channels, hid_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hid_channels, 1)
        self.dwconv = DWConv3x3(hid_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_channels, out_channels, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class BlockFFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hid_channels,
        norm_type="sync_bn",
        ls_init_val=1e-2,
        drop_path=0.0,
    ):
        super().__init__()
        self.norm = NormLayer(in_channels, norm_type=norm_type)
        self.ffn = FFN(in_channels, out_channels, hid_channels)
        self.layer_scale = LayerScale(in_channels, init_value=ls_init_val)
        self.drop_path = StochasticDepth(p=drop_path)

    def forward(self, x):
        skip = x.clone()

        x = self.norm(x)
        x = self.ffn(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        op = skip + x
        return op


class MSCA(nn.Module):
    def __init__(self, dim):
        super(MSCA, self).__init__()
        # input
        self.maxpl = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.avgpl = nn.AvgPool2d(3, stride = 1, padding = 1)
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # split into multipats of multiscale attention
        self.conv17_0 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv17_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv111_0 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv111_1 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv211_0 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv211_1 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv11 = nn.Conv2d(dim, dim, 1)  # channel mixer

    def forward(self, x):

        skip = x.clone()
        x2 = self.maxpl(x.clone())
        x1 = self.avgpl(x.clone())
        
        c55 = self.conv55(x)
        c17 = self.conv17_0(x)
        c17 = self.conv17_1(c17)
        c111 = self.conv111_0(x)
        c111 = self.conv111_1(c111)
        c211 = self.conv211_0(x)
        c211 = self.conv211_1(c211)

        add = c55 + c17 + c111 + c211

        mixer = self.conv11(add)


        c55_1 = self.conv55(x1)
        c17_1 = self.conv17_0(x1)
        c17_1 = self.conv17_1(c17_1)
        c111_1 = self.conv111_0(x1)
        c111_1 = self.conv111_1(c111_1)
        c211_1 = self.conv211_0(x1)
        c211_1 = self.conv211_1(c211_1)

        add_1 = c55_1 + c17_1 + c111_1 + c211_1

        mixer_1 = self.conv11(add_1)

        c55_2 = self.conv55(x2)
        c17_2 = self.conv17_0(x2)
        c17_2 = self.conv17_1(c17_2)
        c111_2 = self.conv111_0(x2)
        c111_2 = self.conv111_1(c111_2)
        c211_2 = self.conv211_0(x2)
        c211_2 = self.conv211_1(c211_2)

        add_2 = c55_2 + c17_2 + c111_2 + c211_2

        mixer_2 = self.conv11(add_2)

        op = (mixer + mixer_1 + mixer_2) * skip
        return op


class BlockMSCA(nn.Module):
    def __init__(self, dim, ls_init_val=1e-2, drop_path=0.0, norm_type="sync_bn"):
        super().__init__()
        self.norm = NormLayer(dim, norm_type=norm_type)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.msca = MSCA(dim)
        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.layer_scale = LayerScale(dim, init_value=ls_init_val)
        self.drop_path = StochasticDepth(p=drop_path)
        # print(f'BlockMSCA {drop_path}')

    def forward(self, x):

        skip = x.clone()

        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.msca(x)
        x = self.proj2(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        out = x + skip

        return out


class StageMSCA(nn.Module):
    def __init__(
        self, dim, ffn_ratio=4.0, ls_init_val=1e-2, drop_path=0.0, norm_type="sync_bn"
    ):
        super().__init__()
        # print(f'StageMSCA {drop_path}')
        self.msca_block = BlockMSCA(dim, ls_init_val, drop_path, norm_type)

        ffn_hid_dim = int(dim * ffn_ratio)
        self.ffn_block = BlockFFN(
            in_channels=dim,
            out_channels=dim,
            hid_channels=ffn_hid_dim,
            norm_type=norm_type,
            ls_init_val=ls_init_val,
            drop_path=drop_path,
        )

    def forward(self, hidden_states):  # input coming form Stem
        # B, N, C = x.shape
        # x = x.permute()
        x = self.msca_block(hidden_states)
        x = self.ffn_block(x)

        return (x,)


class SegNextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        drop_path_decays = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))
        ]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            if i == 0:
                embeddings.append(
                    StemConv(
                        config.num_channels, config.hidden_sizes[0], config.norm_type
                    )
                )
            else:
                embeddings.append(
                    DownSample(
                        in_channels=config.hidden_sizes[i - 1],
                        embed_dim=config.hidden_sizes[i],
                    )
                )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    StageMSCA(
                        dim=config.hidden_sizes[i],
                        ffn_ratio=config.ffn_ratios[i],
                        ls_init_val=1e-2,
                        drop_path=drop_path_decays[cur + j],
                        norm_type=config.norm_type["type"],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList(
            [
                NormLayer(config.hidden_sizes[i], norm_type=config.norm_type["type"])
                for i in range(config.num_encoder_blocks)
            ]
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, x in enumerate(
            zip(self.patch_embeddings, self.block, self.layer_norm)
        ):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SegNextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SegNextConfig
    base_model_prefix = "segnext"
    main_input_name = "pixel_values"

    # def _init_weights(self, module):
    #     """Initialize the weights"""
    #     if isinstance(module, (nn.Linear, nn.Conv2d)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, val=1.0)
            nn.init.constant_(module.bias, val=0.0)
        if isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            nn.init.normal_(module.weight, std=math.sqrt(2.0 / fan_out), mean=0)


SEGFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SegformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SEGFORMER_INPUTS_DOCSTRING = r"""

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`SegformerFeatureExtractor`]. See [`SegformerFeatureExtractor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.",
    SEGFORMER_START_DOCSTRING,
)
class SegNextModel(SegNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SegNextEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        SEGFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)")
    )
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden
    states) e.g. for ImageNet.
    """,
    SEGFORMER_START_DOCSTRING,
)
class SegNextForImageClassification(SegNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.segnext = SegNextModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=SegNextImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SegNextImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.segnext(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # convert last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(
            batch_size, -1, self.config.hidden_sizes[-1]
        )

        # global average pooling
        sequence_output = sequence_output.mean(dim=1)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SegNextImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# decoder
class HamDecoder(nn.Module):
    """SegNext"""

    def __init__(self, config):
        super().__init__()

        ham_channels = config.decoder_hidden_size

        self.classifier = nn.Conv2d(
            config.decoder_hidden_size, config.num_labels, kernel_size=1
        )

        self.squeeze = ConvRelu(sum(config.hidden_sizes[1:]), ham_channels)
        self.ham_attn = HamBurger(ham_channels, config)
        self.align = ConvRelu(ham_channels, ham_channels)

    def forward(self, features):

        features = features[1:]  # drop stage 1 features b/c low level
        features = [
            resize(feature, size=features[-3].shape[2:], mode="bilinear")
            for feature in features
        ]
        hidden_states = torch.cat(features, dim=1)

        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.ham_attn(hidden_states)
        hidden_states = self.align(hidden_states)
        logits = self.classifier(hidden_states)

        return logits


@add_start_docstrings(
    """SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.""",
    SEGFORMER_START_DOCSTRING,
)
class SegNextForSemanticSegmentation(SegNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.segnext = SegNextModel(config)
        self.decode_head = HamDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        >>> list(logits.shape)
        [1, 150, 128, 128]
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        outputs = self.segnext(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            if not self.config.num_labels > 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(
                    ignore_index=self.config.semantic_loss_ignore_index
                )
                loss = loss_fct(upsampled_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
