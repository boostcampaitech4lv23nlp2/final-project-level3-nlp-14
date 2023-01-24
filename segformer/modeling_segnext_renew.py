import math

from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from .bricks import (
    DownSample,
    LayerScale,
    StochasticDepth,
    DWConv3x3,
    NormLayer,
    ConvBNRelu,
    ConvRelu,
    resize,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    ImageClassifierOutput,
    SemanticSegmenterOutput,
)

from .configuration_segnext import SegNextConfig

from .hamburger import HamBurger


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
        x = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        return (x, H, W)


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

        c55 = self.conv55(x)
        c17 = self.conv17_0(x)
        c17 = self.conv17_1(c17)
        c111 = self.conv111_0(x)
        c111 = self.conv111_1(c111)
        c211 = self.conv211_0(x)
        c211 = self.conv211_1(c211)

        add = c55 + c17 + c111 + c211

        mixer = self.conv11(add)

        op = mixer * skip

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

    def forward(self, x):  # input coming form Stem
        # B, N, C = x.shape
        # x = x.permute()
        x = self.msca_block(x)
        x = self.ffn_block(x)

        return x


class MSCANet(nn.Module):
    def __init__(
        self,
        in_channnels=3,
        embed_dims=[32, 64, 460, 256],
        ffn_ratios=[4, 4, 4, 4],
        depths=[3, 3, 5, 2],
        num_stages=4,
        ls_init_val=1e-2,
        drop_path=0.0,
        norm_type=dict(type="sync_bn"),
    ):
        super(MSCANet, self).__init__()
        # print(f'MSCANet {drop_path}')
        self.depths = depths
        self.num_stages = num_stages
        # stochastic depth decay rule (similar to linear decay) / just like matplot linspace
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                input_embed = StemConv(in_channnels, embed_dims[0], norm_type)
            else:
                input_embed = DownSample(
                    in_channels=embed_dims[i - 1], embed_dim=embed_dims[i]
                )

            stage = nn.ModuleList(
                [
                    StageMSCA(
                        dim=embed_dims[i],
                        ffn_ratio=ffn_ratios[i],
                        ls_init_val=ls_init_val,
                        drop_path=dpr[cur + j],
                        norm_type=norm_type["type"],
                    )
                    for j in range(depths[i])
                ]
            )

            norm_layer = NormLayer(embed_dims[i], norm_type=norm_type["type"])
            cur += depths[i]

            setattr(self, f"input_embed{i+1}", input_embed)
            setattr(self, f"stage{i+1}", stage)
            setattr(self, f"norm_layer{i+1}", norm_layer)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            input_embed = getattr(self, f"input_embed{i+1}")
            stage = getattr(self, f"stage{i+1}")
            norm_layer = getattr(self, f"norm_layer{i+1}")

            x, H, W = input_embed(x)

            for stg in stage:
                x = stg(x)

            x = norm_layer(x)
            outs.append(x)

        return outs


# decoder
class HamDecoder(nn.Module):
    """SegNext"""

    def __init__(self, outChannels, config, enc_embed_dims=[32, 64, 460, 256]):
        super().__init__()

        ham_channels = config["ham_channels"]

        self.squeeze = ConvRelu(sum(enc_embed_dims[1:]), ham_channels)
        self.ham_attn = HamBurger(ham_channels, config)
        self.align = ConvRelu(ham_channels, outChannels)

    def forward(self, features):

        features = features[1:]  # drop stage 1 features b/c low level
        features = [
            resize(feature, size=features[-3].shape[2:], mode="bilinear")
            for feature in features
        ]
        x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        x = self.ham_attn(x)
        x = self.align(x)

        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=None,
            attentions=None,
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


class SegFormerImageClassifierOutput(ImageClassifierOutput):
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


class SegNextForImageClassification(SegNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.segformer = MSCANet(
            in_channnels=config.num_channels,
            embed_dims=config.hidden_sizes,
            ffn_ratios=config.ffn_ratios,
            depths=config.depths,
            num_stages=config.num_encoder_blocks,
            drop_path=config.drop_path_rate,
            norm_type=config.norm_type,
        )

        # Classifier head
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SegFormerImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.segformer(
            pixel_values,
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

        return SegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SegNextForSemanticSegmentation(SegNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.cls_conv = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(config.decoder_hidden_size, config.num_classes, kernel_size=1),
        )
        self.encoder = MSCANet(
            in_channnels=config.num_channel,
            embed_dims=config.hidden_size,
            ffn_ratios=config.ffn_ratios,
            depths=config.depths,
            num_stages=config.num_encoder_blocks,
            dropout=config.hidden_dropout_probs,
            drop_path=config.drop_path_rate,
            norm_type=config.norm_type,
        )
        self.decoder = HamDecoder(
            outChannels=config.dec_outChannels,
            config=config,
            enc_embed_dims=config.embed_dims,
        )
        self.post_init()

    def forward(self, pixel_values, labels):

        enc_feats = self.encoder(pixel_values)
        dec_out = self.decoder(enc_feats)
        logits = self.cls_conv(dec_out)  # here output will be B x C x H/8 x W/8
        # output = F.interpolate(
        #     output, size=x.size()[-2:], mode="bilinear", align_corners=True
        # )  # now its same as input
        # #  bilinear interpol was used originally

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

        return SemanticSegmenterOutput(loss=loss, logits=logits)
