import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import argparse
import torch
import pytorch_model_summary


from segformer import SegformerForSemanticSegmentation, SegformerConfig
from segformer.modeling_segnext import SegNextForSemanticSegmentation
from segformer.configuration_segnext import SegNextConfig


def main(opt):

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(opt.data_dir, "id2label.json"), "r") as f:
        id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    model = SegformerForSemanticSegmentation(
        config=SegformerConfig(
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        ),
    )
    # model = SegNextForSemanticSegmentation(
    #     config=SegNextConfig(
    #         num_labels=len(id2label),
    #         id2label=id2label,
    #         label2id=label2id,
    #         ignore_mismatched_sizes=True,
    #     ),
    # )

    model = model.to(dev)
    input_vector = torch.randn(1, 3, 512, 512)
    print(
        pytorch_model_summary.summary(
            model.cpu(),
            input_vector,
            show_parent_layers=True,
            show_input=True,
            show_hierarchical=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/opt/ml/final-project-level3-nlp-14/dataset/ADEChallengeData2016",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
