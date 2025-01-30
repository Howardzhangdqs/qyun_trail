import torch
import torch.onnx as onnx
import torchvision.models as models

import os
import sys

import argparse
from rich import print

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from config import config
    from ml import LaneDetectionModel
except ImportError:
    raise ImportError("Please run this script from the root directory of the project")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出模型")

    parser.add_argument("-w", "--weight", type=str, required=True, help="模型权重路径")
    parser.add_argument("-t", "--type", type=str, default="onnx", choices=["onnx"], help="导出格式")

    args = parser.parse_args()

    model = LaneDetectionModel()
    model.load_state_dict(torch.load(args.weight)["model_state"])
    model.eval().to(device)

    os.makedirs("export", exist_ok=True)

    if args.type == "onnx":

        dummy = torch.randn(1, 3, 224, 224).to(device)
        outputs = model(dummy)

        filename = os.path.basename(args.weight)
        filename = filename[: filename.rfind(".")]

        export_file = f"./export/{filename}.onnx"

        onnx.export(model, dummy, export_file)

        print(f"模型导出至 [yellow]`[underline]{export_file}[/underline]`[/yellow]")
