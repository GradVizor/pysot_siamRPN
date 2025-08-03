import argparse
import torch
import torch.nn as nn
import numpy as np

from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg


# ✅ ONNX Export Wrapper
class SiamRPNExportWrapper(nn.Module):
    def __init__(self, model):
        super(SiamRPNExportWrapper, self).__init__()
        self.model = model

    def forward(self, template, search):
        self.model.template(template)
        outputs = self.model.track(search)
        return outputs['cls'], outputs['loc']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml')
    parser.add_argument('--snapshot', type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--output', type=str, default='siamrpn.onnx',
                        help='Output ONNX filename')
    args = parser.parse_args()

    # ✅ Load config and model
    cfg.merge_from_file(args.config)
    cfg.CUDA = False  # Force CPU export

    model = ModelBuilder()
    checkpoint = torch.load(args.snapshot, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # ✅ Wrap for ONNX export
    export_model = SiamRPNExportWrapper(model)

    # ✅ Create dummy inputs
    template = torch.randn(1, 3, 127, 127)
    search = torch.randn(1, 3, 255, 255)

    # ✅ Export to ONNX
    torch.onnx.export(export_model,
                      (template, search),
                      args.output,
                      input_names=['template', 'search'],
                      output_names=['cls', 'loc'],
                      opset_version=11)

    print(f"✅ Successfully exported ONNX model: {args.output}")


if __name__ == '__main__':
    main()
