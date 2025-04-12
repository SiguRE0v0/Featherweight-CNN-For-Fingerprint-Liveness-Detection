import sys
sys.path.append('/root/Fingerprint_liveness_detection/')

from Model import FLDNet
from torchsummary import summary
from io import StringIO
import contextlib
import torch
from thop import profile


def param_summary():
    model = FLDNet(in_channels=1, out_classes=1, enable_se=True, spp=True, invert=True).cuda()
    output_buffer = StringIO()
    summary(model, input_size=(1, 160, 160), device='cuda')
    with contextlib.redirect_stdout(output_buffer):
        summary(model, input_size=(1, 160, 160), device='cuda')

    input = torch.randn(1, 1, 160, 160).cuda()
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"FLOPS: {flops / 1e6} M")

    with open("../model_param.txt", "w") as f:
        f.write(output_buffer.getvalue())
        f.write(f"FLOPS: {flops / 1e6} M")


if __name__ == '__main__':
    param_summary()
