from realesrgan import RealESRGANer
import torch, numpy as np, torch.nn.functional as F

def convert(self: RealESRGANer, img):
    img = torch.from_numpy(np.array(np.transpose(img, (2, 0, 1))))
    img = img.unsqueeze(0).to(self.device)
    if self.half:
        img = img.half()
    else:
        img = img.float()
    return img / 255.0

def pre_process_batched(self: RealESRGANer):
    # pre_pad
    if self.pre_pad != 0:
        self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
    # mod pad for divisible borders
    if self.scale == 2:
        self.mod_scale = 2
    elif self.scale == 1:
        self.mod_scale = 4
    if self.mod_scale is not None:
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.img.size()
        if (h % self.mod_scale != 0):
            self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
        if (w % self.mod_scale != 0):
            self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
        self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')


@torch.no_grad()
def batch_enhance_rgb(self: RealESRGANer, imgs, outscale=None, alpha_upsampler='realesrgan'):
    tensors = []
    for img in imgs:
        # img: numpy
        if np.max(img) > 256:  # 16-bit image
            assert False
        if len(img.shape) == 2:  # gray image
            assert False
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            assert False
        tensors.append(convert(self, img))

    self.img = torch.cat(tensors)
    # ------------------- process image (without the alpha channel) ------------------- #
    pre_process_batched(self)
    if self.tile_size > 0:
        self.tile_process()
    else:
        self.process()
    output_img = self.post_process()
    if outscale is not None and outscale != float(self.scale):
        output_img = F.interpolate(output_img, scale_factor=outscale / float(self.scale), mode='area')
    output_img = (output_img * 255).clamp_(0, 255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()
    for output in output_img:
        yield output
