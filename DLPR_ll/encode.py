import argparse
import torch
import torch.nn as nn
import time
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchac
from ll_model_eval import LosslessCompressor
import pickle
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def coding_order_table7x7(patch_sz=64, mask_type="3P"):
    if mask_type not in ("5P", "4P", "3P", "2P", "P", "S"):
        raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    COT = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)

    if mask_type == "5P":
        for i in range(patch_sz):
            start = 4 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "4P":
        for i in range(patch_sz):
            start = 3 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "3P":
        for i in range(patch_sz):
            start = 2 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "2P":
        for i in range(patch_sz):
            start = i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "P":
        for i in range(patch_sz):
            start = 1
            COT[i, :] = torch.arange(start, start + patch_sz)
    elif mask_type == "S":
        for i in range(patch_sz):
            start = patch_sz * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)

    return COT


def img2patch(img, patch_sz):
    h, w, _ = img.shape
    h_num = h//patch_sz - (patch_sz -(h % patch_sz)) // patch_sz + 1
    w_num = w//patch_sz - (patch_sz -(w % patch_sz)) // patch_sz + 1

    img_pad = nn.functional.pad(img, (0, 0, (patch_sz - w % patch_sz) % patch_sz, 0, (patch_sz - h % patch_sz) % patch_sz, 0))

    patch_h = torch.chunk(img_pad, h_num, dim=0)
    patch_h = torch.stack(patch_h, dim=0)

    patch_w = torch.chunk(patch_h, w_num, dim=2)
    patch = torch.cat(patch_w, dim=0)

    patch = patch.permute(0, 3, 1, 2)

    return patch


def load_image(image_path):

    image = Image.open(image_path)
    image = np.array(image).astype(np.float32)
    return image



def compress(model, input, COT, tau=0, patch_sz=64, mix_num=5):

    norm_scale = 1 / 255. * 2
    half = (0.5 + tau) * norm_scale
    mix_num2 = 2 * mix_num
    code_res = []
    img_shape = input.shape[:2]
    bin_sz = 2 * tau + 1
    samples_end = (255 // bin_sz) * bin_sz

    device = next(model.parameters()).device

    x = img2patch(torch.as_tensor(input), patch_sz=patch_sz).to(device)

    with torch.no_grad():
        time_start = time.time()

        code_lossy = model.lossy_compressor.compress(x)
        rec_lossy = model.lossy_compressor.decompress(code_lossy["img_strings"], code_lossy["shape"])

        time_end_ls = time.time()

        x_hat = rec_lossy["x_hat"]
        res_prior = rec_lossy["res_prior"]
        res = x - x_hat

        res_q = torch.where(res >= 0, (2 * tau + 1) * torch.floor((res + tau) / (2 * tau + 1)),
                            (2 * tau + 1) * torch.ceil((res - tau) / (2 * tau + 1)))

        res_q_max = int(torch.max(res_q).item())
        res_q_min = int(torch.min(res_q).item())

        res_q_max_norm = res_q_max * norm_scale
        res_q_min_norm = res_q_min * norm_scale

        res_q_max_idx = (res_q_max + samples_end) // bin_sz
        res_q_min_idx = (res_q_min + samples_end) // bin_sz
        print("Encode Res range:[{}({}),{}({})], elems num:{}".format(res_q_min, res_q_min_idx, res_q_max, res_q_max_idx, res_q_max_idx - res_q_min_idx + 1))

        samples = torch.arange(res_q_min, res_q_max + 1, step=bin_sz, dtype=torch.float32).to(device)
        samples = samples * norm_scale

        res_tmp = torch.zeros_like(x_hat)

        ctx_total = model.mask_conv(res_q * norm_scale)

        max_step = torch.max(COT)

        for i in range(max_step):
            h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
            ctx = ctx_total[:, :, h_idx, w_idx].unsqueeze(3)
            rp = res_prior[:, :, h_idx, w_idx].unsqueeze(3)

            res_crop = res_q[:, :, h_idx, w_idx]
            res_tmp_crop = res_tmp[:, :, h_idx, w_idx]

            res_tmp_crop = res_tmp_crop.unsqueeze(3)

            rp_ctx = model.fusion(torch.cat((rp, ctx), 1))
            lmm_params = model.residual_compressor(rp_ctx)
            mu, log_sigma, coeffs, weights = torch.split(lmm_params, 15, dim=1)
            coeffs = torch.tanh(coeffs)

            for c in range(3):
                if c == 0:
                    mu_c = mu[:, :mix_num, :, :].permute(0, 2, 1, 3)
                elif c == 1:
                    mu_c = mu[:, mix_num:mix_num2, :, :] + (res_tmp_crop[:, 0:1, :, :] * norm_scale) * coeffs[:, :mix_num, :, :]
                    mu_c = mu_c.permute(0, 2, 1, 3)
                else:
                    mu_c = mu[:, mix_num2:, :, :] + (res_tmp_crop[:, 0:1, :, :] * norm_scale) * coeffs[:, mix_num:mix_num2, :, :] + \
                           (res_tmp_crop[:, 1:2, :, :] * norm_scale) * coeffs[:, mix_num2:, :, :]
                    mu_c = mu_c.permute(0, 2, 1, 3)

                samples_centered = samples - mu_c
                inv_sigma = torch.exp(-log_sigma[:, c * mix_num:(c + 1) * mix_num, :, :].permute(0, 2, 1, 3))
                plus_in = inv_sigma * (samples_centered + half)
                cdf_plus = torch.sigmoid(plus_in)
                min_in = inv_sigma * (samples_centered - half)
                cdf_min = torch.sigmoid(min_in)
                cdf_delta = cdf_plus - cdf_min
                one_minus_cdf_min = torch.exp(-F.softplus(min_in))  # res_q_max
                cdf_plus = torch.exp(plus_in - F.softplus(plus_in))  # res_q_min

                samples2 = samples - torch.zeros_like(mu_c)
                cdf_delta = torch.where(samples2 - half < res_q_min_norm, cdf_plus,
                                        torch.where(samples2 + half > res_q_max_norm, one_minus_cdf_min,
                                                    cdf_delta))

                weights_c = weights.permute(0, 2, 1, 3)
                m = torch.amax(weights_c, 2, keepdim=True)
                weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
                pmf = torch.sum(cdf_delta * weights_c, dim=2)

                pmf = pmf.clamp_(1. / 64800, 1.)
                pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
                cdf = torch.cumsum(pmf, dim=2).clamp_(0., 1.)
                cdf = F.pad(cdf, (1, 0))

                symbol = torch.div(res_crop[:, c, :].short() - res_q_min, bin_sz, rounding_mode='floor')
                res_stream = torchac.encode_float_cdf(cdf.cpu(), symbol.cpu(), needs_normalization=False, check_input_bounds=False)
                code_res.append(res_stream)
                res_tmp_crop[:, c, :, 0] = symbol.float() * bin_sz + res_q_min

            res_tmp[:, :, h_idx, w_idx] = res_tmp_crop.squeeze(3)

        time_end_res = time.time()

    print("total runtime:{:.2f}, lossy runtime:{:.2f}, res runtime:{:.2f}".format(time_end_res - time_start, time_end_ls - time_start, time_end_res - time_end_ls))

    return code_lossy, code_res, img_shape, (res_q_min, res_q_max)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image compression using neural networks.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output bitstream file path')

    args = parser.parse_args()

    ckp_dir = "./ckp_ll"

    I = load_image(args.input)

    device = torch.device('cuda')
    ll_module = LosslessCompressor(192).eval().to(device)

    ckp = torch.load(os.path.join(ckp_dir, "ckp.tar"), map_location=device)
    ll_module.load_state_dict(ckp['model_state_dict'])
    ll_module.lossy_compressor.update(force=True)

    # COT
    COT = coding_order_table7x7()

    code_lossy, code_res, img_shape, res_range = compress(ll_module, I, COT)

    img_ls_sz = sum([len(code_lossy['img_strings'][0][i]) + len(code_lossy['img_strings'][1][i]) for i in
                     range(len(code_lossy['img_strings'][0]))])
    res_sz = sum([len(code_res[i]) for i in range(len(code_res))])
    ll_sz = img_ls_sz + res_sz + 6 * 4  # lossy_image_bitstream + residual_bitstream + z_shape + image_shape + residual_range


    img_bpsp = img_ls_sz * 8 / np.prod(I.shape)
    res_bpsp = res_sz * 8 / np.prod(I.shape)
    bpsp = ll_sz * 8 / np.prod(I.shape)


    print("bpsp:{:.4f}, img_bpsp:{:.4f}, res_bpsp:{:.4f}".format(bpsp, img_bpsp, res_bpsp))

    with open(args.output, 'wb') as f:
        pickle.dump((code_lossy, code_res, img_shape, res_range), f)

    print(f"Compression finished. Bitstream saved to {args.output}")
