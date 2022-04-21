# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

class MAEDecoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        # --------------------------------------------------------------------------
        # MAE decoder specifics 实例化decoder部分
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) #为了替换掉哪些被遮住的块。遮掩的块可以使用这样的全0数值来表示

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding，num_patches表示的是patch的数目。+1加的是cls token

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim) #对decoder的输出也进行layernorm的处理
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch，将embedding变成patch
        # --------------------------------------------------------------------------

    def forward_decoder(self, x): #x是编码器的输出
        # embed tokens
        x = self.decoder_embed(x) #为了将encoder输出的embeeding的维度，降维到encdoer的输入的维度

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection 映射到像素空间上
        x = self.decoder_pred(x)

        # remove cls token x是最终图像，cls token没有用处。这里是一个回归任务。
        return x

    def forward(self, latent):
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        #loss = self.forward_loss(imgs, pred, mask)
        return pred


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MAEDecoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MAEDecoder(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MAEDecoder(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

def test_mae():
    device = device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    my_mae = mae_vit_base_patch16().to(device)
    #print(my_mae)

    input_tensor = torch.rand(64, 196, 768, device=device)
    pred = my_mae(input_tensor)
    print(pred.shape)  #torch.Size([64, 196, 768])

def test_block():
    device = device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    input_tensor = torch.rand(64, 196, 512, device=device) #64个batch，每个batch中196个向量，每个向量的维度是512维

    transformer_block = Block(dim=512, num_heads=16).to(device)

    output = transformer_block(input_tensor)
    print('transformer block output:', output.shape)

if __name__ == '__main__':
    #test_mae()
    test_block()

