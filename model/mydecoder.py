import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
class Residual_3(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, y,**kwargs):
        return self.fn(x, y, **kwargs) + x

class PreNorm_3(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y.half()),**kwargs)
class Attention_mydecoder(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,txt, image ,kv_mask = None,q_mask = None):
        b1, n1, _, h = *image.shape, self.heads
        b2, n2, _, h = *txt.shape, self.heads

        q = self.to_q(txt)
        q = rearrange(q,'b n (h d) -> b h n d', h=h)
        kv = self.to_kv(image).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if kv_mask is not None:
            assert kv_mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(q_mask, 'b i -> b () i ()') * rearrange(kv_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Transformer_mydecoder(nn.Module):          # Transformer Encoder
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual_3(PreNorm_3(dim, Attention_mydecoder(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                # PreNorm_3(dim, Attention_DECODER(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        self.apply(self._convert_weights_to_fp16)
    def forward(self, x, y,kv_mask = None,q_mask = None):
        # for attn, attn_decode, ff in self.layers:
        #     # x = attn(x, mask = q_mask)
        #     x = attn_decode(x, y , kv_mask=kv_mask,q_mask = q_mask)
        #     x = ff(x)
        for attn_decode, ff in self.layers:
            # x = attn(x, mask = q_mask)
            x = attn_decode(x, y , kv_mask=kv_mask,q_mask = q_mask)
            x = ff(x)
        return x

    def _convert_weights_to_fp16(self, l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.LayerNorm)):
            if l.weight.data is not None:
                l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
class mydecoder(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        # self.to_DIM_txt= nn.Linear(patch_dim, dim)
        # self.to_DIM_img = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer_mydecoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, txt,img, kv_mask = None,q_mask = None): # img [64,48,2048] txt [64,Length,2048]
        # img = self.to_DIM_img(img)
        # txt = self.to_DIM_txt(txt)
        b_img, n_img, _ = img.shape
        b_txt, n_txt, _ = txt.shape
        # x += self.pos_embedding[:, :(n + 1)]
        # img = self.dropout(img)
        x = self.transformer(txt, img, kv_mask,q_mask)
        # x = self.to_latent(x)
        # return self.mlp_head(x)
        return x