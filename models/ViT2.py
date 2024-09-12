import torch, math
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class PatchEmbedding(nn.Module):
    #def __init__(self, in_channels=1, patch_size=16, embed_size=256, img_size=224):
    def __init__(self, in_channels=1, patch_size=6, embed_size=36, img_size=28):
        num_patches=(img_size // patch_size) ** 2
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.positions = nn.Parameter(torch.randn(num_patches + 1, embed_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)  # embed_size / num_heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads=8):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Linear layers to project query, key, and value
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

        # 멀티헤드 결과 결합
        self.out_linear = nn.Linear(embed_size, embed_size)

        self.attention = ScaledDotProductAttention()

    def forward(self, query):
        batch_size = query.size(0)

        # Project query, key, and value
        query = self.query_linear(query)
        key = self.key_linear(query)
        value = self.value_linear(query)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = self.attention(query, key, value)

        # Concatenate attention outputs from all heads
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, -1, self.num_heads * self.head_dim)

        # final linear transformation
        output = self.out_linear(attn_output)
        return output

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 16,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 16, n_classes: int = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 16,
                 emb_size: int = 256,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 10,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

if __name__ == "__main__":
    model = ViT()
    summary(model.cuda(), (1, 224, 224))


