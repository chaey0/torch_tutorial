import math
import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    #def __init__(self, in_channels=1, patch_size=16, embed_size=256, img_size=224):
    def __init__(self, in_channels=1, patch_size=4, embed_size=16, img_size=28):
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
    def __init__(self, embed_size, num_heads):
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

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Project query, key, and value
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = self.attention(query, key, value)

        # Concatenate attention outputs from all heads
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, -1, self.num_heads * self.head_dim)

        # final linear transformation
        output = self.out_linear(attn_output)
        return output


class EncoderBlock(nn.Module):
    #def __init__(self, embed_size=256, num_heads=8, mlp_ratio=4, dropout=0.1):
    def __init__(self, embed_size=16, num_heads=4, mlp_ratio=4, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = MultiHeadAttention(embed_size, num_heads)
        self.ln2 = nn.LayerNorm(embed_size)

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, mlp_ratio * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Multi-Head Attention과 잔차 연결
        x = self.ln1(x)
        x = x + self.mha(x, x, x)[0]
        x = self.ln2(x)

        # MLP와 잔차 연결
        x = x + self.mlp(x)
        return x

class VisionTransformer(nn.Module):
    #def __init__(self, in_channels=1, num_classes=10, embed_size=256, img_size=224, patch_size=16, num_layers=6, num_heads=8):

    def __init__(self, in_channels=1, num_classes=10, embed_size=16, img_size=28, patch_size=4, num_layers=6,num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_size, img_size)
        self.encoders = nn.ModuleList([EncoderBlock(embed_size, num_heads) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_size)
        self.mlp_head = nn.Linear(embed_size, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # 패치 임베딩 + 클래스 토큰 + 포지션 임베딩
        x = self.patch_embed(x)

        # 여러 개의 Transformer Encoder Block 통과
        for encoder in self.encoders:
            x = encoder(x)

        # 클래스 토큰
        x = self.ln(x[:, 0])

        # 최종 분류
        x = self.mlp_head(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":

    model = VisionTransformer()
    summary(model.cuda(), (1, 28, 28))
