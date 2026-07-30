import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.physics.units import current


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 384,
            num_heads: int = 6,
            dropout: float = 0.1):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, casual_mask: torch.Tensor) -> torch.Tensor:

        x_norm = self.layer_norm1(x)
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=casual_mask,
            is_causal = False
        )

        x = x + attn_output
        x = self.layer_norm2(x)
        x = self.mlp(x) + x
        return x

class GPT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 dropout: float = 0.1,
                 num_layers: int = 6,
                 block_size: int = 256):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

        casual_mask = torch.triu(
            torch.ones(
                block_size, block_size, dtype=torch.bool
            ),
            diagonal=1
        )

        self.register_buffer(
            'casual_mask', casual_mask)

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model created with {total_params:,} parameters")

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                input_ids: torch.Tensor,
                targets: torch.Tensor = None)->tuple:

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + pos_emb)

        mask = self.casual_mask[:seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, mask)

        x = self.layer_norm3(x)

        logits = self.output_projection(x)

        loss = None
        if targets is not None:

            batch_size, seq_len, vocab_size = logits.shape

            logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
            targets_flat = targets.reshape(batch_size * seq_len)

            loss = self.loss_fn(logits_flat, targets_flat)

        return logits, loss


    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):

        for _ in range(max_new_tokens):
            if input_ids.size(1) <= self.block_size:
                current_input = input_ids
            else:
                current_input = input_ids[:, -self.block_size:]

            logits, _ = self.forward(current_input)
            last_logits = logits[:, -1, :]
            last_logits = last_logits / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create model
    model = GPT(
        vocab_size=65,
        embedding_dim=384,
        num_heads=6,
        num_layers=6,
        block_size=256
    ).to(device)

    # 3. Create dummy input
    batch_size = 4
    seq_len = 64
    dummy_input = torch.randint(0, 65, (batch_size, seq_len)).to(device)
    dummy_targets = torch.randint(0, 65, (batch_size, seq_len)).to(device)

    # 4. Test forward pass
    logits, loss = model(dummy_input, dummy_targets)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # 5. Test generation
    generated = model.generate(dummy_input[:1, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")