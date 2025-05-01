Input x               : (B, 3, 512, 512)
↓ ConvNeXt
feat_map              : (B, 1024, 16, 16)
↓ Flatten + Permute
tokens                : (B, 256, 1024)
square_tokens         : (64, 1024)
↓ Expand across batch
queries               : (B, 64, 1024)
↓ Concatenate
full_input            : (B, 320, 1024)
↓ Transformer
encoded[:64]          : (B, 64, 1024)
↓ Linear Layer
logits                : (B, 64, 13)
↓ Argmax
preds                 : (B, 64)