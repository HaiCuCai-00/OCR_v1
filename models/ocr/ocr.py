from torch import nn

from .backbone.cnn import CNN
from .seqmodel.transformer import LanguageTransformer


class OCR(nn.Module):
    def __init__(self, vocab_size, cfg):
        super(OCR, self).__init__()
        self.cfg = cfg
        self.cnn = CNN(cfg)
        self.seq_modeling = cfg.model.seq_modeling

        if self.seq_modeling == "transformer":
            self.transformer = LanguageTransformer(vocab_size, cfg)
        else:
            raise ("Seq model is not supported!")

    def forward(self, img, tgt_input, tgt_key_padding_mask):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        outputs = self.transformer(
            src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask
        )
        return outputs
