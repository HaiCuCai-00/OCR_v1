import os
import sys

import numpy as np
import torch
from loguru import logger
from torch.nn.functional import log_softmax, softmax

from models.ocr import OCR
from utils.get_path import get_path
from utils.image_processing import ocr_input_processing

from .vocab import Vocab


class TextClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vocab = Vocab(cfg.vocab)
        self.model = self.build_model()

    def build_model(self):
        device = self.cfg.device
        model = OCR(len(self.vocab), self.cfg)
        model = model.to(device)
        weights_dir = get_path(self.cfg.model.weights)
        model.load_state_dict(
            torch.load(weights_dir, map_location=torch.device(device))
        )
        return model

    def predict(self, inputs, return_prob=False):
        inputs = inputs.to(self.cfg.device)

        if self.cfg.model.beam_search:
            sent = self.predict_beam_search(inputs)
            s = sent
            prob = None
        else:
            s, prob = self.predict_normal(inputs)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)

        if return_prob:
            return s, prob
        else:
            return s

    def predict_beam_search(
        self,
        inputs,
        beam_size=4,
        candidates=1,
        max_seq_length=128,
        sos_token=1,
        eos_token=2,
    ):
        # img: BxCxHxW
        self.model.eval()
        device = inputs.device
        sents = []
        with torch.no_grad():
            src = self.model.cnn(inputs)
            memory = self.model.transformer.forward_encoder(src)  # TxNxE
            for i in range(src.size(0)):
                memory = self.model.transformer.get_memory(memories, i)
                sent = beamsearch(
                    memory,
                    self.model,
                    device,
                    beam_size,
                    candidates,
                    max_seq_length,
                    sos_token,
                    eos_token,
                )
                sents.append(sent)
        sents = np.asarray(sents)
        return sents

    def predict_normal(self, inputs, max_seq_length=128, sos_token=1, eos_token=2):
        "data: BxCxHxW"
        self.model.eval()
        device = inputs.device

        with torch.no_grad():
            src = self.model.cnn(inputs)
            memory = self.model.transformer.forward_encoder(src)

            translated_sentence = [[sos_token] * len(inputs)]
            char_probs = [[1] * len(inputs)]

            max_length = 0

            while max_length <= max_seq_length and not all(
                np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
            ):
                tgt_inp = torch.LongTensor(translated_sentence).to(device)
                output, memory = self.model.transformer.forward_decoder(tgt_inp, memory)
                output = softmax(output, dim=-1)
                output = output.to("cpu")

                values, indices = torch.topk(output, 5)

                indices = indices[:, -1, 0]
                indices = indices.tolist()

                values = values[:, -1, 0]
                values = values.tolist()
                char_probs.append(values)

                translated_sentence.append(indices)
                max_length += 1

                del output

            translated_sentence = np.asarray(translated_sentence).T
            char_probs = np.asarray(char_probs).T
            char_probs = np.multiply(char_probs, translated_sentence > 3)
            char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)
        return translated_sentence, char_probs
