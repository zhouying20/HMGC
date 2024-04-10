import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
)


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False, device="cuda"):
        super(Attention, self).__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        stdv = 1.0 / np.sqrt(self.hidden_size)
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.device = device

        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)
        mask = torch.ones(attentions.size(), requires_grad=True)
        if self.device == "cuda":
            mask = mask.cuda()
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)
        attentions = masked.div(_sums)
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        return weighted.sum(1), attentions


class AttenLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, batch_first=True, dropout=0.5, bidirectional=True,
                 num_layers=2, device="gpu"):
        super(AttenLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             bidirectional=bidirectional,
                             batch_first=batch_first)
        self.atten1 = Attention(hidden_size * 2, batch_first=batch_first, device=device)
        self.lstm2 = nn.LSTM(input_size=hidden_size * 2,
                             hidden_size=hidden_size,
                             num_layers=1,
                             bidirectional=bidirectional,
                             batch_first=batch_first)
        self.atten2 = Attention(hidden_size * 2, batch_first=batch_first, device=device)
        self.fc = nn.Linear(hidden_size * num_layers * 2, 2)

    def forward(self, x):
        out1, (_, _) = self.lstm1(x)
        x, _ = self.atten1(out1)
        out2, (_, _) = self.lstm2(out1)
        y, _ = self.atten2(out2)

        z = torch.cat([x, y], dim=1)
        z = self.fc(self.dropout(z))
        return z


class CheckGPTDetectorForPipeline(RobertaPreTrainedModel):
    def __init__(self, config, device="gpu"):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = AttenLSTM(
            input_size=1024, hidden_size=256,
            batch_first=True, dropout=0.5,
            bidirectional=True, num_layers=2, device=device
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
        )
        features = F.pad(outputs.last_hidden_state, (0, 0, 0, 512 - outputs.last_hidden_state.size(1)))
        logits = self.classifier(features)
        return logits


if __name__ == '__main__':
    pass
    # config = RobertaConfig.from_pretrained("roberta-large", num_labels=2)
    # m = RobertaModel(config).cuda()
    # print(get_model_complexity_info(m, (512,), as_strings=True, print_per_layer_stat=True))