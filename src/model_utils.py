from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn


class NBMEModel(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.config = AutoConfig.from_pretrained(checkpoint, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(self.config, 'initializer_range'):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            else:
                module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if hasattr(self.config, 'initializer_range'):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            else:
                module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, **inputs):
        outputs = self.backbone(**{k: v for k, v in inputs.items() if k != 'label'})
        sequence_output = outputs[0]
        logits = self.classifier(self.dropout(sequence_output))
        loss = None
        if 'label' in inputs:
            loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            loss = loss_fct(logits.view(-1, 1), inputs['label'].view(-1, 1).float())
            loss = torch.masked_select(loss, inputs['label'].view(-1, 1) != -100).mean()

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoTokenizer, DataCollatorForTokenClassification
    from data_utils import NBMEDataset

    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
    df = pd.read_pickle('../data/train_processed.pkl')
    dataset = NBMEDataset(tokenizer, df)
    collator = DataCollatorForTokenClassification(tokenizer)
    batch = collator.torch_call([dataset[i] for i in range(16)])
    print(list(batch.keys()))
    model = NBMEModel('microsoft/deberta-base')
    output = model(**batch)
    # print(output.logits.shape)
    # print(output.loss)
