from transformers import GPT2Model

import torch.nn as nn


class VNetwork(nn.Module):

    def __init__(self):
        super(VNetwork, self).__init__()
        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.gpt.h = self.gpt.h[:1]
        self.head = nn.Linear(self.gpt.config.n_embd, 1)

    def forward(self, msg):
        gpt_out = self.gpt(**msg).last_hidden_state.sum(dim=1)
        return self.head(gpt_out)

if __name__ == '__main__':
    v = VNetwork()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    inp = tokenizer("We're all in this together", return_tensors='pt')

    print(v(inp))
