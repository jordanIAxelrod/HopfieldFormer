import hflayers
import torch.nn as nn
from transformers import GPT2PreTrainedModel


class HopfieldGPTBlock(nn.Module):

    def __init__(self, GPTBlock, embed_dim):
        super(HopfieldGPTBlock, self).__init__()
        self.GPTBlock = GPTBlock
        self.hopfield = hflayers.HopfieldLayer(embed_dim)

    def freeze_gpt(self):
        for param in self.GPTBlock.parameters():
            param.requires_grad = False

    def unfreeze_gpt(self):
        for param in self.GPTBlock.parameters():
            param.requires_grad = True

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False
    ):
        memory = self.hopfield(hidden_states)
        transformer = self.GPTBlock(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        hidden_states = memory + transformer[0]
        return hidden_states, *transformer[1:]


class HopfieldFormer(nn.Module):

    def __init__(self, GPTModel: GPT2PreTrainedModel, n_hopfield: int):
        super().__init__()
        self.GPTModel = GPTModel
        self.embed_dim = self.GPTModel.config.n_embd
        self.n_hopfield = n_hopfield
        try:
            self.depth = len(self.GPTModel.h)
        except AttributeError:
            self.depth = len(self.GPTModel.base_model.h)
        # Replace the top n GPT2Blocks with Hopfield-GPT Blocks
        for i in range(self.depth):
            if self.depth - i <= n_hopfield:
                try:
                    self.GPTModel.h[i] = HopfieldGPTBlock(self.GPTModel.h[i], self.embed_dim)
                except AttributeError:
                    self.GPTModel.base_model.h[i] = HopfieldGPTBlock(self.GPTModel.base_model.h[i], self.embed_dim)

    def freeze_gpt(self):
        for i in range(self.depth):
            if self.depth - i <= self.n_hopfield:
                try:
                    self.GPTModel.h[i].freeze_gpt()
                except AttributeError:
                    self.GPTModel.base_model.h[i].freeze_gpt()

    def unfreeze_gpt(self):
        for i in range(self.depth):
            if self.depth - i <= self.n_hopfield:
                try:
                    self.GPTModel.h[i].unfreeze_gpt()
                except AttributeError:
                    self.GPTModel.base_model.h[i].unfreeze_gpt()

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        out = self.GPTModel(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return out
