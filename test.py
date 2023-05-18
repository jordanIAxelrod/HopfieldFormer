import hflayers
import transformers
print(hflayers.Hopfield(15))
tok = transformers.GPT2Tokenizer.from_pretrained('gpt2')
mod = transformers.GPT2Model.from_pretrained('gpt2')
print(mod)
print([name for name, _ in mod.named_children()])
print(mod(**tok('Once upon a time, ',return_tensors='pt')))

