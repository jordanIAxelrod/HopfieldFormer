"""
Make conversation Dataset from the OpenAssistaint Dataset

Create a dataset with the conversation questions that I created.

@author Jordan Axelrod
"""

from torch.utils.data import Dataset
from collections.abc import Iterable
import numpy as np
import pandas as pd
from src.App import utils
from datasets import load_dataset


class OpenAssistantDataset(Dataset):
    def __init__(self, split, url=None):
        if url is not None:
            self.all_conversations = np.load(url)
        else:
            self.orig_dataset = load_dataset("OpenAssistant/oasst1")[split]
            self.all_conversations = np.array(self.conversation_creater())
            np.save(f'../data/OpenAssistantConversations{split}', self.all_conversations)
            pd.DataFrame(self.all_conversations).to_csv(f"../data/OpenAssistantConversations{split}.csv")



    def conversation_creater(self) -> list:
        parents = [i for i, x in enumerate(self.orig_dataset['parent_id']) if x is None and self.orig_dataset[i]['lang'] == 'en']
        conversations = []
        for i, idx in enumerate(parents):

            parent = self.orig_dataset[idx]

            conversation = [parent]
            if i == len(parents) - 1:
                tree_dataset = self.orig_dataset.select(range(idx, self.orig_dataset.num_rows))
            else:
                tree_dataset = self.orig_dataset.select(range(idx, parents[i + 1]))
            conversation = self._conversation_creater(parent["message_id"], conversation.copy(), tree_dataset)
            conversations.extend(conversation)
        return conversations

    def _conversation_creater(self, parent_id, message, dataset):
        children = dataset.filter(lambda example: example['parent_id'] == parent_id, load_from_cache_file=False, keep_in_memory=True)
        if len(children) == 0:
            return [utils.format_message(message)]
        conversation = []
        for i in range(children.num_rows):
            parent = children[i]
            new_message = message.copy()
            new_message.append(parent)
            conversation.extend(self._conversation_creater(parent["message_id"], new_message, dataset))
        return conversation

    def __len__(self):
        return self.all_conversations.shape[0]

    def __getitem__(self, index):
        return {'text': self.all_conversations[index]}



def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

if __name__ == '__main__':
    OpenAssistantDataset('train')
    OpenAssistantDataset('validation')
