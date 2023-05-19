"""
This class handles creating responses in the chatbot and implementing RLHF and continuous learning

"""
from __future__ import annotations

import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel
import src.training.HopfieldFormer as hf
import VNetwork
import numpy as np


class RLTrainingBuffer:
    def __init__(self, msg_t0=None):
        self.msg_t0 = msg_t0
        self.msg_t1 = None
        self.reward_t1 = None
        self.input = None

    def get_preds(self, model, time):
        if time:
            return model(**self.msg_t1)
        return model(**self.msg_t0)

    def get_V(self, model, time):
        if time:
            return model(self.msg_t1)
        return model(self.msg_t0)


class ModelPipeline:

    def __init__(self, model_path, user_model, accum_grad=False, batch_size=None, gamma: float = .1):
        """
        Initializes the model of the Chatbot app following a MVC structure
        :param model_path: path to default model
        :param user_model: path to user specific model
        :param accum_grad: whether we update everytime
        :param batch_size: if we accumulate for how much
        :param gamma: future discount
        """
        self.model = self.load_model(user_model, model_path)

        self.accum_grad = accum_grad
        self.batch_size = batch_size
        self.gamma = gamma

        # Reinforcement learning state variables.
        self.V = VNetwork.VNetwork()
        self.V_optimizer = torch.optim.Adam(self.V.parameters())
        self.training_buffer = [RLTrainingBuffer() for _ in range(3)]

        if self.accum_grad:
            assert self.batch_size is not None, 'Must specify a batch size if accumulating gradients'

        self.model.freeze_gpt()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.eos_token_id = self.tokenizer.encode(self.tokenizer.eos_token)[0]
        print(self.eos_token_id)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.V.to(self.device)
        self.step = 0

    def load_model(self, url_user, url_standard):
        """
        Loads the model
        :param url_user: user specific model
        :param url_standard: standard model
        :return: the loaded Hopfield Model
        """
        try:
            model_state_dict = torch.load(url_user)
        except FileNotFoundError:
            model_state_dict = torch.load(url_standard)
        gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        model = hf.HopfieldFormer(gpt, 4)
        model.load_state_dict(model_state_dict)
        return model

    async def generate_response(self, msg: str, username: str):
        """
        Generates a response given the input has a prompt that is put in front of the chat history
        :param msg: the chat history
        :param username: the username of the user
        :return: the response as a string
        """
        # Generate the response
        prompt = f"Good morning. You are a friendly chat bot. Your perogative is to answer the humans questions after" \
                 f" the '<ChatBot>:' start\n The following is an example.\n\n <{username}> Hi chat bot, can you " \
                 f"tell me who the lead singer of queen is?\n\n\n<ChatBot> Hi {username}, the lead signer of " \
                 f"queen is Freddie Mercury!"
        response = await self.probability_decode(msg, username, temperature=.8)
        print('response Generated')
        return response

    async def train_network(self, msg: str, rlhf_input: float, response: str):
        """
        Runs the online reinforcement learning using A2C
        :param msg: the chat history
        :param rlhf_input: the reward for the previous response
        :param response: the generated response from the chat history
        :return: None
        """
        # Set the network to train only the hopfield networks again
        self.model.train()
        self.model.freeze_gpt()
        # Get the probabilities for these output
        response_tokenize, tokens_over = await self.front_truncate(msg + response)

        if tokens_over > 0:
            inp = self.front_truncate(msg, tokens_over)
        else:
            inp = self.tokenizer(msg, return_tensors='pt')
        # A2C Loss
        loss = await self.a2c(response_tokenize, rlhf_input, inp)

        if self.step > 1:
            loss.backward()
        if not self.accum_grad or self.step % self.batch_size == 0:
            await self.optim_step()

        # Else we accumulate the gradients
        self.step += 1

    async def front_truncate(self, msg, length=-1024):
        """
        Removes the front of the list of tokens to truncate to desired size
        :param msg: the string to truncate
        :param length: truncation length
        :return:
        """
        tokenize = self.tokenizer.tokenize(msg)
        tokens_over = len(tokenize) + length
        if tokens_over > 0:
            tokenize = tokenize[length:]
        string = self.tokenizer.convert_tokens_to_string(tokenize)
        return self.tokenizer(string, return_tensors='pt').to(self.device), tokens_over

    async def a2c(self, response_tokenize, rlhf_input, inp) -> torch.Tensor:
        """
        The meat of the a2c algorithm. Helper method for train_network
        :param response_tokenize: the response tokenized
        :param rlhf_input: the reward for the previous response
        :param inp: the chat history
        :return: the actor critic average reward given the V network the response and the reward
        """
        print(self.step)
        new_buffer = RLTrainingBuffer(response_tokenize)
        new_buffer.input = inp
        self.training_buffer[self.step % 3] = new_buffer
        if self.step == 0:
            return torch.zeros(1)
        if self.step >= 1:
            self.training_buffer[(self.step - 1) % 3].msg_t1 = response_tokenize
            a2c = torch.zeros(0)
        if self.step >= 2:
            buffer = self.training_buffer[(self.step - 2) % 3]
            buffer.reward_t1 = rlhf_input
            preds = buffer.get_preds(self.model, 0).logits
            response_start = buffer.input['input_ids'].shape[1]
            response_preds = preds[:, response_start - 1: -1]
            y = buffer.msg_t0['input_ids'][:, response_start:]

            ce = nn.functional.cross_entropy(response_preds.squeeze(), y.squeeze())
            advantage = float(buffer.reward_t1) + self.gamma * buffer.get_V(self.V, 1) - buffer.get_V(self.V, 0)
            a2c = ce * advantage
        return a2c

    async def optim_step(self):
        """
        Steps all the optimizers and zeros them
        :return: None
        """
        self.optimizer.step()
        self.V_optimizer.step()
        self.V_optimizer.zero_grad()
        self.optimizer.zero_grad()

    async def probability_decode(self, msg: str, username: str, temperature: float = .5, length_pen: float = 1.):
        """
        Uses a greedy decode by sampling the stochastic distribution produced by the model
        :param msg: chat history
        :param username: user's name
        :param temperature: how to scale the distribution
        :param length_pen: penalty for length
        :return: the greedy response as a string
        """
        is_reached = False
        iteration = 0
        self.model.eval()
        response = ''
        while not is_reached:
            msg_tokenized, tokens_over = await self.front_truncate(msg)
            output = self.model(**msg_tokenized).logits[:, -1]
            output[:, -1] *= length_pen ** iteration
            output = nn.functional.softmax(output / temperature, dim=-1)
            action = np.random.choice(output.shape[-1], p=output[0].cpu().detach().numpy())
            word = self.tokenizer.decode(action)
            msg += word
            response += word
            is_eos_token = word == self.tokenizer.eos_token == word
            is_too_long = iteration >= self.tokenizer.model_max_length // 5
            is_human_last_token = msg.endswith(f'<{username}>')
            if is_too_long or is_eos_token or is_human_last_token:
                is_reached = True
            iteration += 1
        return response

    def beam_response(self, msg, beam_size=3):
        """
        Uses a beam decode. Much slower
        :param msg: chat history
        :param beam_size: amount of beams to use
        :return: the best response
        """
        self.model.eval()
        msg = self.tokenizer(msg, return_tensors='pt').to(self.device)
        V = len(self.tokenizer.get_vocab())

        hyps = [Hyp(None, None, 0, self.tokenizer.eos_token)]
        derivations = []

        current_states = [0 for _ in range(beam_size)]
        finished_beams = 0

        for t in range(self.tokenizer.model_max_length // 5):
            xs = []
            with torch.no_grad():
                for i, hyp in enumerate(hyps):
                    if hyp.token is None:
                        output = nn.functional.log_softmax(self.model(**msg).logits, dim=-1)
                        output = output[:, -1, :] + hyp.score
                        current_states.append(self.tokenizer.decode(msg.input_ids.tolist()[0]))
                    else:
                        new_msg = current_states[beams[i]] + f' {hyp.token}'
                        new_msg = self.tokenizer(new_msg, return_tensors='pt').to(self.device)
                        output = nn.functional.log_softmax(self.model(**new_msg).logits, dim=-1)
                        output = output[:, -1, :] + hyp.score
                        xs.append(output)
                        current_states.append(self.tokenizer.decode(new_msg.input_ids.tolist()[0]))

                del current_states[:-len(hyps)]
                if hyps[0].token is not None:
                    output = torch.cat(xs, dim=0)
                output = output.squeeze()

                scores, beam = torch.topk(output.flatten(), beam_size - finished_beams)
                words = beam % V
                beams = beam // V
                hyps = [
                    Hyp(
                        self.tokenizer.decode(words[i]),
                        hyps[beams[i]],
                        scores[i].item(),
                        self.tokenizer.eos_token
                    )
                    for i in range(beam_size - finished_beams)
                ]

                if self.eos_token_id in words:
                    take_outs = []
                    for i, word in enumerate(words):
                        if word == self.eos_token_id:
                            derivations.append(hyps[i])
                            take_outs.append(i)
                    for j, i in enumerate(take_outs):
                        del hyps[i - j]
                        finished_beams += 1

                if finished_beams == beam_size:
                    break

        return max(hyps, key=lambda y: y.score)

    async def __call__(self, msg: str, rlhf_input: int = 0):
        full_text = await self.generate_response(msg, rlhf_input)


# make


class Hyp:

    def __init__(self, token: str, parent: Hyp, score: float, eos_token: int):
        """
            args:
              token: a word (as a string) representing the most recent word added to this hypothesis
              parent: the Hyp object representing the prefix to which we've added this token
              score: the score of this hypothesis
            """
        self.token = token
        self.parent = parent
        self.score = score
        self.eos_token = eos_token

    def trace(self):
        """
        Traces backward through the linked-list to recover the whole hypothesis.
        returns:
          A list of word tokens representing the entire hypothesis, WITHOUT an EOS.
          (This means you can use this function to create a list of words to be passed
          to the constructor of a Derivation object, which similarly does not want an EOS
          at the end).
        """
        pred = []
        temp = self.parent if self.token == self.eos_token else self
        while temp.token is not None:
            pred.append(temp.token)
            temp = temp.parent
        return pred[::-1]


if __name__ == '__main__':
    pipe = ModelPipeline('../../models/BaseHopfieldNetwork.pth')
    msg1 = 'What prompt would make you the best chat bot? Please make it succinct'
    # x = pipe.beam_response(msg, beam_size=1)
    # print(msg + ''.join(x.trace()))
    x = pipe.probability_decode(msg1, temperature=.8)
    print(x)
