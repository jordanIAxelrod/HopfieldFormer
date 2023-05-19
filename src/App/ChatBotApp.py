import tkinter as tk
from tkinter import ttk

from ModelPipeline import ModelPipeline
from ChatBotView import ChatBotView
from ChatBotController import ChatBotController
import asyncio

import torch

class App(tk.Tk):
    def __init__(self, async_loop):
        super(App, self).__init__()
        self.title('HopChat')
        self.username = 'Jordan Axelrod'
        print(1)
        # Create the model

        self.model = ModelPipeline(
            '../models/OAHopfieldNetwork.pth', # Standard Hopfield
            f'../models/{self.username} HopfieldNetwork.pth', # User Hopfield
            accum_grad=True,
            batch_size=5
        )
        print(2)
        # Create the view and place it on the root window
        view = ChatBotView(username, async_loop)
        view.grid(row=0, column=0, padx=10, pady=10)
        print(3)
        # create a controller
        controller = ChatBotController('Jordan Axelrod', view, self.model)

        view.set_controller(controller)

    def on_closing(self):
        torch.save(self.model.model.state_dict(), f'../models/{self.username} HopfieldNetwork.pth')
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    app = App(loop)
    app.mainloop()
