import tkinter as tk
from tkinter import ttk

from ModelPipeline import ModelPipeline
from ChatBotView import ChatBotView
from ChatBotController import ChatBotController
import asyncio


class App(tk.Tk):
    def __init__(self, async_loop):
        super(App, self).__init__()
        self.title('HopChat')
        print(1)
        # Create the model
        model = ModelPipeline('../models/BaseHopfieldNetwork.pth', accum_grad=True, batch_size=5)
        print(2)
        # Create the view and place it on the root window
        view = ChatBotView('Jordan Axelrod', async_loop)
        view.grid(row=0, column=0, padx=10, pady=10)
        print(3)
        # create a controller
        controller = ChatBotController('Jordan Axelrod', view, model)

        view.set_controller(controller)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    app = App(loop)
    app.mainloop()
