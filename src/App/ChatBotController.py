"""
This file contains the window for the chat bot. it includes message history, a chat box and a scoring system

"""

import tkinter
from ModelPipeline import ModelPipeline
from ChatBotView import ChatBotView


class ChatBotController:
    def __init__(self, username, view: ChatBotView, model: ModelPipeline):
        self.username = username
        self.model = model
        self.view = view

    async def send(self, message, text, feedback):
        # Format the incoming message
        send = f'\n\n\n<{self.username}>: {message}\n\n\n<ChatBot>:'

        # Add it to the text box
        self.view.txt.insert(tkinter.END, send)

        # Make sure we got feedback
        feedback = await self.ensure_feedback(feedback)

        # Get a response
        full_text = text + send
        response = await self.model.generate_response(full_text, self.username)
        # Send response to the view
        await self.view.handle_response(response)
        await self.model.train_network(send, feedback, response)

    async def ensure_feedback(self, feedback):
        if feedback != '':
            return feedback
        self.view.request_feedback()


if __name__ == '__main__':
    ChatBotController('Jordan')
