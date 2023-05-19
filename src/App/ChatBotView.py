from tkinter import ttk
import tkinter as tk
import asyncio
import threading

class ChatBotView(ttk.Frame):

    def __init__(self, username, async_loop):
        super(ChatBotView, self).__init__()
        self.controller = None
        self.username = username
        self.async_loop = async_loop


        self.txt = tk.Text(self, width=60)
        self.txt.grid(row=1, column=0, columnspan=2)
        self.scrollbar = tk.Scrollbar(self.txt)
        self.scrollbar.place(relheight=1, relx=0.974)

        self.e = tk.Entry(self, width=45)
        self.e.grid(row=2, column=0)

        tk.Button(self, text='Send Msg', command=self.send_button).grid(row=2, column=1)
        self.feed_back = tk.Entry(self, width=5)
        self.feed_back.grid(row=2, column=2)

        self.popup_entry = None

    def set_controller(self, controller):
        self.controller = controller

    def send_button(self):
        if self.controller:
            args = (self.e.get(), self.txt.get('1.0', 'end-1c'), self.feed_back.get())
            threading.Thread(target=self._asyncio_thread, args=args).start()

    def _asyncio_thread(self, entry, history, feedback):
        self.async_loop.run_until_complete(self.controller.send(entry, history, feedback))

    async def handle_response(self, response):
        self.txt.insert(tk.END, response)
        self.txt.see('end')

    def request_feedback(self):
        top = tk.Toplevel(self)
        top.geometry("750x250")
        top.title("Need Feedback")
        label = ttk.Label(top, text="I'm trying to get better\n Could you please provide feedback?")
        label.grid(row=1, column=0, columnspan=2)

        self.popup_entry = ttk.Entry(top, width=5)
        self.popup_entry.grid(row=2, column=0)
        ttk.Button(top, text="Submit Feedback", command=self.send_feedback).grid(row=2, column=1)

    def send_feedback(self):
        self.e.insert(tk.END, self.popup_entry.get())
