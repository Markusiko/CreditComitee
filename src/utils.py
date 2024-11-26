from IPython.display import display, Markdown
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram import Update
import asyncio


def print_md(text):
    display(Markdown(text))


def log_output_to_telegram(func):
    async def wrapper(state):
        output = func(state)

        # Отправка сообщений через Telegram-бота
        bot = state.get("bot")
        chat_id = state.get("chat_id")
        if bot and chat_id:
            await bot.send_message(
                chat_id=chat_id,
                text=f"Node: {func.__name__}\nOutput: {output}"
            )
        return output

    return wrapper


