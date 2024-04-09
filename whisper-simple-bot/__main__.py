import io
import logging
import traceback
import html
import json
import os

import openai_utils

import telegram
from telegram import (
    Update,
    BotCommand
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters
)
from telegram.constants import ParseMode, ChatAction

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)

# check if environment variables OPENAI_API_KEY, TELEGRAM_BOT_TOKEN and ALLOWED_TELEGRAM_USERNAMES are set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

if not os.getenv("TELEGRAM_BOT_TOKEN"):
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

if not os.getenv("ALLOWED_TELEGRAM_USERNAMES"):
    raise ValueError("ALLOWED_TELEGRAM_USERNAMES environment variable is not set")


allowed_telegram_usernames=os.getenv("ALLOWED_TELEGRAM_USERNAMES", "").split(",")

HELP_MESSAGE = """Commands:
⚪ /help – Show help
"""

def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

async def is_bot_mentioned(update: Update, context: CallbackContext):
     try:
         message = update.message

         if message.chat.type == "private":
             return True

         if message.text is not None and ("@" + context.bot.username) in message.text:
             return True

         if message.reply_to_message is not None:
             if message.reply_to_message.from_user.id == context.bot.id:
                 return True
     except:
         return True
     else:
         return False


async def voice_message_handle(update: Update, context: CallbackContext):
    logging.info("Received voice")
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    user_id = update.message.from_user.id

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(buf)
    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def audio_handle(update: Update, context: CallbackContext):
    logging.info("Received audio")
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    user_id = update.message.from_user.id

    audio = update.message.audio
    audio_file = await context.bot.get_file(audio.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await audio_file.download_to_memory(buf)
    buf.name = audio.file_name  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(buf)
    text = f"<b>{audio.file_name}</b> 🔊: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(update.effective_chat.id, "Some error in error handler")

async def help_handle(update: Update, context: CallbackContext):
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/help", "Show help message"),
    ])

async def unsupported_message_handle(update: Update, context: CallbackContext, message=None):
    error_text = f"I don't know how to read files or videos. Send the picture in normal mode (Quick Mode)."
    logger.error(error_text)
    await update.message.reply_text(error_text)
    return

async def message_handle(update: Update, context: CallbackContext):
    logger.info("Received message")
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    user_id = update.message.from_user.id
    text = update.message.text

    await update.message.reply_text(f"📝: <i>{text}</i>", parse_mode=ParseMode.HTML)

def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(os.environ.get("TELEGRAM_BOT_TOKEN"))
        .concurrent_updates(True)
        # .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .get_updates_write_timeout(20)
        .post_init(post_init)
        .build()
    )

    # add handlers
    user_filter = filters.ALL
    if len(allowed_telegram_usernames) > 0:
        usernames = [x for x in allowed_telegram_usernames if isinstance(x, str)]
        any_ids = [x for x in allowed_telegram_usernames if isinstance(x, int)]
        user_ids = [x for x in any_ids if x > 0]
        group_ids = [x for x in any_ids if x < 0]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids) | filters.Chat(chat_id=group_ids)

    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))

    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))

    application.add_handler(MessageHandler(filters.AUDIO & ~filters.COMMAND & user_filter, audio_handle))

    application.add_error_handler(error_handle)

    # start the bot
    application.run_polling(timeout=30)

if __name__ == "__main__":
    logger.info("Allowed usernames: %s", allowed_telegram_usernames)
    run_bot()
