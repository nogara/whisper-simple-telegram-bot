import io
import logging
import traceback
import html
import json
import os
import re
import tempfile
import glob

from . import openai_utils

from pydub import AudioSegment
import yt_dlp

import telegram
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram.constants import ParseMode, ChatAction

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")],
)

logger = logging.getLogger(__name__)

# check if environment variables OPENAI_API_KEY, TELEGRAM_BOT_TOKEN and ALLOWED_TELEGRAM_USERNAMES are set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

if not os.getenv("TELEGRAM_BOT_TOKEN"):
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

if not os.getenv("ALLOWED_TELEGRAM_USERNAMES"):
    raise ValueError("ALLOWED_TELEGRAM_USERNAMES environment variable is not set")


allowed_telegram_usernames = os.getenv("ALLOWED_TELEGRAM_USERNAMES", "").split(",")

# Configurable settings directory
SETTINGS_DIR = os.getenv("CHAT_SETTINGS_DIR", ".")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "chat_settings.json")


def load_chat_settings():
    """Load chat settings from JSON file."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading chat settings: {e}")
    return {}


def save_chat_settings():
    """Save chat settings to JSON file."""
    try:
        # Ensure the settings directory exists
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(chat_settings, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat settings: {e}")


chat_settings = load_chat_settings()

HELP_MESSAGE = """Commands:
⚪ /help – Show help
⚪ /version – Show version
⚪ /show_settings – Show current settings
⚪ /set_language &lt;lang&gt; – Set transcription language (e.g., pt, en)
⚪ /set_model &lt;model&gt; – Set Whisper model (e.g., whisper-1)
⚪ /set_gpt_model &lt;model&gt; – Set GPT model (e.g., gpt-3.5-turbo, gpt-4)
⚪ /set_max_tokens &lt;number&gt; – Set max tokens for GPT response (e.g., 1000)
⚪ /set_temperature &lt;float&gt; – Set temperature for GPT response (e.g., 0.7)
⚪ /summarize – Summarize the replied transcription
⚪ /bullets – Create bullet points for the replied transcription
⚪ /translate &lt;language&gt; – Translate the replied transcription to the specified language
"""


def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def get_version_string():
    # get version from version file
    version = "unknown"
    try:
        with open("VERSION") as f:
            version = f.read().strip()
    except:
        pass
    return version


async def is_bot_mentioned(update: Update, context: CallbackContext):
    try:
        message = update.message
        if not message:
            return False

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
    if not update.message:
        return
    logging.info("Received voice")
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    chat_id = update.effective_chat.id
    settings = chat_settings.get(
        chat_id,
        {
            "language": "pt",
            "model": "whisper-1",
            "gpt_model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    )

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(
        buf, **{k: v for k, v in settings.items() if k in ["language", "model"]}
    )
    text = f"🎤: <i>{html.escape(transcribed_text)}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def audio_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    logging.info("Received audio")
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    chat_id = update.effective_chat.id
    settings = chat_settings.get(
        chat_id,
        {
            "language": "pt",
            "model": "whisper-1",
            "gpt_model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    )

    audio = update.message.audio
    audio_file = await context.bot.get_file(audio.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await audio_file.download_to_memory(buf)
    buf.name = audio.file_name  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(
        buf, **{k: v for k, v in settings.items() if k in ["language", "model"]}
    )

    if len(transcribed_text) < 4096:
        text = f"<b>{html.escape(audio.file_name)}</b> 🔊: <i>{html.escape(transcribed_text)}</i>"
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    # split text into multiple messages due to 4096 character limit
    i = 0
    for message_chunk in split_text_into_chunks(transcribed_text, 4096):
        try:
            i = i + 1
            text = f"<b>{html.escape(audio.file_name)}</b> 🔊 ({i}): <i>{html.escape(message_chunk)}</i>"
            await context.bot.send_message(
                update.effective_chat.id, text, parse_mode=ParseMode.HTML
            )
        except telegram.error.BadRequest:
            # answer has invalid characters, so we send it without parse_mode
            await context.bot.send_message(update.effective_chat.id, message_chunk)


async def video_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    logging.info("Received video")
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    chat_id = update.effective_chat.id
    settings = chat_settings.get(
        chat_id,
        {
            "language": "pt",
            "model": "whisper-1",
            "gpt_model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    )

    video = update.message.video
    video_file = await context.bot.get_file(video.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await video_file.download_to_memory(buf)
    buf.seek(0)  # move cursor to the beginning of the buffer

    # extract audio from video
    try:
        # determine format from mime_type or default to mp4
        mime_type = video.mime_type or "video/mp4"
        format = mime_type.split("/")[-1]  # e.g., 'mp4'
        audio_segment = AudioSegment.from_file(buf, format=format)

        # export audio to mp3 in memory
        audio_buf = io.BytesIO()
        audio_segment.export(audio_buf, format="mp3")
        audio_buf.name = "audio.mp3"
        audio_buf.seek(0)

        transcribed_text = await openai_utils.transcribe_audio(
            audio_buf,
            **{k: v for k, v in settings.items() if k in ["language", "model"]},
        )
    except Exception as e:
        logging.error(f"Error extracting audio from video: {e}")
        await update.message.reply_text("Error: failed to extract audio from video")
        return

    file_name = video.file_name or "video"

    if len(transcribed_text) < 4096:
        text = f"<b>{html.escape(file_name)}</b> 🎥: <i>{html.escape(transcribed_text)}</i>"
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    # split text into multiple messages due to 4096 character limit
    i = 0
    for message_chunk in split_text_into_chunks(transcribed_text, 4096):
        try:
            i = i + 1
            text = f"<b>{html.escape(file_name)}</b> 🎥 ({i}): <i>{html.escape(message_chunk)}</i>"
            await context.bot.send_message(
                update.effective_chat.id, text, parse_mode=ParseMode.HTML
            )
        except telegram.error.BadRequest:
            # answer has invalid characters, so we send it without parse_mode
            await context.bot.send_message(update.effective_chat.id, message_chunk)


async def error_handle(update: Update, context: CallbackContext) -> None:
    if not update:
        return
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(
            None, context.error, context.error.__traceback__
        )
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
                await context.bot.send_message(
                    update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML
                )
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(
            update.effective_chat.id, "Some error in error handler"
        )


async def help_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


async def version_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    version = get_version_string()
    await update.message.reply_text(f"Version: {version}")


async def set_language_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    chat_id = update.effective_chat.id
    if context.args:
        lang = context.args[0]
        if chat_id not in chat_settings:
            chat_settings[chat_id] = {
                "language": "pt",
                "model": "whisper-1",
                "gpt_model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.7,
            }
        chat_settings[chat_id]["language"] = lang
        save_chat_settings()
        await update.message.reply_text(f"Language set to {lang}")
    else:
        await update.message.reply_text("Usage: /set_language <lang>")


async def set_model_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    chat_id = update.effective_chat.id
    if context.args:
        model = context.args[0]
        if chat_id not in chat_settings:
            chat_settings[chat_id] = {
                "language": "pt",
                "model": "whisper-1",
                "gpt_model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.7,
            }
        chat_settings[chat_id]["model"] = model
        save_chat_settings()
        await update.message.reply_text(f"Model set to {model}")
    else:
        await update.message.reply_text("Usage: /set_model <model>")


async def set_gpt_model_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    chat_id = update.effective_chat.id
    if context.args:
        gpt_model = context.args[0]
        if chat_id not in chat_settings:
            chat_settings[chat_id] = {
                "language": "pt",
                "model": "whisper-1",
                "gpt_model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.7,
            }
        chat_settings[chat_id]["gpt_model"] = gpt_model
        save_chat_settings()
        await update.message.reply_text(f"GPT model set to {gpt_model}")
    else:
        await update.message.reply_text("Usage: /set_gpt_model <model>")


async def set_max_tokens_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    chat_id = update.effective_chat.id
    if context.args:
        try:
            max_tokens = int(context.args[0])
            if chat_id not in chat_settings:
                chat_settings[chat_id] = {
                    "language": "pt",
                    "model": "whisper-1",
                    "gpt_model": "gpt-3.5-turbo",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                }
            chat_settings[chat_id]["max_tokens"] = max_tokens
            save_chat_settings()
            await update.message.reply_text(f"Max tokens set to {max_tokens}")
        except ValueError:
            await update.message.reply_text("Usage: /set_max_tokens <number>")
    else:
        await update.message.reply_text("Usage: /set_max_tokens <number>")


async def set_temperature_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    chat_id = update.effective_chat.id
    if context.args:
        try:
            temperature = float(context.args[0])
            if chat_id not in chat_settings:
                chat_settings[chat_id] = {
                    "language": "pt",
                    "model": "whisper-1",
                    "gpt_model": "gpt-3.5-turbo",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                }
            chat_settings[chat_id]["temperature"] = temperature
            save_chat_settings()
            await update.message.reply_text(f"Temperature set to {temperature}")
        except ValueError:
            await update.message.reply_text("Usage: /set_temperature <float>")
    else:
        await update.message.reply_text("Usage: /set_temperature <float>")


async def show_settings_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    chat_id = update.effective_chat.id

    # Get current settings for this chat
    settings = chat_settings.get(
        chat_id,
        {
            "language": "pt",
            "model": "whisper-1",
            "gpt_model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    )

    settings_text = f"""Current settings for this chat:

🎙️ <b>Transcription Settings:</b>
• Language: {html.escape(str(settings['language']))}
• Whisper Model: {html.escape(str(settings['model']))}

🤖 <b>GPT Response Settings:</b>
• GPT Model: {html.escape(str(settings['gpt_model']))}
• Max Tokens: {html.escape(str(settings['max_tokens']))}
• Temperature: {html.escape(str(settings['temperature']))}"""

    await update.message.reply_text(settings_text, parse_mode=ParseMode.HTML)


async def post_init(application: Application):
    await application.bot.set_my_commands(
        [
            BotCommand("/help", "Show help message"),
            BotCommand("/version", "Show version information"),
            BotCommand("/show_settings", "Show current settings"),
            BotCommand("/set_language", "Set transcription language"),
            BotCommand("/set_model", "Set Whisper model"),
            BotCommand("/set_gpt_model", "Set GPT model"),
            BotCommand("/set_max_tokens", "Set max tokens for GPT"),
            BotCommand("/set_temperature", "Set temperature for GPT"),
        ]
    )


async def unsupported_message_handle(
    update: Update, context: CallbackContext, message=None
):
    error_text = f"I don't know how to read files or videos. Send the picture in normal mode (Quick Mode)."
    logger.error(error_text)
    await update.message.reply_text(error_text)
    return


def is_url(text):
    url_pattern = re.compile(r"https?://[^\s]+")
    return url_pattern.match(text.strip()) is not None


async def download_audio_from_url(url):
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Find the mp3 file
            audio_files = glob.glob(os.path.join(temp_dir, "*.mp3"))
            if audio_files:
                audio_file = audio_files[0]
                with open(audio_file, "rb") as f:
                    audio_data = f.read()
                return io.BytesIO(audio_data), info.get("title", "video")
    return None, None


def extract_transcription_from_message(message_text):
    # Extract transcription from bot messages
    # Messages are in format: "prefix 🎤/🔊/🎥/🔗: transcription" or "prefix 🎤/🔊/🎥/🔗 (number): transcription"
    # Find the transcription part after the emoji and colon
    pattern = r"[\w\s\.\-\(\)]*?[🎤🔊🎥🔗]\s*:\s*(.*)"
    match = re.search(pattern, message_text)
    if match:
        return match.group(1).strip()
    return None


async def message_handle(update: Update, context: CallbackContext):
    if not update.message:
        return
    logger.info("Received message")
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    chat_id = update.effective_chat.id
    text = update.message.text
    settings = chat_settings.get(
        chat_id,
        {
            "language": "pt",
            "model": "whisper-1",
            "gpt_model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    )

    # Unified handling for replies to bot transcriptions and related commands
    replied = update.message.reply_to_message

    # If the user issued a special command that requires replying to a transcription
    if text and (
        text.startswith("/summarize")
        or text.startswith("/bullets")
        or text.startswith("/translate")
    ):
        if (
            not replied
            or not getattr(replied, "from_user", None)
            or replied.from_user.id != context.bot.id
        ):
            await update.message.reply_text(
                "Please reply to a transcription message with this command."
            )
            return

        replied_text = replied.text or replied.caption or ""
        transcription = extract_transcription_from_message(replied_text)
        if not transcription or not any(
            emoji in replied_text for emoji in ["🎤", "🔊", "🎥", "🔗"]
        ):
            await update.message.reply_text(
                "Please reply to a valid transcription message."
            )
            return

        # Build the appropriate prompt based on the command
        if text.startswith("/summarize"):
            user_prompt = (
                f"Summarize the following text in the same language: {transcription}"
            )
        elif text.startswith("/bullets"):
            user_prompt = f"Create bullet points for the following text in the same language: {transcription}"
        else:  # /translate
            parts = text.split(" ", 1)
            if len(parts) == 2 and parts[1].strip():
                lang = parts[1].strip()
                user_prompt = f"Translate the following text to {lang}, keeping the same content and format: {transcription}"
            else:
                await update.message.reply_text(
                    "Please specify a language for translation, e.g., /translate en"
                )
                return

        await update.message.reply_text("Generating response based on transcription...")
        response = await openai_utils.generate_gpt_response(
            transcription,
            user_prompt,
            **{
                k: v
                for k, v in settings.items()
                if k in ["gpt_model", "max_tokens", "temperature"]
            },
        )
        await update.message.reply_text(
            f"🤖: <i>{html.escape(response)}</i>", parse_mode=ParseMode.HTML
        )
        return

    # If replying to a bot message (likely a transcription) with a free-form prompt
    if (
        replied
        and getattr(replied, "from_user", None)
        and replied.from_user.id == context.bot.id
    ):
        replied_text = replied.text or replied.caption or ""
        transcription = extract_transcription_from_message(replied_text)
        if transcription and any(
            emoji in replied_text for emoji in ["🎤", "🔊", "🎥", "🔗"]
        ):
            user_prompt = text or ""
            await update.message.reply_text(
                "Generating response based on transcription..."
            )
            response = await openai_utils.generate_gpt_response(
                transcription,
                user_prompt,
                **{
                    k: v
                    for k, v in settings.items()
                    if k in ["gpt_model", "max_tokens", "temperature"]
                },
            )
            await update.message.reply_text(
                f"🤖: <i>{html.escape(response)}</i>", parse_mode=ParseMode.HTML
            )
            return

    if is_url(text):
        # treat as video URL
        await update.message.reply_text(
            "Downloading and transcribing video from URL... Please wait."
        )
        audio_buf, title = await download_audio_from_url(text)
        if audio_buf:
            audio_buf.name = "audio.mp3"
            audio_buf.seek(0)
            transcribed_text = await openai_utils.transcribe_audio(
                audio_buf,
                **{k: v for k, v in settings.items() if k in ["language", "model"]},
            )
            file_name = title or "video from URL"
            if len(transcribed_text) < 4096:
                text_reply = f"<b>{html.escape(file_name)}</b> 🔗: <i>{html.escape(transcribed_text)}</i>"
                await update.message.reply_text(text_reply, parse_mode=ParseMode.HTML)
                return
            # split text
            i = 0
            for message_chunk in split_text_into_chunks(transcribed_text, 4096):
                try:
                    i = i + 1
                    text_reply = f"<b>{html.escape(file_name)}</b> 🔗 ({i}): <i>{html.escape(message_chunk)}</i>"
                    await context.bot.send_message(
                        update.effective_chat.id, text_reply, parse_mode=ParseMode.HTML
                    )
                except telegram.error.BadRequest:
                    await context.bot.send_message(
                        update.effective_chat.id, message_chunk
                    )
        else:
            await update.message.reply_text(
                "Failed to download or extract audio from URL."
            )
    else:
        # Only echo non-reply messages
        if not update.message.reply_to_message:
            if text.startswith("/"):
                await update.message.reply_text(
                    "Please reply to a transcription message with this command."
                )
                return
            await update.message.reply_text(
                f"📝: <i>{html.escape(text)}</i>", parse_mode=ParseMode.HTML
            )


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
        user_filter = (
            filters.User(username=usernames)
            | filters.User(user_id=user_ids)
            | filters.Chat(chat_id=group_ids)
        )

    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))
    application.add_handler(
        CommandHandler("version", version_handle, filters=user_filter)
    )
    application.add_handler(
        CommandHandler("set_language", set_language_handle, filters=user_filter)
    )
    application.add_handler(
        CommandHandler("set_model", set_model_handle, filters=user_filter)
    )
    application.add_handler(
        CommandHandler("set_gpt_model", set_gpt_model_handle, filters=user_filter)
    )
    application.add_handler(
        CommandHandler("set_max_tokens", set_max_tokens_handle, filters=user_filter)
    )
    application.add_handler(
        CommandHandler("set_temperature", set_temperature_handle, filters=user_filter)
    )
    application.add_handler(
        CommandHandler("show_settings", show_settings_handle, filters=user_filter)
    )

    application.add_handler(MessageHandler(filters.TEXT & user_filter, message_handle))

    application.add_handler(
        MessageHandler(filters.VOICE & user_filter, voice_message_handle)
    )

    application.add_handler(
        MessageHandler(filters.AUDIO & ~filters.COMMAND & user_filter, audio_handle)
    )

    application.add_handler(MessageHandler(filters.VIDEO & user_filter, video_handle))

    application.add_error_handler(error_handle)

    # start the bot
    application.run_polling(timeout=30)


if __name__ == "__main__":
    logger.info("Allowed usernames: %s", allowed_telegram_usernames)
    run_bot()
