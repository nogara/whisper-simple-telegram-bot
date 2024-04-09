import base64
from io import BytesIO
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

async def transcribe_audio(audio_file) -> str:

    client = OpenAI()

    # check if the file is too large
    if audio_file.getbuffer().nbytes > 20_000_000:
        logger.warning("Audio file is too large")
        return "Error: file too large"

    # catch errors from the OpenAI API
    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            language="pt",
            file=audio_file
        )
        logger.info(f"Transcribed audio: {transcription.text}")
        print(transcription.text)
        return transcription.text or ""
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return "Error: transcription failed"

