import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


async def transcribe_audio(audio_file, language="pt", model="whisper-1") -> str:

    client = OpenAI()

    # check if the file is too large
    if audio_file.getbuffer().nbytes > 20_000_000:
        logger.warning("Audio file is too large")
        return "Error: file too large"

    # catch errors from the OpenAI API
    try:
        transcription = client.audio.transcriptions.create(
            model=model, language=language, file=audio_file
        )
        logger.info(f"Transcribed audio: {transcription.text}")
        print(transcription.text)
        return transcription.text or ""
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return "Error: transcription failed"


async def generate_gpt_response(
    context: str,
    user_prompt: str,
    gpt_model="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=0.7,
) -> str:
    client = OpenAI()

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant. The following is a transcription of audio/video content: {context}",
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating GPT response: {e}")
        return "Error: failed to generate response"
