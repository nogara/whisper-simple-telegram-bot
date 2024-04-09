# Whisper Simple Bot

Use OpenAI Whisper to transform audio sent to the bot in text.

## Configuration

Create a `.env` file with the following content:

```bash
OPENAI_API_KEY=your-openai-api-key
TELGRAM_BOT_TOKEN=your-telegram-bot-token
ALLOWED_TELEGRAM_USERNAMES=telegram_username1,telegram_username2
```

## Building Docker image

```bash
docker build -t whisper-simple-bot .
```

## Running Docker container

```bash
docker run -d --env-file .env whisper-simple-bot
```
