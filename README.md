# Whisper Simple Bot

Use OpenAI Whisper to transform audio sent to the bot in text.

## Configuration

Create a `.env` file with the following content:

```bash
OPENAI_API_KEY=your-openai-api-key
TELGRAM_BOT_TOKEN=your-telegram-bot-token
ALLOWED_TELEGRAM_USERNAMES=telegram_username1,telegram_username2
```

## Commands

The bot supports the following commands:

⚪ /start – Start the bot
⚪ /help – Show help message
⚪ /show_settings – Show current settings
⚪ /set_language <lang> – Set transcription language (e.g., pt, en)
⚪ /set_model <model> – Set Whisper model (e.g., whisper-1)
⚪ /set_gpt_model <model> – Set GPT model (e.g., gpt-3.5-turbo, gpt-4)
⚪ /set_max_tokens <number> – Set max tokens for GPT response (e.g., 1000)
⚪ /set_temperature <float> – Set temperature for GPT response (e.g., 0.7)

## Development

### Hot Reloading

For development with automatic restarts on code changes:

```bash
# Install development dependencies
uv sync --extra dev

# Run with hot reloading
uv run hupper python -m whisper_simple_bot
```

### Manual Run

To run without hot reloading:

```bash
uv run python -m whisper_simple_bot
```

## Debugging

### VS Code Setup

The project includes VS Code configuration for debugging:

1. **Set Python Interpreter**:

   - Open Command Palette (`Ctrl+Shift+P`)
   - Select "Python: Select Interpreter"
   - Choose the interpreter from `./.venv/bin/python`

2. **Debug Configurations**:

   - Go to Run & Debug panel (`Ctrl+Shift+D`)
   - Select "Debug Whisper Bot" for standard debugging
   - Select "Debug Whisper Bot (Hot Reload)" for debugging with hot reloading

3. **Using Breakpoints**:
   - Click in the gutter next to line numbers to set breakpoints
   - Start debugging with F5 or the play button
   - The bot will run and stop at breakpoints

### Environment Variables

Make sure your `.env` file is in the workspace root, or set environment variables:

- `OPENAI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `ALLOWED_TELEGRAM_USERNAMES`

## Building Docker image

```bash
docker build -t whisper-simple-bot .
```

## Running Docker container

```bash
docker run -d --env-file .env whisper-simple-bot
```
