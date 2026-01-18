from flask import Flask, render_template_string, request
import os
import json

app = Flask(__name__)

# Import the settings from __main__
# Since it's the same package, but to avoid circular, perhaps duplicate or use a config file.

# For simplicity, hardcode or read from env.

# But since chat_settings is global, perhaps import.

# To avoid issues, let's make a simple dashboard showing logs and allowing to set global defaults.

# But for now, a basic one.

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Whisper Bot Dashboard</title>
</head>
<body>
    <h1>Whisper Simple Bot Dashboard</h1>
    <h2>Logs</h2>
    <pre>{{ logs }}</pre>
    <h2>Settings</h2>
    <p>Global settings: {{ settings }}</p>
</body>
</html>
"""

@app.route('/')
def dashboard():
    try:
        with open('bot.log', 'r') as f:
            logs = f.read()
    except:
        logs = "No logs available"

    # For settings, since chat_settings is in __main__, perhaps save to a file or something.
    # For now, placeholder.
    settings = "Default: language=auto, model=whisper-1"

    return render_template_string(HTML_TEMPLATE, logs=logs[-10000:], settings=settings)  # last 10k chars

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)