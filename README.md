# Crypto Trading Signal Bot

This bot analyzes cryptocurrency pairs on HTX exchange and sends trading signals to a Telegram channel.

## Features

- Monitors 15 major cryptocurrency pairs
- Uses EMA crossover strategy combined with RSI and volume analysis
- Sends signals to Telegram channel
- Includes cooldown period to avoid signal spam
- Comprehensive logging system

## Strategy Details

The bot uses a combination of three technical indicators:
1. EMA Crossover (9 and 21 periods)
2. RSI (14 periods)
3. Volume analysis (20-period moving average)

Buy Signal Conditions:
- EMA 9 crosses above EMA 21
- RSI below 70
- Volume > 1.5x 20-period volume MA

Sell Signal Conditions:
- EMA 9 crosses below EMA 21
- RSI above 30
- Volume > 1.5x 20-period volume MA

## Local Setup Instructions

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Create a Telegram bot:
- Message @BotFather on Telegram
- Create new bot using /newbot command
- Copy the API token

3. Create a Telegram channel and add your bot as admin

4. Create .env file:
- Copy .env.example to .env
- Add your Telegram bot token
- Add your channel ID

5. Run the bot:
```bash
python trading_bot.py
```

## Deployment on Render.com

1. Create a new account on Render.com if you haven't already

2. Fork or push this repository to GitHub

3. In Render Dashboard:
   - Click "New +"
   - Select "Web Service"
   - Connect your GitHub repository
   - Choose the repository with the bot

4. Configure the service:
   - Name: crypto-trading-bot (or your preferred name)
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python trading_bot.py`

5. Add Environment Variables:
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHANNEL_ID

6. Click "Create Web Service"

The bot will automatically deploy and start running on Render's infrastructure.

## Configuration

You can modify the following parameters in trading_bot.py:
- TRADING_PAIRS: List of cryptocurrency pairs to monitor
- Timeframe: Currently set to 1h
- Signal cooldown: Currently set to 1 hour
- Technical indicator parameters (EMA periods, RSI periods, etc.)

## Logging

The bot creates a trading_bot.log file with detailed operation logs.
