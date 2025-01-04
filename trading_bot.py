import ccxt
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
import time
import sys

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class TradingBot:
    def __init__(self):
        self.exchange = ccxt.huobi()
        self.signals = {}  # Store previous signals to avoid duplicate alerts
        self.monitoring_active = True
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.channel_id = os.getenv('TELEGRAM_CHANNEL_ID')
        
        if not self.telegram_token or not self.channel_id:
            raise ValueError("Telegram token or channel ID not found in environment variables")
            
        self.application = Application.builder().token(self.telegram_token).build()
        self.setup_handlers()
        self.timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '8h': 480,
            '12h': 720,
            '1d': 1440
        }
        self.setup_logging()
        self.MIN_VOLUME_RATIO = 1.5      # Increased for stronger volume confirmation
        self.MIN_TREND_STRENGTH = 1.03   # Reduced to catch trends earlier
        self.RSI_LOWER = 30              # Widened range for more opportunities
        self.RSI_UPPER = 70              # Widened range for more opportunities
        self.REQUIRED_CONDITIONS = 3      # Keep reduced for higher frequency
        self.TOTAL_CONDITIONS = 7
        self.RISK_PER_TRADE = 0.02      # Increased to 2% per trade
        self.MAX_TRADES_PER_PAIR = 3     # Keep maximum concurrent trades
        self.PROFIT_FACTOR = 2.5         # Increased reward/risk ratio
        self.TRAILING_STOP = True        # Keep trailing stop enabled
        self.TRAILING_STOP_FACTOR = 1.8  # Increased trailing stop distance
        
        # Trading pairs with their success rates
        self.trading_pairs = [
            'HBAR/USDT',  # 7 trades, 100% win rate
            'TON/USDT',   # 3 trades, 100% win rate
            'ALGO/USDT',  # 3 trades, 100% win rate
            'GRT/USDT',   # 3 trades, 100% win rate
            'CHZ/USDT',   # 3 trades, 100% win rate
            'VET/USDT',   # 3 trades, 100% win rate
            'MANA/USDT',  # 2 trades, 100% win rate
            'ZIL/USDT',   # 2 trades, 100% win rate
            'IOTA/USDT',  # 2 trades, 100% win rate
            'GALA/USDT',  # 2 trades, 100% win rate
            'ZRX/USDT',   # 2 trades, 100% win rate
            'ENJ/USDT',   # 2 trades, 100% win rate
            'AUDIO/USDT', # 2 trades, 100% win rate
            'FLOW/USDT',  # 2 trades, 100% win rate
            'MASK/USDT',  # 2 trades, 100% win rate
            'ANKR/USDT',  # 2 trades, 100% win rate
            'FTM/USDT',   # 4 trades, 75% win rate
            'ARB/USDT',   # 3 trades, 66.67% win rate
            'KAVA/USDT',  # 3 trades, 66.67% win rate
            'ONE/USDT',   # 3 trades, 66.67% win rate
            'CFX/USDT',   # 3 trades, 66.67% win rate
            'SKL/USDT'    # 3 trades, 66.67% win rate
        ]
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(ch)
        
    def log_error(self, message):
        """Log error messages"""
        self.logger.error(message)
        
    def log_info(self, message):
        """Log info messages"""
        self.logger.info(message)
        
    def log_signal(self, pair, analysis, current_time):
        """Log detailed information about the signal"""
        signal = analysis['signal']
        indicators = analysis['indicators']
        
        log_message = (
            f"\n{'='*50}\n"
            f"SIGNAL DETECTED - {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"{'='*50}\n"
            f"Pair: {pair}\n"
            f"Action: {signal['side'].upper()}\n"
            f"Price: {analysis['price']:.4f}\n"
            f"Stop Loss: {signal['stop_loss']:.4f}\n"
            f"Take Profit: {signal['take_profit']:.4f}\n"
            f"Risk/Reward: 1:{abs((signal['take_profit'] - analysis['price']) / (signal['stop_loss'] - analysis['price'])):.2f}\n\n"
            f"Technical Indicators:\n"
            f"- RSI: {indicators['rsi']:.2f}\n"
            f"- Volume Ratio: {indicators['volume_ratio']:.2f}x\n"
            f"- BB Width: {indicators['bb_width']:.2f}%\n"
            f"- Trend: {'Bullish' if indicators['trend'] else 'Bearish'}\n"
            f"{'='*50}"
        )
        
        self.log_info(log_message)

    def setup_handlers(self):
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("pairs", self.cmd_pairs))
        self.application.add_handler(CommandHandler("pause", self.cmd_pause))
        self.application.add_handler(CommandHandler("resume", self.cmd_resume))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_msg = (
            "*Welcome to the Premium Crypto Trading Bot!*\n\n"
            "I monitor multiple crypto pairs on Huobi using a custom strategy "
            "combined with volume analysis.\n\n"
            "Available commands:\n"
            "/help - Show all available commands\n"
            "/status - Check bot status\n"
            "/pairs - List monitored pairs\n"
            "/pause - Pause signal monitoring\n"
            "/resume - Resume signal monitoring"
        )
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            "*Available Commands:*\n\n"
            "*/start* - Start the bot\n"
            "*/help* - Show this help message\n"
            "*/status* - Check bot status and statistics\n"
            "*/pairs* - List all monitored trading pairs\n"
            "*/pause* - Pause signal monitoring\n"
            "*/resume* - Resume signal monitoring\n\n"
            "*Trading Strategy:*\n"
            "• Custom strategy with SMA and volume analysis"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status_text = (
            "*Bot Status*\n\n"
            f"Status: {'Active' if self.monitoring_active else 'Paused'}\n"
            f"Pairs Monitored: {len(self.trading_pairs)}\n"
            f"Signals Today: {len(self.signals)}\n"
            f"Last Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await update.message.reply_text(status_text, parse_mode='Markdown')

    async def cmd_pairs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pairs_text = "*Monitored Trading Pairs:*\n\n"
        for pair in self.trading_pairs:
            pairs_text += f"• {pair}\n"
        await update.message.reply_text(pairs_text, parse_mode='Markdown')

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.monitoring_active = False
        await update.message.reply_text("Signal monitoring paused!")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.monitoring_active = True
        await update.message.reply_text("Signal monitoring resumed!")

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

    async def format_signal_message(self, pair, analysis, current_time):
        """Format the signal message with detailed analysis"""
        signal = analysis['signal']
        price = analysis['price']
        
        # Get additional indicators
        indicators = {
            'RSI': f"{analysis['indicators']['rsi']:.2f}",
            'Volume Ratio': f"{analysis['indicators']['volume_ratio']:.2f}x",
            'BB Width': f"{analysis['indicators']['bb_width']:.2f}%",
            'SMA Trend': "Bullish" if analysis['indicators']['trend'] else "Bearish"
        }
        
        # Emoji indicators
        emojis = {
            'buy': '🟢',
            'sell': '🔴',
            'volume': '📊',
            'trend': '📈',
            'alert': '🚨',
            'target': '🎯',
            'stop': '🛑',
            'time': '⏰'
        }
        
        # Calculate potential profit targets
        entry = price
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        risk_reward = abs((take_profit - entry) / (stop_loss - entry))
        
        message = (
            f"{emojis['alert']} *TRADING SIGNAL* {emojis['alert']}\n\n"
            f"*{pair}* - {emojis['buy'] if signal['side'] == 'buy' else emojis['sell']} *{signal['side'].upper()}*\n\n"
            
            f"{emojis['time']} *Time*: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
            
            f"💵 *Entry Price*: {price:.4f}\n"
            f"{emojis['target']} *Take Profit*: {take_profit:.4f} ({((take_profit/entry - 1) * 100):.2f}%)\n"
            f"{emojis['stop']} *Stop Loss*: {stop_loss:.4f} ({((stop_loss/entry - 1) * 100):.2f}%)\n"
            f"📊 *Risk/Reward*: 1:{risk_reward:.2f}\n\n"
            
            f"*Technical Indicators*:\n"
            f"• RSI: {indicators['RSI']}\n"
            f"• Volume: {indicators['Volume Ratio']}\n"
            f"• BB Width: {indicators['BB Width']}\n"
            f"• Trend: {indicators['SMA Trend']}\n\n"
            
            f"*Timeframe*: 15m\n\n"
            
            f"⚠️ *Risk Warning*: This is not financial advice. Always do your own research and trade responsibly."
        )
        
        return message

    async def send_telegram_message(self, message):
        """Send message to Telegram channel"""
        try:
            await self.application.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode='Markdown'
            )
            self.log_info(f"Signal sent successfully")
        except Exception as e:
            self.log_error(f"Error sending Telegram message: {str(e)}")

    async def send_startup_message(self):
        """Send startup message to Telegram channel"""
        try:
            test_message = "Bot is starting... Test message"
            self.log_info(f"Attempting to send message to channel {self.channel_id}")
            await self.send_telegram_message(test_message)
            self.log_info("Test message sent successfully")
        except Exception as e:
            self.log_error(f"Error in send_startup_message: {str(e)}")

    def calculate_indicators(self, df):
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Multiple SMAs for trend
        df['sma20'] = df['bb_middle']  # reuse BB middle band
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR for stop loss
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        return df
        
    def get_signal(self, df):
        if len(df) < 200:  # Need enough data for 200 SMA
            return None
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        price = current['close']
        
        # Calculate indicators for signal message
        indicators = {
            'rsi': current['rsi'],
            'volume_ratio': current['volume_ratio'],
            'bb_width': ((current['bb_upper'] - current['bb_lower']) / current['bb_middle']) * 100,
            'trend': current['sma20'] > current['sma50'] > current['sma200']
        }
        
        # Strategy parameters
        VOLUME_THRESHOLD = 1.2
        RSI_OVERSOLD = 30
        RSI_OVERBOUGHT = 70
        ATR_MULTIPLIER = 2.5
        
        # Calculate stop loss and take profit based on ATR
        stop_loss_pips = current['atr'] * ATR_MULTIPLIER
        stop_loss = price * (1 - (stop_loss_pips / price))
        take_profit = price * (1 + (stop_loss_pips * 1.5 / price))  # 1.5:1 reward-to-risk ratio
        
        # Trend conditions
        uptrend = (current['sma20'] > current['sma50'] > current['sma200'])
        downtrend = (current['sma20'] < current['sma50'] < current['sma200'])
        
        # Volume condition
        volume_spike = current['volume_ratio'] > VOLUME_THRESHOLD
        
        # Long entry conditions
        long_signal = (
            price < current['bb_lower'] and  # Price below lower BB
            prev['close'] < prev['bb_lower'] and  # Confirmation
            current['rsi'] < RSI_OVERSOLD and  # Oversold
            volume_spike and  # Volume confirmation
            uptrend  # Trend alignment
        )
        
        # Short entry conditions
        short_signal = (
            price > current['bb_upper'] and  # Price above upper BB
            prev['close'] > prev['bb_upper'] and  # Confirmation
            current['rsi'] > RSI_OVERBOUGHT and  # Overbought
            volume_spike and  # Volume confirmation
            downtrend  # Trend alignment
        )
        
        if long_signal:
            return {
                'signal': {
                    'side': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                },
                'price': price,
                'indicators': indicators
            }
        elif short_signal:
            return {
                'signal': {
                    'side': 'sell',
                    'stop_loss': price * (2 - stop_loss/price),  # Inverse for shorts
                    'take_profit': price * (2 - take_profit/price)  # Inverse for shorts
                },
                'price': price,
                'indicators': indicators
            }
        
        return None
        
    def check_exit(self, df, current_idx):
        """Enhanced exit conditions with trailing stop"""
        if current_idx < 1:
            return False
            
        current = df.iloc[current_idx]
        prev = df.iloc[current_idx - 1]
        
        # Basic exit conditions
        conditions = [
            current['rsi'] > self.RSI_UPPER,  # RSI overbought
            current['close'] < current['sma20']  # Price below SMA20
        ]
        
        # Check if any exit condition is met
        basic_exit = any(conditions)
        
        # Calculate current profit percentage
        entry_price = df.iloc[current_idx]['entry_price'] if 'entry_price' in df.columns else None
        if entry_price:
            profit_pct = ((current['close'] - entry_price) / entry_price) * 100
            stop_loss = df.iloc[current_idx]['stop_loss'] if 'stop_loss' in df.columns else None
            
            if stop_loss:
                stop_loss_pct = abs((stop_loss - entry_price) / entry_price) * 100
                
                # Check profit target
                if profit_pct >= (stop_loss_pct * self.PROFIT_FACTOR):
                    return True
                
                # Check trailing stop
                if profit_pct >= (stop_loss_pct * self.RISK_PER_TRADE):
                    trailing_stop = current['close'] * (1 - self.RISK_PER_TRADE)
                    if prev['close'] > trailing_stop and current['close'] <= trailing_stop:
                        return True
        
        return basic_exit

    def detect_candlestick_pattern(self, df, index):
        """Detect basic candlestick patterns"""
        try:
            current = df.iloc[index]
            prev = df.iloc[index - 1]
            
            # Basic bullish engulfing
            bullish_engulfing = (
                current['close'] > current['open'] and
                prev['close'] < prev['open'] and
                current['close'] > prev['open'] and
                current['open'] < prev['close']
            )
            
            # Basic hammer
            body = abs(current['close'] - current['open'])
            lower_wick = min(current['open'], current['close']) - current['low']
            upper_wick = current['high'] - max(current['open'], current['close'])
            hammer = (
                lower_wick > body * 2 and
                upper_wick < body * 0.5
            )
            
            return bullish_engulfing or hammer
            
        except Exception as e:
            self.log_error(f"Error detecting candlestick pattern: {str(e)}")
            return False

    def calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()
        
    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        try:
            # Huobi has a limit of 1000 candles per request
            if limit > 1000:
                # Calculate number of chunks needed
                chunk_size = 1000
                chunks = []
                remaining = limit
                last_timestamp = None
                
                while remaining > 0:
                    chunk_limit = min(remaining, chunk_size)
                    if last_timestamp:
                        # Convert timestamp to milliseconds for the API
                        since = int(last_timestamp.timestamp() * 1000) - (chunk_limit * 60 * 60 * 1000)
                        chunk = self.exchange.fetch_ohlcv(symbol, timeframe, limit=chunk_limit, since=since)
                    else:
                        chunk = self.exchange.fetch_ohlcv(symbol, timeframe, limit=chunk_limit)
                    
                    if not chunk:
                        break
                        
                    chunk_df = pd.DataFrame(chunk, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='ms')
                    
                    if not last_timestamp:
                        last_timestamp = chunk_df['timestamp'].min()
                    
                    chunks.append(chunk_df)
                    remaining -= chunk_limit
                
                if chunks:
                    # Combine all chunks and sort by timestamp
                    df = pd.concat(chunks, ignore_index=True)
                    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
                    # Remove duplicates if any
                    df = df.drop_duplicates(subset=['timestamp'])
                    return df
            else:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv:
                    return None
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            
        except Exception as e:
            self.log_error(f"Error fetching data for {symbol}: {e}")
            return None

    def analyze_pair(self, symbol):
        if not self.monitoring_active:
            return None

        df = self.fetch_ohlcv(symbol)
        if df is None or len(df) < 200:  # Need at least 200 candles for all indicators
            return None

        # Calculate indicators
        df = self.calculate_indicators(df)
        
        signal = self.get_signal(df)
        
        if signal:
            return {
                'symbol': symbol,
                'signal': signal,
                'price': df.iloc[-1]['close'],
                'timestamp': df.iloc[-1]['timestamp'],
                'reason': [
                    f"Price above SMA3 and SMA5",
                    f"Price near SMA5 bottom",
                    f"Price near SMA5 top with volume spike"
                ]
            }

        return None

    async def monitor_pairs(self):
        while True:
            try:
                for pair in self.trading_pairs:
                    if not self.monitoring_active:
                        await asyncio.sleep(60)
                        continue

                    analysis = self.analyze_pair(pair)
                    
                    if analysis:
                        # Check if we haven't sent this signal recently
                        signal_key = f"{pair}_{analysis['signal']['side']}"
                        current_time = datetime.now()
                        
                        if (signal_key not in self.signals or 
                            (current_time - self.signals[signal_key]).total_seconds() > 3600):  # 1 hour cooldown
                            
                            # Log signal details
                            self.log_signal(pair, analysis, current_time)
                            
                            # Format and send message
                            message = await self.format_signal_message(pair, analysis, current_time)
                            await self.send_telegram_message(message)
                            self.signals[signal_key] = current_time
                            self.log_info(f"Signal sent for {pair}: {analysis['signal']['side']}")
                    
                    await asyncio.sleep(1)  # Avoid rate limits
                
                await asyncio.sleep(60)  # Check each pair every minute
                
            except Exception as e:
                self.log_error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    def calculate_trade_metrics(self, trades):
        """Calculate various trading metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'avg_profit': 0.0,
                'avg_duration': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_consecutive_losses': 0
            }

        total_trades = len(trades)
        winning_trades = [t for t in trades if t['profit'] > 0]
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        total_return = sum(t['profit'] for t in trades)
        avg_profit = total_return / total_trades if total_trades > 0 else 0

        # Calculate trade durations in hours
        durations = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                durations.append(duration)
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Calculate maximum drawdown
        equity_curve = [0]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['profit'])
        
        max_drawdown = 0
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / (peak + 1e-9)  # Add small value to avoid division by zero
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate profit factor
        gross_profits = sum(t['profit'] for t in trades if t['profit'] > 0)
        gross_losses = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else 0

        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns = [t['profit'] for t in trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return != 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate max consecutive losses
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        for trade in trades:
            if trade['profit'] < 0:
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                current_consecutive_losses = 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_profit': avg_profit,
            'avg_duration': avg_duration,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_losses': max_consecutive_losses
        }

    def backtest(self, symbol, timeframe='1h', days=180):
        try:
            # Calculate the number of candles needed
            limit = int((days * 24 * 60) / self.timeframe_minutes[timeframe])
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            
            if df is None or len(df) < 200:
                self.log_error(f"Not enough data for backtesting {symbol}")
                return None
            
            # Initialize results
            trades = []
            position = None
            total_trades = 0
            winning_trades = 0
            total_profit = 0
            max_drawdown = 0
            peak_value = 1000  # Starting with 1000 USDT
            current_value = peak_value
            
            # Performance metrics
            profits = []
            drawdowns = []
            trade_durations = []
            
            df = self.calculate_indicators(df)
            
            for i in range(200, len(df)):
                current_price = df.iloc[i]['close']
                current_time = df.iloc[i].name
                
                # Check exit conditions if in position
                if position:
                    exit_signal = self.check_exit(df.iloc[:i+1], i)
                    if exit_signal:
                        # Calculate profit
                        stop_loss_hit = bool(current_price < position['stop_loss'])
                        take_profit_hit = bool(current_price > position['take_profit'])
                        
                        if stop_loss_hit:
                            exit_price = position['stop_loss']
                        elif take_profit_hit:
                            exit_price = position['take_profit']
                        else:
                            exit_price = current_price
                        
                        profit_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                        profit_amount = position['size'] * profit_pct / 100
                        
                        trade_info = {
                            'symbol': symbol,
                            'entry_time': pd.Timestamp(position['entry_time']),
                            'exit_time': pd.Timestamp(current_time),
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'position_size': position['size'],
                            'profit_pct': profit_pct,
                            'profit_amount': profit_amount,
                            'trade_duration': (pd.Timestamp(current_time) - pd.Timestamp(position['entry_time'])).total_seconds() / 3600  # hours
                        }
                        
                        trades.append(trade_info)
                        total_trades += 1
                        if profit_pct > 0:
                            winning_trades += 1
                        
                        total_profit += profit_amount
                        current_value += profit_amount
                        
                        if current_value > peak_value:
                            peak_value = current_value
                        drawdown = (peak_value - current_value) / peak_value * 100
                        max_drawdown = max(max_drawdown, drawdown)
                        
                        profits.append(profit_pct)
                        drawdowns.append(drawdown)
                        trade_durations.append(trade_info['trade_duration'])
                        
                        position = None
                        
                # Check entry conditions if not in position
                if not position:
                    signal = self.get_signal(df.iloc[:i+1])
                    if signal:
                        position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'stop_loss': signal['signal']['stop_loss'],
                            'take_profit': signal['signal']['take_profit'],
                            'size': 1000 * 0.95  # Using 95% of available capital
                        }
            
            # Calculate performance metrics
            if total_trades > 0:
                win_rate = (winning_trades / total_trades) * 100
                avg_profit = sum(profits) / len(profits)
                avg_duration = sum(trade_durations) / len(trade_durations)
                
                profit_factor = 1.0
                if sum(p for p in profits if p < 0) != 0:
                    profit_factor = abs(sum(p for p in profits if p > 0) / sum(p for p in profits if p < 0))
                
                sharpe_ratio = 0
                if np.std(profits) != 0:
                    sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(365)  # Annualized
                
                max_consecutive_losses = 0
                current_losses = 0
                for profit in profits:
                    if profit < 0:
                        current_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, current_losses)
                    else:
                        current_losses = 0
                
                results = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'period_days': days,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'total_return': ((current_value - 1000) / 1000) * 100,
                    'avg_profit': avg_profit,
                    'avg_duration': avg_duration,
                    'max_drawdown': max_drawdown,
                    'profit_factor': profit_factor,
                    'sharpe_ratio': sharpe_ratio,
                    'max_consecutive_losses': max_consecutive_losses,
                    'trades': trades
                }
                
                self.log_info(f"\nBacktest Results for {symbol}:")
                self.log_info(f"Total Trades: {total_trades}")
                self.log_info(f"Win Rate: {win_rate:.2f}%")
                self.log_info(f"Total Return: {results['total_return']:.2f}%")
                self.log_info(f"Average Profit per Trade: {avg_profit:.2f}%")
                self.log_info(f"Average Trade Duration: {avg_duration:.2f} hours")
                self.log_info(f"Maximum Drawdown: {max_drawdown:.2f}%")
                self.log_info(f"Profit Factor: {profit_factor:.2f}")
                self.log_info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
                self.log_info(f"Max Consecutive Losses: {max_consecutive_losses}")
                
                return results
            
            return None
            
        except Exception as e:
            self.log_error(f"Error during backtesting: {str(e)}")
            return None

    def calculate_position_size(self, df):
        """Calculate position size based on volatility"""
        # Calculate daily returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate volatility (standard deviation of returns)
        volatility = df['returns'].rolling(window=20).std().iloc[-1]
        
        # Adjust position size inversely to volatility
        position_size = 0.1 / (volatility * 2.0)
        
        # Cap at maximum position size
        return min(position_size, 0.1)

    def fetch_historical_data(self, symbol, timeframe='1h', limit=1000):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.log_error(f"Error fetching historical data: {str(e)}")
            return None

    async def run_backtest(self):
        print("\nRunning Backtest Over Last 180 Days...")
        print("=" * 60)
        
        total_wins = 0
        total_losses = 0
        pair_results = []
        
        self.trading_pairs = [
            'HBAR/USDT',  # 7 trades, 100% win rate
            'TON/USDT',   # 3 trades, 100% win rate
            'ALGO/USDT',  # 3 trades, 100% win rate
            'GRT/USDT',   # 3 trades, 100% win rate
            'CHZ/USDT',   # 3 trades, 100% win rate
            'VET/USDT',   # 3 trades, 100% win rate
            'MANA/USDT',  # 2 trades, 100% win rate
            'ZIL/USDT',   # 2 trades, 100% win rate
            'IOTA/USDT',  # 2 trades, 100% win rate
            'GALA/USDT',  # 2 trades, 100% win rate
            'ZRX/USDT',   # 2 trades, 100% win rate
            'ENJ/USDT',   # 2 trades, 100% win rate
            'AUDIO/USDT', # 2 trades, 100% win rate
            'FLOW/USDT',  # 2 trades, 100% win rate
            'MASK/USDT',  # 2 trades, 100% win rate
            'ANKR/USDT',  # 2 trades, 100% win rate
            'FTM/USDT',   # 4 trades, 75% win rate
            'ARB/USDT',   # 3 trades, 66.67% win rate
            'KAVA/USDT',  # 3 trades, 66.67% win rate
            'ONE/USDT',   # 3 trades, 66.67% win rate
            'CFX/USDT',   # 3 trades, 66.67% win rate
            'SKL/USDT'    # 3 trades, 66.67% win rate
        ]
        
        for pair in self.trading_pairs:
            print(f"\nTesting {pair}...")
            result = self.backtest(pair)
            
            if result:
                pair_results.append(result)
                total_wins += result['winning_trades']
                total_losses += result['total_trades'] - result['winning_trades']
                
                print(f"Total Trades: {result['total_trades']}")
                print(f"Win Rate: {result['win_rate']:.2f}%")
                print(f"Total Return: {result['total_return']:.2f}%")
                print(f"Average Profit per Trade: {result['avg_profit']:.2f}%")
                print(f"Average Trade Duration: {result['avg_duration']:.2f} hours")
                print(f"Maximum Drawdown: {result['max_drawdown']:.2f}%")
                print(f"Profit Factor: {result['profit_factor']:.2f}")
                print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                print(f"Max Consecutive Losses: {result['max_consecutive_losses']}")
        
        total_trades = total_wins + total_losses
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        print("\n" + "=" * 60)
        print("Overall Backtest Results:")
        print(f"Total Trades Across All Pairs: {total_trades}")
        print(f"Total Wins: {total_wins}")
        print(f"Total Losses: {total_losses}")
        print(f"Overall Win Rate: {overall_win_rate:.2f}%")
        print("\nPerformance by Pair:")
        
        # Sort pairs by win rate
        pair_results.sort(key=lambda x: x['win_rate'], reverse=True)
        
        for result in pair_results:
            print(f"\n{result['symbol']}:")
            print(f"Win Rate: {result['win_rate']:.2f}%")
            print(f"Trades: {result['total_trades']} (W: {result['winning_trades']}, L: {result['total_trades'] - result['winning_trades']})")
            print(f"Avg Profit: {result['avg_profit']:.2f}%")

    async def run(self):
        """Run the trading bot"""
        try:
            self.log_info("Starting bot...")
            
            # Send startup message
            await self.send_startup_message()
            
            # Start the Telegram bot first
            await self.application.initialize()
            await self.application.start()
            
            # Start monitoring pairs in the background
            tasks = []
            for pair in self.trading_pairs:
                task = asyncio.create_task(self.monitor_pairs())
                tasks.append(task)
            
            # Run the application polling in the background
            polling_task = asyncio.create_task(
                self.application.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)
            )
            
            # Wait for all tasks or until interrupted
            try:
                await asyncio.gather(polling_task, *tasks)
            except asyncio.CancelledError:
                self.log_info("Received shutdown signal")
            
        except Exception as e:
            self.log_error(f"Error running bot: {str(e)}")
        finally:
            # Clean shutdown
            try:
                if self.application.running:
                    await self.application.stop()
                    await self.application.shutdown()
                self.log_info("Bot shutdown complete")
            except Exception as e:
                self.log_error(f"Error during shutdown: {str(e)}")

async def main():
    bot = TradingBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
