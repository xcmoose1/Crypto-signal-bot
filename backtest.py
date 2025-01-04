import ccxt
import pandas as pd
import asyncio
import numpy as np
from datetime import datetime, timedelta

# Trading pairs
TRADING_PAIRS = [
    # Major pairs
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT',
    'MATIC/USDT', 'TRX/USDT', 'DOT/USDT', 'LINK/USDT', 'TON/USDT',
    'ADA/USDT', 'DOGE/USDT', 'SHIB/USDT', 'LTC/USDT', 'UNI/USDT',
    
    # Mid caps
    'BCH/USDT', 'ICP/USDT', 'INJ/USDT', 'NEAR/USDT', 'OP/USDT',
    'FIL/USDT', 'ATOM/USDT', 'ARB/USDT', 'AAVE/USDT', 'APT/USDT',
    'ALGO/USDT', 'SAND/USDT', 'MANA/USDT', 'GRT/USDT', 'EGLD/USDT',
    
    # DeFi & Gaming
    'SNX/USDT', 'CRV/USDT', 'SUSHI/USDT', '1INCH/USDT', 'CAKE/USDT',
    'AXS/USDT', 'GMT/USDT', 'IMX/USDT', 'GALA/USDT', 'CHZ/USDT',
    'ROSE/USDT', 'DYDX/USDT', 'ENS/USDT', 'STX/USDT', 'KAVA/USDT',
    
    # Layer 1s & 2s
    'FTM/USDT', 'ONE/USDT', 'CELO/USDT', 'ZIL/USDT', 'QTUM/USDT',
    'KSM/USDT', 'WAVES/USDT', 'NEO/USDT', 'IOTA/USDT', 'XTZ/USDT',
    'THETA/USDT', 'ETC/USDT', 'XLM/USDT', 'VET/USDT', 'HBAR/USDT',
    
    # Exchange & Infrastructure
    'HT/USDT', 'CRO/USDT', 'OKB/USDT', 'KCS/USDT', 'FTT/USDT',
    'GT/USDT', 'WOO/USDT', 'CAKE/USDT', 'SXP/USDT', 'COMP/USDT',
    'MKR/USDT', 'ZRX/USDT', 'BAT/USDT', 'REN/USDT', 'BNT/USDT',
    
    # Metaverse & NFT
    'ENJ/USDT', 'AUDIO/USDT', 'ALICE/USDT', 'FLOW/USDT', 'ILV/USDT',
    'SLP/USDT', 'TLM/USDT', 'SUPER/USDT', 'RARE/USDT', 'HERO/USDT',
    
    # Additional Promising Projects
    'CFX/USDT', 'MASK/USDT', 'AGIX/USDT', 'FET/USDT', 'OCEAN/USDT',
    'API3/USDT', 'ANKR/USDT', 'CTSI/USDT', 'SKL/USDT', 'NU/USDT'
]

async def fetch_historical_data(exchange, symbol, timeframe='1h', limit=4800):  # ~200 days
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df):
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

def get_signal(df, current_idx):
    if current_idx < 200:  # Need enough data for 200 SMA
        return None
        
    current = df.iloc[current_idx]
    prev = df.iloc[current_idx - 1]
    price = current['close']
    
    # Strategy parameters
    VOLUME_THRESHOLD = 1.2
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    ATR_MULTIPLIER = 2.5
    
    # Calculate stop loss and take profit based on ATR
    stop_loss_pips = current['atr'] * ATR_MULTIPLIER
    STOP_LOSS = 1 - (stop_loss_pips / price)
    TAKE_PROFIT = 1 + (stop_loss_pips * 1.5 / price)  # 1.5:1 reward-to-risk ratio
    
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
            'side': 'buy',
            'stop_loss': STOP_LOSS,
            'take_profit': TAKE_PROFIT
        }
    elif short_signal:
        return {
            'side': 'sell',
            'stop_loss': 2 - STOP_LOSS,  # Inverse for shorts
            'take_profit': 2 - TAKE_PROFIT  # Inverse for shorts
        }
    
    return None

async def backtest():
    # Initialize exchange
    exchange = ccxt.binance({
        'enableRateLimit': True
    })
    
    total_trades = 0
    total_wins = 0
    total_profit = 0
    trades_per_pair = {}
    
    print("\nRunning backtest...")
    
    for symbol in TRADING_PAIRS:
        print(f"\nTesting {symbol}...")
        df = await fetch_historical_data(exchange, symbol)
        if df is None:
            continue
            
        df = calculate_indicators(df)
        
        trades = []
        active_trade = None
        pair_profit = 0
        wins = 0
        
        for i in range(len(df) - 25):  # -25 to ensure we have enough future data for outcome
            if active_trade:
                next_candle = df.iloc[i + 1]
                
                # Check stop loss and take profit
                if active_trade['side'] == 'buy':
                    if next_candle['low'] <= active_trade['stop_loss']:
                        trades.append({
                            'side': 'buy',
                            'entry': active_trade['entry_price'],
                            'exit': active_trade['stop_loss'],
                            'profit': -0.8,
                            'outcome': 'loss'
                        })
                        pair_profit -= 0.8
                        active_trade = None
                    elif next_candle['high'] >= active_trade['take_profit']:
                        trades.append({
                            'side': 'buy',
                            'entry': active_trade['entry_price'],
                            'exit': active_trade['take_profit'],
                            'profit': 1.5,
                            'outcome': 'win'
                        })
                        pair_profit += 1.5
                        wins += 1
                        active_trade = None
                elif active_trade['side'] == 'sell':
                    if next_candle['high'] >= active_trade['stop_loss']:
                        trades.append({
                            'side': 'sell',
                            'entry': active_trade['entry_price'],
                            'exit': active_trade['stop_loss'],
                            'profit': -0.8,
                            'outcome': 'loss'
                        })
                        pair_profit -= 0.8
                        active_trade = None
                    elif next_candle['low'] <= active_trade['take_profit']:
                        trades.append({
                            'side': 'sell',
                            'entry': active_trade['entry_price'],
                            'exit': active_trade['take_profit'],
                            'profit': 1.5,
                            'outcome': 'win'
                        })
                        pair_profit += 1.5
                        wins += 1
                        active_trade = None
            else:
                signal = get_signal(df, i)
                if signal:
                    entry_price = df.iloc[i + 1]['open']  # Enter on next candle open
                    active_trade = {
                        'side': signal['side'],
                        'entry_price': entry_price,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit']
                    }
        
        # Calculate pair statistics
        pair_trades = len(trades)
        if pair_trades > 0:
            win_rate = (wins / pair_trades) * 100
            trades_per_pair[symbol] = {
                'trades': pair_trades,
                'wins': wins,
                'win_rate': win_rate,
                'profit': pair_profit
            }
            
            total_trades += pair_trades
            total_wins += wins
            total_profit += pair_profit
            
            print(f"{symbol} Results:")
            print(f"Total trades: {pair_trades}")
            print(f"Win rate: {win_rate:.2f}%")
            print(f"Profit: {pair_profit:.2f}%")
    
    # Print overall results
    if total_trades > 0:
        overall_win_rate = (total_wins / total_trades) * 100
        print(f"\nOverall Results:")
        print(f"Total trades across all pairs: {total_trades}")
        print(f"Overall win rate: {overall_win_rate:.2f}%")
        print(f"Total profit: {total_profit:.2f}%")
        print("\nBest performing pairs:")
        
        # Sort pairs by profit
        sorted_pairs = sorted(trades_per_pair.items(), key=lambda x: x[1]['profit'], reverse=True)
        for symbol, stats in sorted_pairs[:5]:
            print(f"{symbol}: {stats['win_rate']:.2f}% win rate, {stats['profit']:.2f}% profit ({stats['trades']} trades)")

if __name__ == "__main__":
    asyncio.run(backtest())
