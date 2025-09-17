# src/models/backtesting/ml_backtester.py
"""
Complete ML Model Backtesting Framework - Phase 1 Day 9
Comprehensive backtesting system for machine learning trading models

Features:
- Walk-forward analysis for time-series data
- Out-of-sample testing with realistic constraints
- Performance metrics tracking (Sharpe, max drawdown, hit rate)
- Feature importance analysis and model interpretability
- Integration with Alpha Model for trade hypothesis validation
- Realistic transaction costs, slippage, and market impact modeling
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import json
import warnings
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ML and statistical libraries
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

# Technical analysis
import pandas_ta as ta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_position_size: float = 0.20  # 20%
    walk_forward_window: int = 252 * 2  # 2 years
    rebalance_frequency: int = 22  # Monthly
    out_of_sample_ratio: float = 0.2  # 20%
    risk_free_rate: float = 0.05  # 5% annual


@dataclass
class TradeResult:
    """Single trade result structure"""
    trade_id: str
    symbol: str
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime]
    direction: str
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    position_size_pct: float
    pnl_gross: float
    pnl_net: float
    pnl_percentage: float
    holding_period_days: int
    prediction_confidence: float
    alpha_model_pop: float
    actual_outcome: bool
    market_regime: str
    features: Dict[str, Any]


class MLBacktester:
    """
    Comprehensive ML Model Backtesting Framework

    This system performs rigorous backtesting of machine learning trading models with:
    - Walk-forward analysis to prevent lookahead bias
    - Realistic transaction costs and slippage
    - Out-of-sample testing for model validation
    - Performance attribution and risk analysis
    - Feature importance tracking over time
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize ML Backtesting Framework"""

        self.config = config or BacktestConfig()

        # File management
        self.results_dir = Path("results/backtesting")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Database connection
        self.db_path = "data/marketpulse.db"
        self._init_database()

        # Results storage
        self.backtest_results = {}
        self.performance_metrics = {}
        self.feature_importance_history = {}
        self.trade_results = []

        logger.info("✅ ML Backtesting Framework initialized")

    def _init_database(self):
        """Initialize database tables for backtesting results"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Backtesting runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,

                -- Configuration
                initial_capital REAL NOT NULL,
                transaction_cost REAL,
                slippage REAL,
                max_position_size REAL,

                -- Results Summary
                total_return REAL,
                annual_return REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                calmar_ratio REAL,
                max_drawdown REAL,
                volatility REAL,
                win_rate REAL,
                profit_factor REAL,

                -- Trade Statistics
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                avg_win REAL,
                avg_loss REAL,
                avg_holding_days REAL,

                -- Risk Metrics
                var_95 REAL,
                cvar_95 REAL,
                beta REAL,
                alpha REAL,
                information_ratio REAL,

                -- Model Performance
                model_accuracy REAL,
                model_precision REAL,
                model_recall REAL,
                model_f1_score REAL,
                feature_count INTEGER,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Individual trade results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                trade_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,

                -- Trade Details
                entry_date DATETIME NOT NULL,
                exit_date DATETIME,
                direction TEXT CHECK (direction IN ('BUY', 'SELL', 'HOLD')),

                -- Prices and Execution
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity INTEGER,
                position_size_pct REAL,

                -- Performance
                pnl_gross REAL,
                pnl_net REAL,  -- After costs
                pnl_percentage REAL,
                holding_period_days INTEGER,

                -- Model Predictions
                predicted_direction TEXT,
                prediction_confidence REAL,
                alpha_model_pop REAL,  -- Probability of Profit
                forecast_price REAL,
                actual_outcome INTEGER, -- 1 if profitable, 0 if not

                -- Market Context
                market_regime TEXT,
                volatility_percentile REAL,
                volume_ratio REAL,
                rsi_14 REAL,
                macd_signal REAL,

                -- Feature Values (JSON)
                features TEXT,

                FOREIGN KEY (run_id) REFERENCES backtest_runs (run_id)
            )
        """)

        # Walk-forward analysis results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS walkforward_analysis (
                analysis_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,

                -- Time Period
                training_start DATE NOT NULL,
                training_end DATE NOT NULL,
                testing_start DATE NOT NULL,
                testing_end DATE NOT NULL,

                -- Model Performance
                in_sample_accuracy REAL,
                out_of_sample_accuracy REAL,
                performance_degradation REAL,
                model_auc REAL,
                model_precision REAL,
                model_recall REAL,

                -- Feature Importance (JSON)
                feature_importance TEXT,

                -- Returns
                in_sample_return REAL,
                out_of_sample_return REAL,
                benchmark_return REAL,
                trades_count INTEGER,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (run_id) REFERENCES backtest_runs (run_id)
            )
        """)

        conn.commit()
        conn.close()

        logger.info("✅ Backtesting database tables initialized")

    def run_walkforward_backtest(self,
                                 strategy_func: Callable,
                                 data: pd.DataFrame,
                                 symbol: str,
                                 strategy_name: str = "ML_Strategy") -> Dict[str, Any]:
        """
        Run walk-forward backtesting analysis

        This is the core backtesting method that:
        1. Splits data into overlapping train/test windows
        2. Trains model on historical data
        3. Tests on out-of-sample data
        4. Tracks performance degradation over time

        Args:
            strategy_func: Trading strategy function
            data: Historical OHLCV data with features
            symbol: Stock symbol
            strategy_name: Name of the strategy being tested

        Returns:
            Complete backtesting results
        """

        logger.info(f"🚀 Starting walk-forward backtest for {symbol} - {strategy_name}")

        # Generate unique run ID
        run_id = f"{strategy_name}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare data
        data = self._prepare_backtesting_data(data)

        if len(data) < self.config.walk_forward_window + 100:
            raise ValueError(f"Insufficient data: need {self.config.walk_forward_window + 100}, got {len(data)}")

        # Initialize results tracking
        all_trades = []
        equity_curve = []
        walkforward_results = []
        portfolio_value = self.config.initial_capital
        position = 0  # Current position size

        # Walk-forward loop
        start_idx = self.config.walk_forward_window

        while start_idx < len(data) - self.config.rebalance_frequency:
            # Define windows
            train_start_idx = max(0, start_idx - self.config.walk_forward_window)
            train_end_idx = start_idx
            test_start_idx = start_idx
            test_end_idx = min(start_idx + self.config.rebalance_frequency, len(data))

            # Extract data windows
            train_data = data.iloc[train_start_idx:train_end_idx].copy()
            test_data = data.iloc[test_start_idx:test_end_idx].copy()

            logger.info(f"📊 Walk-forward window: Train={len(train_data)} days, Test={len(test_data)} days")

            try:
                # Train models on historical data
                model_performance = self._train_models_for_period(train_data, symbol)

                # Generate signals for test period
                test_signals = self._generate_test_signals(test_data, symbol, strategy_func, model_performance)

                # Execute trades and track performance
                period_trades, period_equity = self._execute_test_trades(
                    test_signals, test_data, portfolio_value, position
                )

                # Update portfolio
                all_trades.extend(period_trades)
                equity_curve.extend(period_equity)

                if period_equity:
                    portfolio_value = period_equity[-1]['portfolio_value']
                    position = period_equity[-1]['position']

                # Store walk-forward results
                wf_result = {
                    'analysis_id': str(uuid.uuid4()),
                    'run_id': run_id,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'in_sample_accuracy': model_performance.get('accuracy', 0),
                    'out_of_sample_accuracy': self._calculate_period_accuracy(period_trades),
                    'model_auc': model_performance.get('auc', 0),
                    'period_return': (portfolio_value / self.config.initial_capital) - 1 if portfolio_value > 0 else -1,
                    'trades_count': len(period_trades),
                    'feature_importance': model_performance.get('feature_importance', {})
                }
                walkforward_results.append(wf_result)

                logger.info(f"✅ Period complete: Portfolio = ₹{portfolio_value:,.0f}, Trades = {len(period_trades)}")

            except Exception as e:
                logger.error(f"❌ Walk-forward period failed: {e}")
                continue

            # Move to next period
            start_idx = test_end_idx

        # Calculate comprehensive results
        results = self._calculate_backtest_results(
            run_id, strategy_name, symbol, data,
            all_trades, equity_curve, walkforward_results
        )

        # Save results to database
        self._save_backtest_results(results)

        # Generate performance plots
        self._generate_performance_plots(results)

        logger.info(f"🎉 Walk-forward backtest complete for {symbol}")
        logger.info(f"📊 Final Portfolio Value: ₹{portfolio_value:,.0f}")
        logger.info(f"📈 Total Return: {((portfolio_value / self.config.initial_capital) - 1) * 100:.2f}%")

        return results

    def _prepare_backtesting_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and enrich data for backtesting"""

        data = data.copy()

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
            else:
                data.index = pd.to_datetime(data.index)

        # Add comprehensive technical features
        data = self._add_technical_features(data)

        # Add market regime classification
        data = self._classify_market_regime(data)

        # Add forward returns for labeling
        data = self._add_forward_returns(data)

        # Clean data
        data = data.dropna()

        logger.info(f"✅ Data prepared: {len(data)} samples with {len(data.columns)} features")

        return data

    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical analysis features"""

        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            data[f'sma_{period}'] = ta.sma(data['close'], length=period)
            data[f'ema_{period}'] = ta.ema(data['close'], length=period)

        # Price relative to moving averages
        data['price_vs_sma20'] = data['close'] / data['sma_20'] - 1
        data['price_vs_sma50'] = data['close'] / data['sma_50'] - 1

        # Technical indicators
        data['rsi_14'] = ta.rsi(data['close'], length=14)
        data['rsi_21'] = ta.rsi(data['close'], length=21)

        # MACD
        macd = ta.macd(data['close'])
        if isinstance(macd, pd.DataFrame) and 'MACD_12_26_9' in macd.columns:
            data['macd'] = macd['MACD_12_26_9']
            data['macd_signal'] = macd['MACDs_12_26_9']
            data['macd_histogram'] = macd['MACDh_12_26_9']

        # Bollinger Bands
        bb = ta.bbands(data['close'], length=20)
        if isinstance(bb, pd.DataFrame) and 'BBU_20_2.0' in bb.columns:
            data['bb_upper'] = bb['BBU_20_2.0']
            data['bb_lower'] = bb['BBL_20_2.0']
            data['bb_middle'] = bb['BBM_20_2.0']
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

        # Volatility
        data['atr_14'] = ta.atr(data['high'], data['low'], data['close'], length=14)
        data['volatility_20'] = data['returns'].rolling(20).std()

        # Volume indicators (if volume exists)
        if 'volume' in data.columns:
            data['volume_sma_20'] = ta.sma(data['volume'], length=20)
            data['volume_ratio'] = data['volume'] / data['volume_sma_20']

        # Momentum indicators
        data['momentum_10'] = ta.mom(data['close'], length=10)
        data['roc_10'] = ta.roc(data['close'], length=10)

        # Stochastic
        stoch = ta.stoch(data['high'], data['low'], data['close'])
        if isinstance(stoch, pd.DataFrame) and 'STOCHk_14_3_3' in stoch.columns:
            data['stoch_k'] = stoch['STOCHk_14_3_3']
            data['stoch_d'] = stoch['STOCHd_14_3_3']

        # Williams %R
        data['williams_r'] = ta.willr(data['high'], data['low'], data['close'])

        # ADX
        adx_result = ta.adx(data['high'], data['low'], data['close'])
        if isinstance(adx_result, pd.DataFrame) and 'ADX_14' in adx_result.columns:
            data['adx_14'] = adx_result['ADX_14']

        return data

    def _classify_market_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify market regime for each period"""

        # Simple regime classification based on SMA trends
        data['price_above_sma_20'] = data['close'] > data['sma_20']
        data['price_above_sma_50'] = data['close'] > data['sma_50']
        data['sma_20_above_sma_50'] = data['sma_20'] > data['sma_50']

        # Volatility regime
        data['volatility_percentile'] = data['volatility_20'].rolling(60).rank(pct=True)
        data['high_volatility'] = data['volatility_percentile'] > 0.7

        # Market regime classification
        def classify_regime(row):
            if pd.isna(row.get('price_above_sma_20')) or pd.isna(row.get('sma_20_above_sma_50')):
                return 'UNKNOWN'
            elif row['price_above_sma_20'] and row['price_above_sma_50'] and row['sma_20_above_sma_50']:
                return 'BULL'
            elif not row['price_above_sma_20'] and not row['price_above_sma_50'] and not row['sma_20_above_sma_50']:
                return 'BEAR'
            else:
                return 'SIDEWAYS'

        data['market_regime'] = data.apply(classify_regime, axis=1)

        return data

    def _add_forward_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add forward-looking returns for signal labeling"""

        # Forward returns for different horizons
        for days in [1, 3, 5, 10, 20]:
            data[f'forward_return_{days}d'] = data['close'].shift(-days) / data['close'] - 1

        # Binary classification labels
        threshold = 0.02  # 2% threshold for significant moves

        data['label_1d'] = (data['forward_return_1d'] > threshold).astype(int)
        data['label_5d'] = (data['forward_return_5d'] > threshold).astype(int)
        data['label_20d'] = (data['forward_return_20d'] > threshold).astype(int)

        return data

    def _train_models_for_period(self, train_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train ML models for a specific backtesting period"""

        try:
            # Prepare features and labels
            feature_columns = [col for col in train_data.columns
                               if not col.startswith('forward_return') and
                               not col.startswith('label_') and
                               col not in ['open', 'high', 'low', 'close', 'volume'] and
                               not col.endswith('_above_sma_20') and
                               not col.endswith('_above_sma_50')]

            # Select only numeric columns
            numeric_columns = []
            for col in feature_columns:
                if col in train_data.columns and train_data[col].dtype in ['int64', 'float64']:
                    numeric_columns.append(col)

            if len(numeric_columns) < 5:
                logger.warning("Insufficient numeric features for training")
                return {'accuracy': 0.5, 'auc': 0.5, 'feature_importance': {}}

            X = train_data[numeric_columns].fillna(0)
            y = train_data['label_5d'].fillna(0)  # 5-day forward return classification

            # Remove any remaining NaN values
            valid_indices = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_indices]
            y = y[valid_indices]

            if len(X) < 50:
                logger.warning("Insufficient training data for period")
                return {'accuracy': 0.5, 'auc': 0.5, 'feature_importance': {}}

            # Train XGBoost model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )

            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)

            # Cross-validation
            cv_scores = []
            cv_auc_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train_cv, y_train_cv)

                # Predictions
                y_pred = model.predict(X_val_cv)
                y_pred_proba = model.predict_proba(X_val_cv)[:, 1]

                # Metrics
                accuracy = accuracy_score(y_val_cv, y_pred)
                auc = roc_auc_score(y_val_cv, y_pred_proba) if len(np.unique(y_val_cv)) > 1 else 0.5

                cv_scores.append(accuracy)
                cv_auc_scores.append(auc)

            # Final training on full dataset
            model.fit(X, y)

            # Feature importance
            feature_importance = dict(zip(numeric_columns, model.feature_importances_))

            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(),
                                             key=lambda x: x[1], reverse=True))

            return {
                'model': model,
                'accuracy': np.mean(cv_scores),
                'accuracy_std': np.std(cv_scores),
                'auc': np.mean(cv_auc_scores),
                'auc_std': np.std(cv_auc_scores),
                'feature_importance': feature_importance,
                'feature_columns': numeric_columns,
                'training_samples': len(X)
            }

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {'accuracy': 0.5, 'auc': 0.5, 'feature_importance': {}}

    def _generate_test_signals(self, test_data: pd.DataFrame, symbol: str,
                               strategy_func: Callable, model_performance: Dict) -> List[Dict[str, Any]]:
        """Generate trading signals for test period using trained models"""

        signals = []
        model = model_performance.get('model')
        feature_columns = model_performance.get('feature_columns', [])

        for i, (timestamp, row) in enumerate(test_data.iterrows()):
            try:
                # Prepare signal features
                signal_features = {
                    'symbol': symbol,
                    'entry_price': row['close'],
                    'rsi_14': row.get('rsi_14', 50),
                    'macd_signal': row.get('macd_signal', 0),
                    'bb_position': row.get('bb_position', 0.5),
                    'volume_ratio': row.get('volume_ratio', 1),
                    'market_regime': row.get('market_regime', 'SIDEWAYS'),
                    'volatility_percentile': row.get('volatility_percentile', 0.5)
                }

                # Get ML model prediction if available
                ml_prediction = 0.5  # Default
                ml_confidence = 0.5

                if model and feature_columns:
                    try:
                        # Prepare features for model
                        model_features = []
                        for col in feature_columns:
                            value = row.get(col, 0)
                            if pd.isna(value):
                                value = 0
                            model_features.append(value)

                        if len(model_features) == len(feature_columns):
                            model_input = np.array(model_features).reshape(1, -1)
                            ml_prediction = model.predict_proba(model_input)[0][1]  # Probability of class 1
                            ml_confidence = max(ml_prediction, 1 - ml_prediction)  # Distance from 0.5

                    except Exception as e:
                        logger.warning(f"ML prediction failed: {e}")

                # Apply strategy function to generate signal
                strategy_signal = strategy_func(row, signal_features, ml_prediction)

                # Create comprehensive signal
                signal = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'direction': strategy_signal.get('direction', 'HOLD'),
                    'confidence': strategy_signal.get('confidence', 0.5),
                    'entry_price': row['close'],
                    'ml_prediction': ml_prediction,
                    'ml_confidence': ml_confidence,
                    'market_regime': row.get('market_regime', 'SIDEWAYS'),
                    'features': signal_features,
                    'actual_forward_return': row.get('forward_return_5d', 0)
                }

                signals.append(signal)

            except Exception as e:
                logger.error(f"Signal generation failed for {timestamp}: {e}")
                continue

        return signals

    def _execute_test_trades(self, signals: List[Dict[str, Any]],
                             test_data: pd.DataFrame,
                             initial_portfolio_value: float,
                             initial_position: float) -> Tuple[List[Dict], List[Dict]]:
        """Execute trades based on signals and track performance"""

        trades = []
        equity_curve = []

        portfolio_value = initial_portfolio_value
        position = initial_position
        cash = portfolio_value * (1 - abs(position))

        current_trade = None

        for i, signal in enumerate(signals):
            try:
                timestamp = signal['timestamp']
                price = signal['entry_price']
                direction = signal['direction']
                confidence = signal['confidence']
                ml_prediction = signal.get('ml_prediction', 0.5)

                # Position sizing based on confidence and ML prediction
                base_size = min(self.config.max_position_size, confidence * 0.2)

                # Adjust size based on ML prediction strength
                if ml_prediction > 0.6:
                    adjusted_size = base_size * (ml_prediction * 2)  # Scale up for high confidence
                elif ml_prediction < 0.4:
                    adjusted_size = base_size * ((1 - ml_prediction) * 2)  # Scale up for high confidence short
                else:
                    adjusted_size = base_size * 0.5  # Reduce for low confidence

                # Cap at maximum position size
                adjusted_size = min(adjusted_size, self.config.max_position_size)

                # Trading logic
                new_position = 0

                if direction == 'BUY' and ml_prediction > 0.55:
                    new_position = adjusted_size
                elif direction == 'SELL' and ml_prediction < 0.45:
                    new_position = -adjusted_size
                else:
                    new_position = 0  # HOLD or low confidence

                # Execute position change
                if abs(new_position - position) > 0.01:
                    # Close existing position if any
                    if current_trade and abs(position) > 0.01:
                        exit_price = price
                        gross_pnl = position * portfolio_value * ((exit_price / current_trade['entry_price']) - 1)

                        # Apply transaction costs
                        transaction_cost = abs(position * portfolio_value) * self.config.transaction_cost
                        slippage_cost = abs(position * portfolio_value) * self.config.slippage
                        net_pnl = gross_pnl - transaction_cost - slippage_cost

                        # Calculate actual outcome
                        actual_return = current_trade.get('actual_forward_return', 0)
                        actual_profitable = 1 if actual_return > 0.02 else 0

                        # Record completed trade
                        trade_record = {
                            **current_trade,
                            'trade_id': f"{signal['symbol']}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                            'exit_timestamp': timestamp,
                            'exit_price': exit_price,
                            'pnl_gross': gross_pnl,
                            'pnl_net': net_pnl,
                            'pnl_percentage': (net_pnl / (abs(position) * portfolio_value)) * 100,
                            'holding_period_days': (timestamp - current_trade['entry_timestamp']).days,
                            'actual_outcome': actual_profitable
                        }

                        trades.append(trade_record)

                        # Update portfolio
                        portfolio_value += net_pnl
                        cash = portfolio_value * (1 - abs(new_position))

                        current_trade = None

                    # Open new position
                    if abs(new_position) > 0.01:
                        # Apply transaction costs for opening
                        transaction_cost = abs(new_position * portfolio_value) * self.config.transaction_cost
                        slippage_cost = abs(new_position * portfolio_value) * self.config.slippage
                        effective_entry_price = price * (
                            1 + self.config.slippage if new_position > 0 else 1 - self.config.slippage)

                        current_trade = {
                            'symbol': signal['symbol'],
                            'entry_timestamp': timestamp,
                            'entry_price': effective_entry_price,
                            'direction': 'BUY' if new_position > 0 else 'SELL',
                            'position_size_pct': abs(new_position),
                            'predicted_direction': direction,
                            'prediction_confidence': confidence,
                            'ml_prediction': ml_prediction,
                            'market_regime': signal.get('market_regime'),
                            'features': json.dumps(signal.get('features', {})),
                            'actual_forward_return': signal.get('actual_forward_return', 0)
                        }

                        # Update cash after costs
                        cash -= (transaction_cost + slippage_cost)
                        portfolio_value = cash / (1 - abs(new_position)) if abs(new_position) < 1 else cash

                    position = new_position

                # Track equity curve
                if abs(position) > 0.01 and current_trade:
                    # Mark-to-market current position
                    unrealized_pnl = position * portfolio_value * ((price / current_trade['entry_price']) - 1)
                    current_portfolio_value = portfolio_value + unrealized_pnl
                else:
                    current_portfolio_value = portfolio_value

                equity_curve.append({
                    'timestamp': timestamp,
                    'portfolio_value': current_portfolio_value,
                    'position': position,
                    'cash': cash,
                    'price': price
                })

            except Exception as e:
                logger.error(f"Trade execution failed at {signal.get('timestamp')}: {e}")
                continue

        # Close any remaining position at the end
        if current_trade and abs(position) > 0.01:
            final_price = test_data['close'].iloc[-1]
            gross_pnl = position * portfolio_value * ((final_price / current_trade['entry_price']) - 1)
            transaction_cost = abs(position * portfolio_value) * self.config.transaction_cost
            slippage_cost = abs(position * portfolio_value) * self.config.slippage
            net_pnl = gross_pnl - transaction_cost - slippage_cost

            actual_return = current_trade.get('actual_forward_return', 0)
            actual_profitable = 1 if actual_return > 0.02 else 0

            trade_record = {
                **current_trade,
                'trade_id': f"{current_trade['symbol']}_{test_data.index[-1].strftime('%Y%m%d_%H%M%S')}",
                'exit_timestamp': test_data.index[-1],
                'exit_price': final_price,
                'pnl_gross': gross_pnl,
                'pnl_net': net_pnl,
                'pnl_percentage': (net_pnl / (abs(position) * portfolio_value)) * 100,
                'holding_period_days': (test_data.index[-1] - current_trade['entry_timestamp']).days,
                'actual_outcome': actual_profitable
            }

            trades.append(trade_record)
            portfolio_value += net_pnl

        return trades, equity_curve

    def _calculate_period_accuracy(self, period_trades: List[Dict]) -> float:
        """Calculate prediction accuracy for a period"""

        if not period_trades:
            return 0.0

        correct_predictions = 0
        total_predictions = 0

        for trade in period_trades:
            if 'ml_prediction' in trade and 'actual_outcome' in trade:
                predicted_positive = trade['ml_prediction'] > 0.5
                actual_positive = trade['actual_outcome'] == 1

                if predicted_positive == actual_positive:
                    correct_predictions += 1

                total_predictions += 1

        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

    def _calculate_backtest_results(self, run_id: str, strategy_name: str, symbol: str,
                                    full_data: pd.DataFrame, all_trades: List[Dict],
                                    equity_curve: List[Dict],
                                    walkforward_results: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive backtesting performance metrics"""

        if not equity_curve:
            return {'error': 'No equity curve data available'}

        # Convert to DataFrames
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

        # Basic performance metrics
        initial_value = self.config.initial_capital
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_value) - 1

        # Time-based metrics
        start_date = equity_df['timestamp'].iloc[0]
        end_date = equity_df['timestamp'].iloc[-1]
        days = (end_date - start_date).days
        years = days / 365.25

        annual_return = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0

        # Risk metrics
        returns = equity_df['portfolio_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Sharpe ratio
        sharpe_ratio = (annual_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0

        # Sortino ratio (using downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (
                                    annual_return - self.config.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Maximum drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        if not trades_df.empty:
            winning_trades = len(trades_df[trades_df['pnl_net'] > 0])
            losing_trades = len(trades_df[trades_df['pnl_net'] <= 0])
            total_trades = len(trades_df)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            avg_win = trades_df[trades_df['pnl_net'] > 0]['pnl_net'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl_net'] <= 0]['pnl_net'].mean() if losing_trades > 0 else 0
            avg_holding_days = trades_df['holding_period_days'].mean() if total_trades > 0 else 0

            profit_factor = abs(
                avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 0
        else:
            winning_trades = losing_trades = total_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = avg_holding_days = 0

        # Additional risk metrics
        # VaR (Value at Risk)
        var_95 = returns.quantile(0.05) * np.sqrt(252)  # 95% VaR annualized

        # CVaR (Conditional Value at Risk)
        cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * np.sqrt(252)

        # Model performance metrics
        if walkforward_results:
            avg_in_sample_accuracy = np.mean([w.get('in_sample_accuracy', 0) for w in walkforward_results])
            avg_out_sample_accuracy = np.mean([w.get('out_of_sample_accuracy', 0) for w in walkforward_results])
            avg_model_auc = np.mean([w.get('model_auc', 0) for w in walkforward_results])
        else:
            avg_in_sample_accuracy = avg_out_sample_accuracy = avg_model_auc = 0

        # Calculate model prediction accuracy from actual trades
        model_accuracy = model_precision = model_recall = model_f1 = 0

        if not trades_df.empty and 'ml_prediction' in trades_df.columns and 'actual_outcome' in trades_df.columns:
            # Convert predictions to binary
            predicted_labels = (trades_df['ml_prediction'] > 0.5).astype(int)
            actual_labels = trades_df['actual_outcome'].astype(int)

            # Calculate metrics
            model_accuracy = accuracy_score(actual_labels, predicted_labels)

            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                model_precision = precision_score(actual_labels, predicted_labels, average='binary')
                model_recall = recall_score(actual_labels, predicted_labels, average='binary')
                model_f1 = f1_score(actual_labels, predicted_labels, average='binary')
            except:
                pass

        # Compile results
        results = {
            'run_id': run_id,
            'strategy_name': strategy_name,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'total_days': days,

            # Performance
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,

            # Trade statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_holding_days': avg_holding_days,
            'profit_factor': profit_factor,

            # Model performance
            'model_accuracy': model_accuracy,
            'model_precision': model_precision,
            'model_recall': model_recall,
            'model_f1_score': model_f1,
            'avg_in_sample_accuracy': avg_in_sample_accuracy,
            'avg_out_sample_accuracy': avg_out_sample_accuracy,
            'avg_model_auc': avg_model_auc,

            # Raw data
            'equity_curve': equity_df.to_dict('records'),
            'trades': trades_df.to_dict('records') if not trades_df.empty else [],
            'walkforward_results': walkforward_results,

            # Additional analysis
            'feature_importance_history': self._aggregate_feature_importance(walkforward_results)
        }

        return results

    def _aggregate_feature_importance(self, walkforward_results: List[Dict]) -> Dict[str, float]:
        """Aggregate feature importance across all walk-forward periods"""

        feature_importance_agg = {}

        for wf_result in walkforward_results:
            feature_importance = wf_result.get('feature_importance', {})

            for feature, importance in feature_importance.items():
                if feature not in feature_importance_agg:
                    feature_importance_agg[feature] = []
                feature_importance_agg[feature].append(importance)

        # Calculate mean importance for each feature
        aggregated = {}
        for feature, importance_list in feature_importance_agg.items():
            aggregated[feature] = np.mean(importance_list)

        # Sort by importance
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))

    def _save_backtest_results(self, results: Dict[str, Any]):
        """Save backtest results to database"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert main backtest run
            cursor.execute("""
                INSERT INTO backtest_runs (
                    run_id, strategy_name, symbol, start_date, end_date,
                    initial_capital, transaction_cost, slippage, max_position_size,
                    total_return, annual_return, sharpe_ratio, sortino_ratio, calmar_ratio,
                    max_drawdown, volatility, win_rate, profit_factor,
                    total_trades, winning_trades, losing_trades,
                    avg_win, avg_loss, avg_holding_days,
                    var_95, cvar_95, model_accuracy, model_precision, model_recall, model_f1_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                results['run_id'], results['strategy_name'], results['symbol'],
                results['start_date'], results['end_date'],
                results['initial_capital'], self.config.transaction_cost,
                self.config.slippage, self.config.max_position_size,
                results['total_return'], results['annual_return'],
                results['sharpe_ratio'], results['sortino_ratio'], results['calmar_ratio'],
                results['max_drawdown'], results['volatility'],
                results['win_rate'], results['profit_factor'],
                results['total_trades'], results['winning_trades'], results['losing_trades'],
                results['avg_win'], results['avg_loss'], results['avg_holding_days'],
                results['var_95'], results['cvar_95'],
                results['model_accuracy'], results['model_precision'],
                results['model_recall'], results['model_f1_score']
            ))

            # Insert individual trades
            for trade in results.get('trades', []):
                cursor.execute("""
                    INSERT INTO backtest_trades (
                        trade_id, run_id, symbol, entry_date, exit_date, direction,
                        entry_price, exit_price, position_size_pct,
                        pnl_gross, pnl_net, pnl_percentage, holding_period_days,
                        prediction_confidence, alpha_model_pop, market_regime,
                        actual_outcome, features
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.get('trade_id'), results['run_id'], trade.get('symbol'),
                    trade.get('entry_timestamp'), trade.get('exit_timestamp'), trade.get('direction'),
                    trade.get('entry_price'), trade.get('exit_price'), trade.get('position_size_pct'),
                    trade.get('pnl_gross'), trade.get('pnl_net'), trade.get('pnl_percentage'),
                    trade.get('holding_period_days'), trade.get('prediction_confidence'),
                    trade.get('ml_prediction'), trade.get('market_regime'),
                    trade.get('actual_outcome'), trade.get('features', '{}')
                ))

            # Insert walk-forward analysis results
            for wf_result in results.get('walkforward_results', []):
                cursor.execute("""
                    INSERT INTO walkforward_analysis (
                        analysis_id, run_id, training_start, training_end, 
                        testing_start, testing_end, in_sample_accuracy, 
                        out_of_sample_accuracy, model_auc, trades_count,
                        feature_importance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    wf_result.get('analysis_id'), results['run_id'],
                    wf_result.get('train_start'), wf_result.get('train_end'),
                    wf_result.get('test_start'), wf_result.get('test_end'),
                    wf_result.get('in_sample_accuracy'), wf_result.get('out_of_sample_accuracy'),
                    wf_result.get('model_auc'), wf_result.get('trades_count'),
                    json.dumps(wf_result.get('feature_importance', {}))
                ))

            conn.commit()
            conn.close()

            logger.info(f"✅ Backtest results saved: {results['run_id']}")

        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")

    def _generate_performance_plots(self, results: Dict[str, Any]):
        """Generate comprehensive performance visualization plots"""

        try:
            # Create plots directory
            plots_dir = self.results_dir / results['run_id']
            plots_dir.mkdir(exist_ok=True)

            # Equity curve plot
            self._plot_equity_curve(results, plots_dir)

            # Drawdown plot
            self._plot_drawdown(results, plots_dir)

            # Trade analysis plot
            self._plot_trade_analysis(results, plots_dir)

            # Feature importance plot
            self._plot_feature_importance(results, plots_dir)

            # Model performance over time
            self._plot_model_performance(results, plots_dir)

            logger.info(f"✅ Performance plots saved to {plots_dir}")

        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

    def _plot_equity_curve(self, results: Dict[str, Any], plots_dir: Path):
        """Plot equity curve and benchmark comparison"""

        equity_df = pd.DataFrame(results['equity_curve'])

        if equity_df.empty:
            return

        plt.figure(figsize=(12, 8))

        # Main equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_df['timestamp'], equity_df['portfolio_value'],
                 label='Strategy', color='blue', linewidth=2)

        # Calculate buy-and-hold benchmark
        initial_price = equity_df['price'].iloc[0]
        benchmark_value = results['initial_capital'] * (equity_df['price'] / initial_price)
        plt.plot(equity_df['timestamp'], benchmark_value,
                 label='Buy & Hold', color='gray', linewidth=1, alpha=0.7)

        plt.title(f"Equity Curve - {results['strategy_name']} ({results['symbol']})")
        plt.ylabel('Portfolio Value (₹)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Position over time
        plt.subplot(2, 1, 2)
        plt.plot(equity_df['timestamp'], equity_df['position'],
                 color='orange', alpha=0.7)
        plt.fill_between(equity_df['timestamp'], 0, equity_df['position'],
                         alpha=0.3, color='orange')
        plt.title('Position Size Over Time')
        plt.ylabel('Position Size')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_drawdown(self, results: Dict[str, Any], plots_dir: Path):
        """Plot drawdown analysis"""

        equity_df = pd.DataFrame(results['equity_curve'])

        if equity_df.empty:
            return

        # Calculate drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max * 100

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(equity_df['timestamp'], equity_df['portfolio_value'],
                 color='blue', linewidth=2)
        plt.fill_between(equity_df['timestamp'], rolling_max, equity_df['portfolio_value'],
                         where=(equity_df['portfolio_value'] < rolling_max),
                         color='red', alpha=0.3, label='Drawdown')
        plt.title('Portfolio Value with Drawdown Periods')
        plt.ylabel('Portfolio Value (₹)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.fill_between(equity_df['timestamp'], 0, drawdown,
                         color='red', alpha=0.6)
        plt.title(f'Drawdown Analysis (Max: {results["max_drawdown"] * 100:.2f}%)')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'drawdown_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_trade_analysis(self, results: Dict[str, Any], plots_dir: Path):
        """Plot trade analysis and distribution"""

        trades_df = pd.DataFrame(results['trades'])

        if trades_df.empty:
            return

        plt.figure(figsize=(15, 10))

        # PnL distribution
        plt.subplot(2, 3, 1)
        trades_df['pnl_percentage'].hist(bins=30, alpha=0.7, color='blue')
        plt.title('PnL Distribution (%)')
        plt.xlabel('PnL (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Win/Loss analysis
        plt.subplot(2, 3, 2)
        wins = trades_df[trades_df['pnl_net'] > 0]['pnl_percentage']
        losses = trades_df[trades_df['pnl_net'] <= 0]['pnl_percentage']

        plt.hist([wins, losses], bins=20, alpha=0.7,
                 label=['Wins', 'Losses'], color=['green', 'red'])
        plt.title('Win/Loss Distribution')
        plt.xlabel('PnL (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Holding period analysis
        plt.subplot(2, 3, 3)
        trades_df['holding_period_days'].hist(bins=20, alpha=0.7, color='orange')
        plt.title('Holding Period Distribution')
        plt.xlabel('Days')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # ML Prediction vs Actual Outcome
        plt.subplot(2, 3, 4)
        if 'ml_prediction' in trades_df.columns and 'actual_outcome' in trades_df.columns:
            winning_preds = trades_df[trades_df['actual_outcome'] == 1]['ml_prediction']
            losing_preds = trades_df[trades_df['actual_outcome'] == 0]['ml_prediction']

            plt.hist([winning_preds, losing_preds], bins=20, alpha=0.7,
                     label=['Actual Winners', 'Actual Losers'], color=['green', 'red'])
            plt.title('ML Prediction Distribution by Outcome')
            plt.xlabel('ML Prediction')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # PnL over time
        plt.subplot(2, 3, 5)
        trades_df['cumulative_pnl'] = trades_df['pnl_net'].cumsum()
        plt.plot(trades_df.index, trades_df['cumulative_pnl'],
                 color='blue', linewidth=2)
        plt.title('Cumulative PnL from Trades')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative PnL (₹)')
        plt.grid(True, alpha=0.3)

        # Market Regime Performance
        plt.subplot(2, 3, 6)
        if 'market_regime' in trades_df.columns:
            regime_pnl = trades_df.groupby('market_regime')['pnl_percentage'].mean()
            regime_pnl.plot(kind='bar', color=['blue', 'red', 'green'])
            plt.title('Average PnL by Market Regime')
            plt.xlabel('Market Regime')
            plt.ylabel('Average PnL (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'trade_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, results: Dict[str, Any], plots_dir: Path):
        """Plot feature importance analysis"""

        feature_importance = results.get('feature_importance_history', {})

        if not feature_importance:
            return

        # Top 15 features
        top_features = dict(list(feature_importance.items())[:15])

        plt.figure(figsize=(10, 8))

        features = list(top_features.keys())
        importance = list(top_features.values())

        plt.barh(range(len(features)), importance, color='skyblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Average Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_performance(self, results: Dict[str, Any], plots_dir: Path):
        """Plot model performance over time"""

        walkforward_results = results.get('walkforward_results', [])

        if not walkforward_results:
            return

        wf_df = pd.DataFrame(walkforward_results)

        plt.figure(figsize=(12, 8))

        # In-sample vs Out-of-sample accuracy
        plt.subplot(2, 2, 1)
        plt.plot(range(len(wf_df)), wf_df['in_sample_accuracy'],
                 'o-', label='In-Sample', color='blue')
        plt.plot(range(len(wf_df)), wf_df['out_of_sample_accuracy'],
                 'o-', label='Out-of-Sample', color='red')
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Period')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Model AUC over time
        plt.subplot(2, 2, 2)
        plt.plot(range(len(wf_df)), wf_df.get('model_auc', [0.5] * len(wf_df)),
                 'o-', color='green')
        plt.title('Model AUC Over Time')
        plt.xlabel('Period')
        plt.ylabel('AUC')
        plt.grid(True, alpha=0.3)

        # Performance degradation
        plt.subplot(2, 2, 3)
        degradation = wf_df['in_sample_accuracy'] - wf_df['out_of_sample_accuracy']
        plt.plot(range(len(wf_df)), degradation, 'o-', color='orange')
        plt.title('Performance Degradation (Overfitting)')
        plt.xlabel('Period')
        plt.ylabel('In-Sample - Out-of-Sample Accuracy')
        plt.grid(True, alpha=0.3)

        # Number of trades per period
        plt.subplot(2, 2, 4)
        plt.bar(range(len(wf_df)), wf_df.get('trades_count', [0] * len(wf_df)),
                color='purple', alpha=0.7)
        plt.title('Trades per Walk-Forward Period')
        plt.xlabel('Period')
        plt.ylabel('Number of Trades')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def compare_strategies(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple strategy backtest results"""

        comparison_data = []

        for result in results_list:
            comparison_data.append({
                'Strategy': result['strategy_name'],
                'Symbol': result['symbol'],
                'Total Return': f"{result['total_return']:.2%}",
                'Annual Return': f"{result['annual_return']:.2%}",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{result['max_drawdown']:.2%}",
                'Win Rate': f"{result['win_rate']:.1%}",
                'Total Trades': result['total_trades'],
                'Model Accuracy': f"{result['model_accuracy']:.1%}",
                'Profit Factor': f"{result['profit_factor']:.2f}"
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Save comparison
        comparison_path = self.results_dir / 'strategy_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)

        logger.info(f"✅ Strategy comparison saved to {comparison_path}")

        return comparison_df


# Example strategy functions for testing
def simple_ml_strategy(row: pd.Series, signal_features: Dict, ml_prediction: float) -> Dict[str, Any]:
    """Simple ML-based strategy example"""

    # Only trade if ML prediction is significantly different from 50%
    if ml_prediction > 0.6:
        direction = 'BUY'
        confidence = min(1.0, (ml_prediction - 0.5) * 2)
    elif ml_prediction < 0.4:
        direction = 'SELL'
        confidence = min(1.0, (0.5 - ml_prediction) * 2)
    else:
        direction = 'HOLD'
        confidence = 0.0

    # Additional technical filters
    rsi = signal_features.get('rsi_14', 50)
    bb_position = signal_features.get('bb_position', 0.5)

    # Reduce confidence in extreme conditions
    if direction == 'BUY' and rsi > 80:
        confidence *= 0.5
    if direction == 'SELL' and rsi < 20:
        confidence *= 0.5

    # Boost confidence if Bollinger Band position supports signal
    if direction == 'BUY' and bb_position < 0.2:
        confidence *= 1.2
    if direction == 'SELL' and bb_position > 0.8:
        confidence *= 1.2

    confidence = min(1.0, confidence)

    return {
        'direction': direction,
        'confidence': confidence,
        'reasoning': f"ML: {ml_prediction:.3f}, RSI: {rsi:.1f}, BB: {bb_position:.2f}"
    }


# Example usage
if __name__ == "__main__":
    """Example usage of Complete ML Backtesting Framework"""

    print("🚀 Complete ML Backtesting Framework Demo")
    print("=" * 60)

    # Initialize backtester with custom configuration
    config = BacktestConfig(
        initial_capital=100000.0,
        transaction_cost=0.001,
        slippage=0.0005,
        max_position_size=0.15,
        walk_forward_window=500,
        rebalance_frequency=30
    )

    backtester = MLBacktester(config)

    # Create realistic sample data
    print("\n📊 Generating sample market data...")

    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-12-01', freq='D')

    # Generate realistic price data with trends
    returns = np.random.randn(len(dates)) * 0.02
    trend = np.sin(np.arange(len(dates)) * 2 * np.pi / 252) * 0.001
    regime_changes = np.where(np.random.randn(len(dates)) > 1.5, 0.01, 0)

    cumulative_returns = np.cumsum(returns + trend + regime_changes)
    prices = 2500 * np.exp(cumulative_returns)

    # Generate OHLCV data
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(len(dates)) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)

    # Ensure OHLC logic is correct
    sample_data['high'] = np.maximum.reduce([sample_data['open'], sample_data['close'], sample_data['high']])
    sample_data['low'] = np.minimum.reduce([sample_data['open'], sample_data['close'], sample_data['low']])

    symbol = "RELIANCE"

    print(f"📈 Sample data created:")
    print(f"  Period: {sample_data.index[0]} to {sample_data.index[-1]}")
    print(f"  Total days: {len(sample_data)}")
    print(f"  Price range: ₹{sample_data['close'].min():.0f} - ₹{sample_data['close'].max():.0f}")

    try:
        # Run comprehensive backtest
        results = backtester.run_walkforward_backtest(
            strategy_func=simple_ml_strategy,
            data=sample_data,
            symbol=symbol,
            strategy_name="Simple_ML_Strategy"
        )

        print(f"\n📈 Backtest Results:")
        print("=" * 40)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annual Return: {results['annual_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Model Accuracy: {results['model_accuracy']:.1%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n🎉 Complete ML Backtesting Framework demo finished!")
    print(f"📂 Results saved in: {backtester.results_dir}")
    print(f"📝 Next Steps:")
    print(f"   1. Integrate with real Alpha Model predictions")
    print(f"   2. Add more sophisticated strategy functions")
    print(f"   3. Implement parameter optimization")
    print(f"   4. Build real-time monitoring dashboard")