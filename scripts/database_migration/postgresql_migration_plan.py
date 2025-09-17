# scripts/database_migration/postgresql_migration_plan.py
"""
PostgreSQL Migration Plan - Phase 1 Day 9
Database architecture planning for advanced ML model integration

Migration from SQLite to PostgreSQL for:
- Time-series optimization with TimescaleDB extension
- Advanced ML model storage and versioning
- High-performance real-time data processing
- Enterprise-grade scalability and reliability
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json

# Database libraries
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sqlite3
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgreSQLMigrationPlan:
    """
    Comprehensive PostgreSQL migration planning and execution

    Features:
    - Time-series optimized schema design with TimescaleDB
    - ML model metadata and versioning system
    - High-performance indexing strategy
    - Data migration utilities from SQLite
    - Connection pooling and optimization
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize PostgreSQL migration planner"""

        # Default configuration
        self.config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'marketpulse'),
            'user': os.getenv('POSTGRES_USER', 'marketpulse_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'secure_password'),
            'timescaledb_enabled': True,
            'connection_pool_size': 10
        }

        if config:
            self.config.update(config)

        # Migration tracking
        self.migration_log = []
        self.sqlite_db_path = "data/marketpulse.db"

        logger.info("✅ PostgreSQL Migration Planner initialized")

    def generate_migration_schema(self) -> str:
        """
        Generate comprehensive PostgreSQL schema with time-series optimization

        Returns:
            SQL script for complete database schema
        """

        schema_sql = """
-- MarketPulse PostgreSQL Schema with TimescaleDB Extensions
-- Phase 1 Day 9: Advanced ML Model Integration
-- Generated: {timestamp}

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;

-- =============================================================================
-- MARKET DATA TABLES (Time-series optimized)
-- =============================================================================

-- Real-time market data with TimescaleDB hypertable
CREATE TABLE market_data.price_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- OHLCV data
    open DECIMAL(12, 4) NOT NULL,
    high DECIMAL(12, 4) NOT NULL,
    low DECIMAL(12, 4) NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    volume BIGINT NOT NULL,

    -- Additional market metrics
    vwap DECIMAL(12, 4),
    trades_count INTEGER,

    -- Data quality indicators
    data_source VARCHAR(50),
    quality_score DECIMAL(3, 2) DEFAULT 1.0,

    PRIMARY KEY (timestamp, symbol, timeframe)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_data.price_data', 'timestamp', 
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE);

-- Technical indicators cache
CREATE TABLE market_data.technical_indicators (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Moving averages
    sma_5 DECIMAL(12, 4),
    sma_10 DECIMAL(12, 4),
    sma_20 DECIMAL(12, 4),
    sma_50 DECIMAL(12, 4),
    sma_200 DECIMAL(12, 4),
    ema_12 DECIMAL(12, 4),
    ema_26 DECIMAL(12, 4),

    -- Momentum indicators
    rsi_14 DECIMAL(5, 2),
    rsi_21 DECIMAL(5, 2),
    macd DECIMAL(8, 4),
    macd_signal DECIMAL(8, 4),
    macd_histogram DECIMAL(8, 4),

    -- Bollinger Bands
    bb_upper DECIMAL(12, 4),
    bb_lower DECIMAL(12, 4),
    bb_middle DECIMAL(12, 4),
    bb_width DECIMAL(8, 4),
    bb_position DECIMAL(5, 4),

    -- Volatility indicators
    atr_14 DECIMAL(8, 4),
    adx_14 DECIMAL(5, 2),

    -- Volume indicators
    volume_sma_20 BIGINT,
    volume_ratio DECIMAL(6, 3),

    -- Stochastic
    stoch_k DECIMAL(5, 2),
    stoch_d DECIMAL(5, 2),

    -- Other indicators
    williams_r DECIMAL(6, 2),
    cci_14 DECIMAL(8, 2),

    -- Calculation metadata
    calculation_timestamp TIMESTAMPTZ DEFAULT NOW(),
    indicators_version VARCHAR(10) DEFAULT '1.0',

    PRIMARY KEY (timestamp, symbol, timeframe)
);

-- Convert to hypertable
SELECT create_hypertable('market_data.technical_indicators', 'timestamp',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE);

-- Market regime classification
CREATE TABLE market_data.market_regimes (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Regime classification
    regime VARCHAR(20) NOT NULL, -- BULL, BEAR, SIDEWAYS, VOLATILE
    regime_confidence DECIMAL(3, 2),
    regime_strength DECIMAL(3, 2),

    -- Supporting metrics
    trend_direction DECIMAL(3, 2), -- -1 to 1
    volatility_percentile DECIMAL(3, 2),
    volume_profile VARCHAR(20), -- HIGH, NORMAL, LOW

    -- Classification metadata
    model_version VARCHAR(10),
    classification_timestamp TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('market_data.market_regimes', 'timestamp',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE);

-- =============================================================================
-- ML MODELS SCHEMA (Advanced model management)
-- =============================================================================

-- Model registry for version control
CREATE TABLE ml_models.model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- ALPHA_MODEL, LSTM, PROPHET, ENSEMBLE
    model_version VARCHAR(20) NOT NULL,

    -- Model metadata
    description TEXT,
    author VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Training information
    training_data_start DATE,
    training_data_end DATE,
    training_samples INTEGER,
    feature_count INTEGER,

    -- Model performance
    accuracy DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    roc_auc DECIMAL(5, 4),

    -- Cross-validation results
    cv_mean_accuracy DECIMAL(5, 4),
    cv_std_accuracy DECIMAL(5, 4),
    cv_folds INTEGER,

    -- Model configuration (JSON)
    hyperparameters JSONB,
    feature_importance JSONB,

    -- Status
    status VARCHAR(20) DEFAULT 'TRAINING', -- TRAINING, ACTIVE, DEPRECATED, FAILED
    is_production BOOLEAN DEFAULT FALSE,

    -- Unique constraint on name and version
    UNIQUE(model_name, model_version)
);

-- Model artifacts storage (for large model files)
CREATE TABLE ml_models.model_artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_models.model_registry(model_id),

    -- Artifact details
    artifact_type VARCHAR(50) NOT NULL, -- MODEL_WEIGHTS, SCALER, PREPROCESSING
    artifact_name VARCHAR(200) NOT NULL,
    file_path TEXT,
    file_size_mb DECIMAL(10, 2),
    checksum VARCHAR(64),

    -- Storage information
    storage_backend VARCHAR(50) DEFAULT 'LOCAL', -- LOCAL, S3, GCS
    compression VARCHAR(20), -- GZIP, NONE

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(model_id, artifact_type, artifact_name)
);

-- Model performance tracking over time
CREATE TABLE ml_models.model_performance_history (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_models.model_registry(model_id),

    -- Evaluation period
    evaluation_date DATE NOT NULL,
    evaluation_period_days INTEGER NOT NULL,

    -- Performance metrics
    accuracy DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    roc_auc DECIMAL(5, 4),

    -- Business metrics
    profit_accuracy DECIMAL(5, 4), -- % of profitable predictions
    signal_precision DECIMAL(5, 4), -- True positive rate for signals
    false_positive_rate DECIMAL(5, 4),

    -- Data statistics
    predictions_made INTEGER,
    correct_predictions INTEGER,

    -- Drift detection
    feature_drift_score DECIMAL(5, 4),
    concept_drift_detected BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature store for ML features
CREATE TABLE ml_models.feature_store (
    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- Feature metadata
    feature_set_version VARCHAR(20) NOT NULL,
    feature_hash VARCHAR(64), -- For deduplication

    -- Core features (normalized JSON for flexibility)
    features JSONB NOT NULL,

    -- Feature statistics
    feature_count INTEGER,
    null_feature_count INTEGER,
    feature_quality_score DECIMAL(3, 2),

    -- Processing metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processing_time_ms INTEGER,

    PRIMARY KEY (timestamp, symbol, feature_set_version)
);

SELECT create_hypertable('ml_models.feature_store', 'timestamp',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE);

-- =============================================================================
-- TRADING SCHEMA (Alpha Model Integration)
-- =============================================================================

-- Enhanced trade hypotheses (from Alpha Model)
CREATE TABLE trading.trade_hypotheses (
    signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Signal details
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('BUY', 'SELL', 'HOLD')),
    entry_price DECIMAL(12, 4) NOT NULL,
    predicted_target DECIMAL(12, 4),
    predicted_stop_loss DECIMAL(12, 4),

    -- Position sizing
    risk_reward_ratio DECIMAL(6, 3),
    position_size_pct DECIMAL(5, 4),
    max_risk_pct DECIMAL(5, 4),

    -- Signal characteristics
    signal_source VARCHAR(100) NOT NULL,
    confluence_score DECIMAL(5, 2),
    ai_confidence DECIMAL(4, 3),
    pattern_type VARCHAR(50),

    -- Market context
    market_regime VARCHAR(20),
    volatility_percentile DECIMAL(3, 2),
    sector_momentum DECIMAL(4, 3),
    vix_level DECIMAL(6, 2),

    -- Timing features
    time_of_day INTEGER,
    day_of_week INTEGER,
    intraday_position DECIMAL(4, 3),

    -- AI provider context
    ai_provider_used VARCHAR(50),
    ai_response_time_ms INTEGER,
    ai_cost_usd DECIMAL(8, 4),

    -- Technical indicators snapshot
    technical_features JSONB,

    -- Status tracking
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'EXECUTED', 'CANCELLED', 'EXPIRED')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enhanced trade outcomes
CREATE TABLE trading.trade_outcomes (
    outcome_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID REFERENCES trading.trade_hypotheses(signal_id),

    -- Exit information
    exit_timestamp TIMESTAMPTZ,
    exit_price DECIMAL(12, 4),
    exit_reason VARCHAR(50) CHECK (exit_reason IN ('TARGET_HIT', 'STOP_LOSS', 'MANUAL_CLOSE', 'TIME_EXIT', 'TRAILING_STOP')),

    -- Performance metrics
    pnl_gross DECIMAL(12, 4),
    pnl_net DECIMAL(12, 4), -- After transaction costs
    pnl_percentage DECIMAL(8, 4),
    holding_period_hours INTEGER,
    holding_period_bars INTEGER,

    -- Advanced performance metrics
    max_favorable_excursion DECIMAL(8, 4), -- MFE
    max_adverse_excursion DECIMAL(8, 4),   -- MAE
    unrealized_pnl_peak DECIMAL(12, 4),
    drawdown_from_peak DECIMAL(8, 4),

    -- Trade classification
    is_profitable BOOLEAN,
    hit_target BOOLEAN DEFAULT FALSE,
    hit_stop BOOLEAN DEFAULT FALSE,
    was_profitable_at_close BOOLEAN,

    -- Execution details
    slippage_bps INTEGER, -- Basis points
    transaction_cost DECIMAL(10, 4),
    market_impact_bps INTEGER,

    -- Post-trade analysis
    alpha_generated DECIMAL(8, 4), -- Alpha vs benchmark
    beta_exposure DECIMAL(6, 3),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Price predictions tracking
CREATE TABLE trading.price_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Current market state
    current_price DECIMAL(12, 4) NOT NULL,
    current_volume BIGINT,

    -- Model predictions
    model_used VARCHAR(100) NOT NULL,
    model_version VARCHAR(20),

    -- Multi-horizon predictions
    prediction_1h DECIMAL(12, 4),
    prediction_4h DECIMAL(12, 4),
    prediction_1d DECIMAL(12, 4),
    prediction_1w DECIMAL(12, 4),
    prediction_1m DECIMAL(12, 4),

    -- Confidence intervals
    confidence_interval_95_lower JSONB, -- For each horizon
    confidence_interval_95_upper JSONB,

    -- Prediction confidence
    overall_confidence DECIMAL(4, 3),
    horizon_confidence JSONB, -- Confidence per horizon

    -- Feature importance for this prediction
    feature_importance JSONB,

    -- Model ensemble details
    ensemble_weights JSONB,
    individual_predictions JSONB, -- From each model in ensemble

    -- Technical context
    technical_indicators JSONB,
    market_regime VARCHAR(20),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Backtesting results (enhanced from existing)
CREATE TABLE trading.backtest_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- Time period
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_days INTEGER,

    -- Configuration
    initial_capital DECIMAL(15, 2) NOT NULL,
    transaction_cost DECIMAL(6, 5),
    slippage DECIMAL(6, 5),
    max_position_size DECIMAL(4, 3),

    -- Performance metrics
    total_return DECIMAL(8, 4),
    annual_return DECIMAL(8, 4),
    sharpe_ratio DECIMAL(6, 3),
    sortino_ratio DECIMAL(6, 3),
    calmar_ratio DECIMAL(6, 3),
    max_drawdown DECIMAL(6, 4),
    volatility DECIMAL(6, 4),

    -- Risk metrics
    var_95 DECIMAL(6, 4), -- Value at Risk
    cvar_95 DECIMAL(6, 4), -- Conditional VaR
    downside_deviation DECIMAL(6, 4),

    -- Trade statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(5, 4),
    avg_win DECIMAL(10, 4),
    avg_loss DECIMAL(10, 4),
    profit_factor DECIMAL(6, 3),

    -- Model performance
    model_accuracy DECIMAL(5, 4),
    model_precision DECIMAL(5, 4),
    model_recall DECIMAL(5, 4),
    model_f1_score DECIMAL(5, 4),

    -- Walk-forward analysis
    walkforward_periods INTEGER,
    avg_in_sample_accuracy DECIMAL(5, 4),
    avg_out_sample_accuracy DECIMAL(5, 4),
    performance_degradation DECIMAL(6, 4),

    -- Benchmark comparison
    benchmark_return DECIMAL(8, 4),
    alpha_generated DECIMAL(8, 4),
    beta DECIMAL(6, 3),
    information_ratio DECIMAL(6, 3),

    -- Configuration and metadata
    strategy_config JSONB,
    model_versions_used JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- =============================================================================
-- ANALYTICS SCHEMA (Performance monitoring and insights)
-- =============================================================================

-- Real-time analytics dashboard data
CREATE TABLE analytics.dashboard_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Metric identification
    metric_category VARCHAR(50) NOT NULL, -- TRADING, MODEL, SYSTEM, RISK
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 6),
    metric_unit VARCHAR(20),

    -- Context
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    model_name VARCHAR(100),

    -- Aggregation level
    aggregation_level VARCHAR(20), -- REAL_TIME, HOURLY, DAILY, WEEKLY
    aggregation_period TIMESTAMPTZ,

    -- Additional data
    metadata JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('analytics.dashboard_metrics', 'timestamp',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE);

-- System performance monitoring
CREATE TABLE analytics.system_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- System metrics
    cpu_usage_percent DECIMAL(5, 2),
    memory_usage_percent DECIMAL(5, 2),
    disk_usage_percent DECIMAL(5, 2),

    -- Database performance
    db_connections_active INTEGER,
    db_connections_max INTEGER,
    query_avg_duration_ms DECIMAL(8, 2),
    slow_query_count INTEGER,

    -- Application metrics
    api_requests_per_second DECIMAL(8, 2),
    api_avg_response_time_ms DECIMAL(8, 2),
    api_error_rate DECIMAL(5, 4),

    -- ML model performance
    prediction_latency_ms DECIMAL(8, 2),
    model_memory_usage_mb DECIMAL(10, 2),
    feature_processing_time_ms DECIMAL(8, 2),

    -- AI provider metrics
    ai_request_count INTEGER,
    ai_avg_response_time_ms DECIMAL(8, 2),
    ai_cost_usd DECIMAL(8, 4),
    ai_error_rate DECIMAL(5, 4),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('analytics.system_performance', 'timestamp',
                        chunk_time_interval => INTERVAL '1 hour',
                        if_not_exists => TRUE);

-- =============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =============================================================================

-- Market data indexes
CREATE INDEX CONCURRENTLY idx_price_data_symbol_timestamp 
    ON market_data.price_data (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_price_data_timeframe 
    ON market_data.price_data (timeframe, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_technical_indicators_symbol 
    ON market_data.technical_indicators (symbol, timestamp DESC);

-- Trading indexes
CREATE INDEX CONCURRENTLY idx_trade_hypotheses_symbol_timestamp 
    ON trading.trade_hypotheses (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_trade_hypotheses_status 
    ON trading.trade_hypotheses (status, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_trade_outcomes_signal_id 
    ON trading.trade_outcomes (signal_id);
CREATE INDEX CONCURRENTLY idx_trade_outcomes_exit_timestamp 
    ON trading.trade_outcomes (exit_timestamp DESC) WHERE exit_timestamp IS NOT NULL;

-- ML model indexes
CREATE INDEX CONCURRENTLY idx_model_registry_status 
    ON ml_models.model_registry (status, is_production);
CREATE INDEX CONCURRENTLY idx_model_performance_model_date 
    ON ml_models.model_performance_history (model_id, evaluation_date DESC);
CREATE INDEX CONCURRENTLY idx_feature_store_symbol_timestamp 
    ON ml_models.feature_store (symbol, timestamp DESC);

-- Analytics indexes
CREATE INDEX CONCURRENTLY idx_dashboard_metrics_category_timestamp 
    ON analytics.dashboard_metrics (metric_category, timestamp DESC);
CREATE INDEX CONCURRENTLY idx_system_performance_timestamp 
    ON analytics.system_performance (timestamp DESC);

-- GIN indexes for JSONB columns (for fast JSON queries)
CREATE INDEX CONCURRENTLY idx_model_registry_hyperparameters 
    ON ml_models.model_registry USING GIN (hyperparameters);
CREATE INDEX CONCURRENTLY idx_feature_store_features 
    ON ml_models.feature_store USING GIN (features);
CREATE INDEX CONCURRENTLY idx_trade_hypotheses_technical_features 
    ON trading.trade_hypotheses USING GIN (technical_features);

-- =============================================================================
-- DATA RETENTION POLICIES
-- =============================================================================

-- Automatic data retention with TimescaleDB
-- Keep detailed data for 1 year, aggregated data for 3 years
SELECT add_retention_policy('market_data.price_data', INTERVAL '1 year');
SELECT add_retention_policy('market_data.technical_indicators', INTERVAL '1 year');
SELECT add_retention_policy('ml_models.feature_store', INTERVAL '6 months');
SELECT add_retention_policy('analytics.dashboard_metrics', INTERVAL '3 months');
SELECT add_retention_policy('analytics.system_performance', INTERVAL '1 month');

-- =============================================================================
-- FUNCTIONS AND TRIGGERS FOR AUTOMATION
-- =============================================================================

-- Function to update model performance when new outcomes are added
CREATE OR REPLACE FUNCTION update_model_performance()
RETURNS TRIGGER AS $$
BEGIN
    -- Update model performance metrics when new trade outcomes are recorded
    -- This would contain logic to recalculate accuracy, precision, etc.
    -- Implementation would be added based on specific requirements
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update performance metrics
CREATE TRIGGER trigger_update_model_performance
    AFTER INSERT ON trading.trade_outcomes
    FOR EACH ROW
    EXECUTE FUNCTION update_model_performance();

-- Function to automatically classify market regimes
CREATE OR REPLACE FUNCTION classify_market_regime(
    symbol_param VARCHAR(20),
    timestamp_param TIMESTAMPTZ
)
RETURNS VARCHAR(20) AS $$
DECLARE
    regime_result VARCHAR(20);
BEGIN
    -- Implementation for automatic regime classification
    -- Based on technical indicators and price action
    SELECT 'SIDEWAYS' INTO regime_result; -- Placeholder
    RETURN regime_result;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Current market snapshot
CREATE VIEW analytics.current_market_snapshot AS
SELECT 
    pd.symbol,
    pd.timestamp,
    pd.close as current_price,
    pd.volume,
    ti.rsi_14,
    ti.macd,
    ti.bb_position,
    mr.regime as market_regime,
    mr.regime_confidence
FROM market_data.price_data pd
LEFT JOIN market_data.technical_indicators ti 
    ON pd.symbol = ti.symbol AND pd.timestamp = ti.timestamp AND pd.timeframe = ti.timeframe
LEFT JOIN market_data.market_regimes mr 
    ON pd.symbol = mr.symbol AND pd.timestamp = mr.timestamp AND pd.timeframe = mr.timeframe
WHERE pd.timestamp >= NOW() - INTERVAL '1 hour'
    AND pd.timeframe = '1H'
ORDER BY pd.symbol, pd.timestamp DESC;

-- Active models performance
CREATE VIEW ml_models.active_models_performance AS
SELECT 
    mr.model_name,
    mr.model_version,
    mr.model_type,
    mr.accuracy,
    mr.roc_auc,
    mr.is_production,
    mph.evaluation_date,
    mph.profit_accuracy,
    mph.concept_drift_detected
FROM ml_models.model_registry mr
LEFT JOIN ml_models.model_performance_history mph 
    ON mr.model_id = mph.model_id
WHERE mr.status = 'ACTIVE'
    AND (mph.evaluation_date IS NULL OR mph.evaluation_date = (
        SELECT MAX(evaluation_date) 
        FROM ml_models.model_performance_history 
        WHERE model_id = mr.model_id
    ));

-- Trading performance summary
CREATE VIEW trading.performance_summary AS
SELECT 
    th.symbol,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN to_outcome.is_profitable THEN 1 END) as profitable_signals,
    AVG(to_outcome.pnl_percentage) as avg_pnl_pct,
    AVG(th.confluence_score) as avg_confluence_score,
    AVG(th.ai_confidence) as avg_ai_confidence
FROM trading.trade_hypotheses th
LEFT JOIN trading.trade_outcomes to_outcome ON th.signal_id = to_outcome.signal_id
WHERE th.timestamp >= CURRENT_DATE - INTERVAL '30 days'
    AND th.status = 'EXECUTED'
GROUP BY th.symbol
ORDER BY profitable_signals DESC;

-- =============================================================================
-- INITIAL DATA SETUP
-- =============================================================================

-- Insert initial system configuration
INSERT INTO analytics.dashboard_metrics (metric_category, metric_name, metric_value, metric_unit)
VALUES 
    ('SYSTEM', 'DATABASE_VERSION', 1.0, 'version'),
    ('SYSTEM', 'SCHEMA_VERSION', 1.0, 'version'),
    ('SYSTEM', 'MIGRATION_TIMESTAMP', EXTRACT(EPOCH FROM NOW()), 'unix_timestamp');

-- Create initial model registry entry for Alpha Model
INSERT INTO ml_models.model_registry (
    model_name, model_type, model_version, description, status
) VALUES (
    'MarketPulse_Alpha_Model', 'ALPHA_MODEL', '1.0.0', 
    'Core predictive engine for trading signal profitability', 'ACTIVE'
);

COMMIT;

-- =============================================================================
-- PERFORMANCE OPTIMIZATION SETTINGS
-- =============================================================================

-- TimescaleDB specific optimizations
ALTER DATABASE