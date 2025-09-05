-- Initialize the stock_ai database with enterprise tables
-- This file is automatically executed when the PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types
CREATE TYPE timeframe_enum AS ENUM ('1m', '5m', '15m', '1h', '4h', '1d', '1w');
CREATE TYPE risk_tolerance_enum AS ENUM ('conservative', 'moderate', 'aggressive');
CREATE TYPE investment_horizon_enum AS ENUM ('short', 'medium', 'long');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_timestamp ON stock_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timestamp ON technical_indicators(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_prediction_history_symbol_date ON prediction_history(symbol, prediction_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_user_symbol ON portfolio(user_id, symbol);

-- Insert default user profile
INSERT INTO user_profiles (username, email, risk_tolerance, investment_horizon, initial_capital)
VALUES ('demo_user', 'demo@stockai.com', 'moderate', 'medium', 50000.00)
ON CONFLICT (username) DO NOTHING;

-- Create a stored procedure for calculating portfolio performance
CREATE OR REPLACE FUNCTION calculate_portfolio_performance(user_id_param INTEGER)
RETURNS TABLE(
    total_value DECIMAL(15,2),
    total_gain_loss DECIMAL(15,2),
    percentage_change DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        SUM(shares * current_price) as total_value,
        SUM((current_price - avg_purchase_price) * shares) as total_gain_loss,
        CASE 
            WHEN SUM(avg_purchase_price * shares) > 0 THEN
                (SUM((current_price - avg_purchase_price) * shares) / SUM(avg_purchase_price * shares) * 100)
            ELSE 0
        END as percentage_change
    FROM portfolio 
    WHERE user_id = user_id_param;
END;
$$ LANGUAGE plpgsql;

-- Create a view for portfolio summary
CREATE OR REPLACE VIEW portfolio_summary AS
SELECT 
    p.user_id,
    u.username,
    COUNT(p.symbol) as total_positions,
    SUM(p.shares * p.current_price) as total_market_value,
    SUM((p.current_price - p.avg_purchase_price) * p.shares) as total_unrealized_gain_loss,
    AVG(CASE 
        WHEN p.avg_purchase_price > 0 THEN 
            ((p.current_price - p.avg_purchase_price) / p.avg_purchase_price * 100)
        ELSE 0 
    END) as avg_percentage_change
FROM portfolio p
JOIN user_profiles u ON p.user_id = u.id
GROUP BY p.user_id, u.username;