-- =========================================================================
-- Anti-Money Laundering (AML) Data Warehouse Star Schema (SAML-D source)
-- Amounts are stored in the active currency defined in config.py
-- =========================================================================

CREATE TABLE IF NOT EXISTS Dim_Date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL,
    day_of_week INTEGER,
    day_name VARCHAR(15),
    day_of_month INTEGER,
    month_number INTEGER,
    month_name VARCHAR(15),
    year_month VARCHAR(7),
    quarter_number INTEGER,
    quarter_name VARCHAR(10),
    year_number INTEGER,
    is_weekend INTEGER
);

-- Account behaviour profile, aggregated over the full 9.5M-row dataset
CREATE TABLE IF NOT EXISTS Dim_Account (
    account_key INTEGER PRIMARY KEY,
    account_id VARCHAR(30) UNIQUE NOT NULL,
    primary_location VARCHAR(30),
    account_segment VARCHAR(30),
    is_high_risk_location INTEGER,
    out_tx_count INTEGER,
    out_total DOUBLE PRECISION,
    out_avg DOUBLE PRECISION,
    out_std DOUBLE PRECISION,
    out_max DOUBLE PRECISION,
    distinct_receivers INTEGER,      -- fan-out degree
    cash_ratio DOUBLE PRECISION,
    cross_border_ratio DOUBLE PRECISION,
    high_risk_ratio DOUBLE PRECISION,
    fx_ratio DOUBLE PRECISION,
    structuring_ratio DOUBLE PRECISION,
    active_days INTEGER,
    tenure_days INTEGER,
    in_tx_count INTEGER,
    in_total DOUBLE PRECISION,
    distinct_senders INTEGER,        -- fan-in degree
    ever_suspicious INTEGER          -- label; used only for evaluation, never as a model feature
);

CREATE TABLE IF NOT EXISTS Dim_Channel (
    channel_key INTEGER PRIMARY KEY,
    channel_name VARCHAR(30) UNIQUE NOT NULL,
    channel_category VARCHAR(30),
    is_electronic INTEGER,
    is_cash INTEGER
);

CREATE TABLE IF NOT EXISTS Dim_Location (
    location_key INTEGER PRIMARY KEY,
    sender_location VARCHAR(30),
    receiver_location VARCHAR(30),
    corridor VARCHAR(70),
    is_cross_border INTEGER,
    is_high_risk_corridor INTEGER
);

CREATE TABLE IF NOT EXISTS Dim_Currency (
    currency_key INTEGER PRIMARY KEY,
    payment_currency VARCHAR(30),
    received_currency VARCHAR(30),
    currency_pair VARCHAR(70),
    is_fx_conversion INTEGER
);

CREATE TABLE IF NOT EXISTS Fact_Transactions (
    transaction_key INTEGER PRIMARY KEY,
    transaction_id VARCHAR(20) UNIQUE NOT NULL,
    date_key INTEGER REFERENCES Dim_Date(date_key),
    sender_account_key INTEGER REFERENCES Dim_Account(account_key),
    receiver_account_key INTEGER REFERENCES Dim_Account(account_key),
    channel_key INTEGER REFERENCES Dim_Channel(channel_key),
    location_key INTEGER REFERENCES Dim_Location(location_key),
    currency_key INTEGER REFERENCES Dim_Currency(currency_key),
    timestamp TIMESTAMP NOT NULL,
    hour_of_day INTEGER,
    amount DOUBLE PRECISION NOT NULL,
    amount_log DOUBLE PRECISION,
    is_structuring INTEGER,          -- amount within 88%-100% of the reference threshold
    is_suspicious INTEGER NOT NULL,
    laundering_typology VARCHAR(40),
    sar_flag INTEGER NOT NULL,
    -- Point-in-time behaviour: sender/receiver history strictly before this transaction
    s_prior_out_count INTEGER,
    s_prior_in_count INTEGER,
    s_prior_out_avg DOUBLE PRECISION,
    s_prior_distinct_receivers INTEGER,
    s_prior_cash_ratio DOUBLE PRECISION,
    s_prior_cross_border_ratio DOUBLE PRECISION,
    s_prior_high_risk_ratio DOUBLE PRECISION,
    s_secs_since_last_out BIGINT,    -- -1 when the sender has no earlier outgoing transaction
    r_prior_in_count INTEGER,
    r_prior_out_count INTEGER,
    r_prior_distinct_senders INTEGER
);

CREATE INDEX IF NOT EXISTS idx_fact_date ON Fact_Transactions(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_sender ON Fact_Transactions(sender_account_key);
CREATE INDEX IF NOT EXISTS idx_fact_channel ON Fact_Transactions(channel_key);
CREATE INDEX IF NOT EXISTS idx_fact_location ON Fact_Transactions(location_key);
CREATE INDEX IF NOT EXISTS idx_fact_currency ON Fact_Transactions(currency_key);
CREATE INDEX IF NOT EXISTS idx_fact_suspicious ON Fact_Transactions(is_suspicious);
