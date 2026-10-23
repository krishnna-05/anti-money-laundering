-- =========================================================================
-- Anti-Money Laundering (AML) OLAP Operations Collection (SAML-D warehouse)
-- Covers: Roll-up, Drill-down, Slice, Dice, and Pivot
-- Amounts are in the active currency configured in config.py (default GBP).
-- =========================================================================

-- 1. ROLL-UP OPERATION
-- Aggregates transactions, volume and suspicious alerts from Month -> Quarter -> Year
SELECT
    d.year_number,
    d.quarter_name,
    d.year_month,
    COUNT(f.transaction_key) AS total_transactions,
    ROUND(SUM(f.amount), 2) AS total_volume,
    SUM(f.is_suspicious) AS suspicious_tx_count,
    ROUND(SUM(CASE WHEN f.is_suspicious = 1 THEN f.amount ELSE 0 END), 2) AS illicit_volume,
    ROUND(AVG(f.amount), 2) AS avg_ticket
FROM Fact_Transactions f
JOIN Dim_Date d ON f.date_key = d.date_key
GROUP BY ROLLUP (d.year_number, d.quarter_name, d.year_month)
ORDER BY d.year_number, d.quarter_name, d.year_month;

-- 2. DRILL-DOWN OPERATION
-- Corridor -> Receiver bank location -> Payment channel, for high-risk (FATF grey list) corridors
SELECT
    l.corridor,
    l.receiver_location,
    c.channel_name,
    COUNT(f.transaction_key) AS tx_count,
    ROUND(SUM(f.amount), 2) AS total_volume,
    SUM(f.is_suspicious) AS suspicious_count,
    ROUND(SUM(f.is_suspicious) * 100.0 / COUNT(f.transaction_key), 2) AS suspicious_rate_pct
FROM Fact_Transactions f
JOIN Dim_Location l ON f.location_key = l.location_key
JOIN Dim_Channel c ON f.channel_key = c.channel_key
WHERE l.is_high_risk_corridor = 1
GROUP BY l.corridor, l.receiver_location, c.channel_name
ORDER BY suspicious_count DESC, total_volume DESC;

-- 3. SLICE OPERATION
-- Fix a single dimension member: Payment channel = 'Cross-border'
SELECT
    a.account_segment,
    d.quarter_name,
    COUNT(f.transaction_key) AS tx_count,
    ROUND(SUM(f.amount), 2) AS total_volume,
    SUM(f.is_suspicious) AS flagged_suspicious
FROM Fact_Transactions f
JOIN Dim_Channel c ON f.channel_key = c.channel_key
JOIN Dim_Account a ON f.sender_account_key = a.account_key
JOIN Dim_Date d ON f.date_key = d.date_key
WHERE c.channel_name = 'Cross-border'
GROUP BY a.account_segment, d.quarter_name
ORDER BY d.quarter_name, flagged_suspicious DESC;

-- 4. DICE OPERATION
-- Restrict several dimensions: 2023 Q1-Q2 + high-risk corridors + amount >= structuring band (8,800)
SELECT
    d.quarter_name,
    a.account_segment,
    l.corridor,
    c.channel_name,
    COUNT(f.transaction_key) AS transaction_count,
    ROUND(SUM(f.amount), 2) AS diced_volume,
    SUM(f.sar_flag) AS total_sar_alerts
FROM Fact_Transactions f
JOIN Dim_Date d ON f.date_key = d.date_key
JOIN Dim_Account a ON f.sender_account_key = a.account_key
JOIN Dim_Location l ON f.location_key = l.location_key
JOIN Dim_Channel c ON f.channel_key = c.channel_key
WHERE d.quarter_name IN ('2023-Q1', '2023-Q2')
  AND f.amount >= 8800
  AND l.is_high_risk_corridor = 1
GROUP BY d.quarter_name, a.account_segment, l.corridor, c.channel_name
ORDER BY total_sar_alerts DESC, diced_volume DESC;

-- 5. PIVOT (CROSS-TABULATION) OPERATION
-- Payment channel (rows) x Account segment (columns): suspicious transaction count
PIVOT (
    SELECT c.channel_name, a.account_segment, f.is_suspicious
    FROM Fact_Transactions f
    JOIN Dim_Channel c ON f.channel_key = c.channel_key
    JOIN Dim_Account a ON f.sender_account_key = a.account_key
) ON account_segment
USING SUM(is_suspicious)
GROUP BY channel_name
ORDER BY channel_name;
