"""
OLAP Query Engine for the Anti-Money Laundering Data Warehouse
Supports Roll-up, Drill-down, Slice, Dice, and Pivot. Amounts are in the active currency (config.py).
"""

import os
import sys
import duckdb
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from config import DUCKDB_PATH, STRUCTURING_LOW, money_unit


def ensure_duckdb(target_path: str = DUCKDB_PATH):
    if os.path.exists(target_path):
        return
    parts_dir = os.path.join(os.path.dirname(target_path), "warehouse_parts")
    if os.path.exists(parts_dir):
        import glob, gzip, shutil
        parts = sorted(glob.glob(os.path.join(parts_dir, "part_*")))
        if parts:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            gz_temp = target_path + ".temp.gz"
            with open(gz_temp, "wb") as out_f:
                for p in parts:
                    with open(p, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
            with gzip.open(gz_temp, "rb") as gz_in, open(target_path, "wb") as db_out:
                shutil.copyfileobj(gz_in, db_out)
            if os.path.exists(gz_temp):
                os.remove(gz_temp)


class AML_OLAP_Engine:
    def __init__(self, db_path: str = DUCKDB_PATH):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            ensure_duckdb(self.db_path)
        self.con = duckdb.connect(self.db_path, read_only=True)

    def execute_query(self, query: str, params: list = None) -> pd.DataFrame:
        return self.con.execute(query, params or []).df()

    def rollup(self, level: str = "month") -> pd.DataFrame:
        """Day -> Month -> Quarter -> Year aggregation."""
        group_cols = {
            "day": ["d.year_number", "d.quarter_name", "d.year_month", "d.full_date"],
            "month": ["d.year_number", "d.quarter_name", "d.year_month"],
            "quarter": ["d.year_number", "d.quarter_name"],
            "year": ["d.year_number"],
        }[level]
        cols = ", ".join(group_cols)
        query = f"""
        SELECT
            {cols},
            COUNT(f.transaction_key) AS total_transactions,
            ROUND(SUM(f.amount), 2) AS total_volume,
            SUM(f.is_suspicious) AS suspicious_tx_count,
            ROUND(SUM(CASE WHEN f.is_suspicious = 1 THEN f.amount ELSE 0 END), 2) AS illicit_volume,
            ROUND(AVG(f.amount), 2) AS avg_ticket
        FROM Fact_Transactions f
        JOIN Dim_Date d ON f.date_key = d.date_key
        GROUP BY {cols}
        ORDER BY {cols};
        """
        return self.execute_query(query)

    def drilldown(self, high_risk_only: bool = True) -> pd.DataFrame:
        """Corridor -> Receiver location -> Payment channel."""
        filter_clause = "WHERE l.is_high_risk_corridor = 1" if high_risk_only else "WHERE l.is_cross_border = 1"
        query = f"""
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
        {filter_clause}
        GROUP BY l.corridor, l.receiver_location, c.channel_name
        ORDER BY suspicious_count DESC, total_volume DESC
        LIMIT 25;
        """
        return self.execute_query(query)

    def slice(self, channel: str = "Cross-border") -> pd.DataFrame:
        """Fix one dimension (payment channel) and view the remaining cube."""
        query = """
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
        WHERE c.channel_name = ?
        GROUP BY a.account_segment, d.quarter_name
        ORDER BY d.quarter_name, flagged_suspicious DESC;
        """
        return self.execute_query(query, [channel])

    def dice(self, min_amount: float = STRUCTURING_LOW, quarters: tuple = None, high_risk_only: bool = False) -> pd.DataFrame:
        """Restrict several dimensions at once: amount range x quarters x corridor risk."""
        where = ["f.amount >= ?"]
        params = [min_amount]
        if quarters:
            where.append(f"d.quarter_name IN ({', '.join('?' for _ in quarters)})")
            params.extend(quarters)
        if high_risk_only:
            where.append("l.is_high_risk_corridor = 1")
        query = f"""
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
        WHERE {' AND '.join(where)}
        GROUP BY d.quarter_name, a.account_segment, l.corridor, c.channel_name
        ORDER BY total_sar_alerts DESC, diced_volume DESC
        LIMIT 25;
        """
        return self.execute_query(query, params)

    def pivot(self, measure: str = "suspicious") -> pd.DataFrame:
        """Channel (rows) x account segment (columns) cross-tab."""
        agg = {"suspicious": "SUM(f.is_suspicious)", "count": "COUNT(*)", "volume": "ROUND(SUM(f.amount), 2)"}[measure]
        query = f"""
        PIVOT (
            SELECT c.channel_name, a.account_segment, f.is_suspicious, f.amount
            FROM Fact_Transactions f
            JOIN Dim_Channel c ON f.channel_key = c.channel_key
            JOIN Dim_Account a ON f.sender_account_key = a.account_key
        ) f
        ON account_segment
        USING {agg.replace('f.', '')}
        GROUP BY channel_name
        ORDER BY channel_name;
        """
        return self.execute_query(query).fillna(0)

    def quarters(self) -> list:
        return self.execute_query("SELECT DISTINCT quarter_name FROM Dim_Date ORDER BY 1")["quarter_name"].tolist()

    def channels(self) -> list:
        return self.execute_query("SELECT channel_name FROM Dim_Channel ORDER BY 1")["channel_name"].tolist()


if __name__ == "__main__":
    engine = AML_OLAP_Engine()
    div, unit = money_unit()
    pd.set_option("display.width", 160)
    print("\n--- ROLL-UP (month) ---")
    print(engine.rollup().to_string(index=False))
    print("\n--- DRILL-DOWN (high-risk corridors) ---")
    print(engine.drilldown().head(10).to_string(index=False))
    print("\n--- SLICE (channel = Cross-border) ---")
    print(engine.slice("Cross-border").to_string(index=False))
    print("\n--- DICE (amount >= structuring band, high-risk corridors) ---")
    print(engine.dice(high_risk_only=True).head(10).to_string(index=False))
    print("\n--- PIVOT (suspicious count: channel x account segment) ---")
    print(engine.pivot().to_string(index=False))
