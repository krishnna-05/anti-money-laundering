"""
Anti-Money Laundering (AML) Synthetic Dataset Generator - INR (₹) & FIU-IND Compliance Edition
Simulates Indian banking ecosystem under PMLA (Prevention of Money Laundering Act, 2002):
- Reporting threshold: ₹10,00,000 (10 Lakhs INR)
- Sub-threshold Structuring / Smurfing (₹8.8 Lakhs - ₹9.95 Lakhs)
- Hawala & Layering via Forex Wires & Mule Accounts
- Indian payment channels: UPI, IMPS, NEFT, RTGS, Cash CDM, ATM, Forex Wire, Crypto Gateway
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_aml_dataset_inr(
    num_accounts: int = 1200,
    num_transactions: int = 25000,
    illicit_ratio: float = 0.08,
    random_seed: int = 42,
    output_path: str = "data/raw/aml_transactions.csv"
) -> pd.DataFrame:
    np.random.seed(random_seed)
    random.seed(random_seed)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating {num_accounts} Indian banking accounts...")
    account_ids = [f"SBIN{10000 + i}" if i % 3 == 0 else (f"HDFC{10000 + i}" if i % 3 == 1 else f"ICIC{10000 + i}") for i in range(num_accounts)]
    customer_types = ["Individual", "Corporate (Pvt Ltd)", "MSME / Proprietorship", "Trust / NGO", "High-Net-Worth (HNI)"]
    cust_weights = [0.60, 0.18, 0.14, 0.03, 0.05]
    
    cities = ["Mumbai", "Delhi NCR", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad", "Pune", "Dubai (UAE)", "Singapore", "Mauritius", "London (UK)"]
    city_weights = [0.22, 0.18, 0.14, 0.10, 0.08, 0.06, 0.06, 0.06, 0.04, 0.02, 0.02, 0.02]
    high_risk_jurisdictions = {"Dubai (UAE)", "Mauritius", "Cayman Islands", "Panama", "Cyprus"}

    accounts_df = pd.DataFrame({
        "account_id": account_ids,
        "customer_name": [f"Account_Holder_{acc}" for acc in account_ids],
        "customer_type": np.random.choice(customer_types, size=num_accounts, p=cust_weights),
        "primary_city": np.random.choice(cities, size=num_accounts, p=city_weights),
        "account_age_months": np.random.randint(1, 120, size=num_accounts),
        "initial_risk_score": np.random.randint(10, 45, size=num_accounts),
        "is_pep": np.random.choice([0, 1], size=num_accounts, p=[0.97, 0.03]), # Politically Exposed Person
        "pan_verified": np.random.choice([1, 0], size=num_accounts, p=[0.94, 0.06])
    })
    
    # Risk adjustments
    accounts_df.loc[accounts_df["primary_city"].isin(high_risk_jurisdictions), "initial_risk_score"] += 35
    accounts_df.loc[accounts_df["is_pep"] == 1, "initial_risk_score"] += 30
    accounts_df.loc[accounts_df["pan_verified"] == 0, "initial_risk_score"] += 20
    accounts_df["initial_risk_score"] = accounts_df["initial_risk_score"].clip(0, 99)

    accounts_map = accounts_df.set_index("account_id").to_dict("index")

    print(f"Generating {num_transactions} transactions in INR (₹) across legitimate and AML typologies...")
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_seconds = int((end_date - start_date).total_seconds())

    channels = ["UPI / IMPS", "NEFT / RTGS", "Cash Deposit (CDM/Branch)", "Cash Withdrawal", "ATM", "Net Banking", "Forex Wire", "Crypto Gateway"]
    channel_weights = [0.35, 0.25, 0.12, 0.08, 0.07, 0.08, 0.03, 0.02]

    transactions = []
    
    num_illicit = int(num_transactions * illicit_ratio)
    num_normal = num_transactions - num_illicit

    # 1. Normal Transactions (INR Scale: ₹500 to ₹15,00,000)
    for i in range(num_normal):
        tx_id = f"TXN_IN_{200000 + i}"
        sec_offset = random.randint(0, date_range_seconds)
        tx_time = start_date + timedelta(seconds=sec_offset)
        
        sender = random.choice(account_ids)
        receiver = random.choice(account_ids)
        while receiver == sender:
            receiver = random.choice(account_ids)
            
        channel = np.random.choice(channels, p=channel_weights)
        
        # Log-normal distribution for typical Indian banking transactions
        if channel == "UPI / IMPS":
            amount = round(float(np.random.lognormal(mean=7.5, sigma=1.0)), 2) # Typical UPI: ₹500 - ₹50,000
            amount = max(100.0, min(amount, 100000.0))
        elif channel in ["NEFT / RTGS", "Net Banking"]:
            amount = round(float(np.random.lognormal(mean=10.5, sigma=1.2)), 2) # Commercial/Salary: ₹20,000 - ₹15,00,000
            amount = max(5000.0, min(amount, 2500000.0))
        elif channel in ["Cash Deposit (CDM/Branch)", "Cash Withdrawal", "ATM"]:
            amount = round(float(np.random.lognormal(mean=8.5, sigma=0.9)), 2) # Cash: ₹1,000 - ₹2,00,000
            amount = max(500.0, min(amount, 300000.0))
        else:
            amount = round(float(np.random.lognormal(mean=11.0, sigma=1.1)), 2)
            amount = max(10000.0, min(amount, 5000000.0))
            
        sender_city = accounts_map[sender]["primary_city"]
        receiver_city = accounts_map[receiver]["primary_city"]
        is_cross_border = 1 if (sender_city in high_risk_jurisdictions or receiver_city in high_risk_jurisdictions or "Dubai" in receiver_city or "London" in receiver_city or "Singapore" in receiver_city or "Mauritius" in receiver_city) else 0
        
        transactions.append({
            "transaction_id": tx_id,
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account": sender,
            "receiver_account": receiver,
            "amount_inr": amount,
            "currency": "INR",
            "payment_channel": channel,
            "sender_city": sender_city,
            "receiver_city": receiver_city,
            "is_cross_border": is_cross_border,
            "is_suspicious": 0,
            "laundering_typology": "Normal Legitimate",
            "sar_flag": 0
        })

    # 2. Illicit AML Typologies in Indian Context
    typologies = [
        "PMLA ₹10L Structuring / Smurfing",
        "Hawala & Rapid Layering",
        "Mule Account Fan-In Inflow",
        "Fan-Out Fund Dispersion",
        "High-Risk Offshore Forex Wire"
    ]
    
    curr_tx_idx = num_normal
    while len(transactions) < num_transactions:
        typology = random.choice(typologies)
        base_sec = random.randint(0, date_range_seconds - 86400 * 7)
        base_time = start_date + timedelta(seconds=base_sec)
        
        if typology == "PMLA ₹10L Structuring / Smurfing":
            # Multiple cash deposits just below the ₹10,00,000 PMLA reporting threshold (e.g. ₹8,80,000 - ₹9,95,000)
            bad_actor = random.choice(account_ids)
            sub_count = random.randint(3, 5)
            for step in range(sub_count):
                if len(transactions) >= num_transactions:
                    break
                tx_time = base_time + timedelta(hours=step * random.randint(1, 6))
                tx_id = f"TXN_IN_{200000 + curr_tx_idx}"
                curr_tx_idx += 1
                amt = round(random.uniform(880000.0, 995000.0), 2)
                transactions.append({
                    "transaction_id": tx_id,
                    "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_account": bad_actor,
                    "receiver_account": random.choice(account_ids),
                    "amount_inr": amt,
                    "currency": "INR",
                    "payment_channel": random.choice(["Cash Deposit (CDM/Branch)", "ATM"]),
                    "sender_city": accounts_map[bad_actor]["primary_city"],
                    "receiver_city": accounts_map[bad_actor]["primary_city"],
                    "is_cross_border": 0,
                    "is_suspicious": 1,
                    "laundering_typology": "PMLA ₹10L Structuring / Smurfing",
                    "sar_flag": 1
                })
                
        elif typology == "Hawala & Rapid Layering":
            # Inflow of ₹40 Lakhs - ₹1.5 Crores followed immediately by Forex Wire / Crypto Gateway
            bridge_account = random.choice(account_ids)
            in_sender = random.choice(account_ids)
            out_receiver = random.choice(account_ids)
            amt = round(random.uniform(4000000.0, 15000000.0), 2)
            
            # Leg 1: Inflow
            tx1_id = f"TXN_IN_{200000 + curr_tx_idx}"
            curr_tx_idx += 1
            transactions.append({
                "transaction_id": tx1_id,
                "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_account": in_sender,
                "receiver_account": bridge_account,
                "amount_inr": amt,
                "currency": "INR",
                "payment_channel": "NEFT / RTGS",
                "sender_city": accounts_map[in_sender]["primary_city"],
                "receiver_city": accounts_map[bridge_account]["primary_city"],
                "is_cross_border": 0,
                "is_suspicious": 1,
                "laundering_typology": "Hawala & Rapid Layering",
                "sar_flag": 1
            })
            
            # Leg 2: Immediate Outflow via Forex / Crypto
            if len(transactions) < num_transactions:
                tx2_time = base_time + timedelta(minutes=random.randint(15, 90))
                tx2_id = f"TXN_IN_{200000 + curr_tx_idx}"
                curr_tx_idx += 1
                transactions.append({
                    "transaction_id": tx2_id,
                    "timestamp": tx2_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_account": bridge_account,
                    "receiver_account": out_receiver,
                    "amount_inr": round(amt * random.uniform(0.96, 0.99), 2),
                    "currency": "INR",
                    "payment_channel": random.choice(["Forex Wire", "Crypto Gateway"]),
                    "sender_city": accounts_map[bridge_account]["primary_city"],
                    "receiver_city": random.choice(list(high_risk_jurisdictions)),
                    "is_cross_border": 1,
                    "is_suspicious": 1,
                    "laundering_typology": "Hawala & Rapid Layering",
                    "sar_flag": 1
                })

        elif typology == "Mule Account Fan-In Inflow":
            mule_hub = random.choice(account_ids)
            senders = random.sample(account_ids, k=min(4, len(account_ids)))
            for s in senders:
                if len(transactions) >= num_transactions:
                    break
                tx_id = f"TXN_IN_{200000 + curr_tx_idx}"
                curr_tx_idx += 1
                tx_time = base_time + timedelta(hours=random.randint(1, 12))
                transactions.append({
                    "transaction_id": tx_id,
                    "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_account": s,
                    "receiver_account": mule_hub,
                    "amount_inr": round(random.uniform(920000.0, 990000.0), 2),
                    "currency": "INR",
                    "payment_channel": random.choice(["UPI / IMPS", "NEFT / RTGS"]),
                    "sender_city": accounts_map[s]["primary_city"],
                    "receiver_city": accounts_map[mule_hub]["primary_city"],
                    "is_cross_border": 0,
                    "is_suspicious": 1,
                    "laundering_typology": "Mule Account Fan-In Inflow",
                    "sar_flag": 1
                })

        elif typology == "Fan-Out Fund Dispersion":
            source_acc = random.choice(account_ids)
            receivers = random.sample(account_ids, k=min(4, len(account_ids)))
            for r in receivers:
                if len(transactions) >= num_transactions:
                    break
                tx_id = f"TXN_IN_{200000 + curr_tx_idx}"
                curr_tx_idx += 1
                tx_time = base_time + timedelta(hours=random.randint(1, 12))
                transactions.append({
                    "transaction_id": tx_id,
                    "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_account": source_acc,
                    "receiver_account": r,
                    "amount_inr": round(random.uniform(890000.0, 985000.0), 2),
                    "currency": "INR",
                    "payment_channel": random.choice(["UPI / IMPS", "Forex Wire"]),
                    "sender_city": accounts_map[source_acc]["primary_city"],
                    "receiver_city": accounts_map[r]["primary_city"],
                    "is_cross_border": 1 if accounts_map[r]["primary_city"] in high_risk_jurisdictions else 0,
                    "is_suspicious": 1,
                    "laundering_typology": "Fan-Out Fund Dispersion",
                    "sar_flag": 1
                })

        else: # High-Risk Offshore Forex Wire
            sender = random.choice(account_ids)
            receiver = random.choice(account_ids)
            tx_id = f"TXN_IN_{200000 + curr_tx_idx}"
            curr_tx_idx += 1
            transactions.append({
                "transaction_id": tx_id,
                "timestamp": base_time.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_account": sender,
                "receiver_account": receiver,
                "amount_inr": round(random.uniform(5000000.0, 50000000.0), 2), # ₹50L to ₹5 Cr
                "currency": "INR",
                "payment_channel": random.choice(["Forex Wire", "Crypto Gateway"]),
                "sender_city": accounts_map[sender]["primary_city"],
                "receiver_city": random.choice(list(high_risk_jurisdictions)),
                "is_cross_border": 1,
                "is_suspicious": 1,
                "laundering_typology": "High-Risk Offshore Forex Wire",
                "sar_flag": 1
            })

    df = pd.DataFrame(transactions)
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Save raw files
    df.to_csv(output_path, index=False)
    accounts_df.to_csv("data/raw/accounts_metadata.csv", index=False)
    
    print(f"INR AML Dataset generated! Total Txns: {len(df)}, Suspicious: {df['is_suspicious'].sum()} ({df['is_suspicious'].mean()*100:.2f}%)")
    return df

if __name__ == "__main__":
    generate_aml_dataset_inr()
