"""
Synthetic General Ledger (GL) Transaction Generator
-----------------------------------------------------
Generates realistic-but-fake accounting transactions and deliberately
injects known audit "red flag" patterns so the downstream SQL/Python
detection pipeline has real signal to find.

Anomaly types injected (each tagged in a hidden 'true_anomaly_flag'
column so we can later measure precision/recall of our detection logic):
  1. Round-number transactions       (e.g., exactly 5000.00, 10000.00)
  2. Just-under-threshold amounts    (e.g., 4,950 when approval limit is 5,000)
  3. Weekend / after-hours postings  (transactions posted Sat/Sun)
  4. Duplicate invoices               (same vendor, same amount, same/near date)
  5. Benford's-Law-breaking amounts  (artificially forced leading digits)

Output: data/gl_transactions.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

RNG = np.random.default_rng(42)
N_NORMAL = 14000          # normal, "clean" transactions
N_ROUND = 250              # round-number anomalies
N_THRESHOLD = 180           # just-under-threshold anomalies
N_WEEKEND = 220             # weekend posting anomalies
N_DUPLICATE_PAIRS = 150     # duplicate invoice pairs (300 rows)
N_BENFORD_BREAK = 300        # amounts with forced non-natural leading digits

START_DATE = datetime(2024, 4, 1)   # Indian financial year starts April
END_DATE = datetime(2025, 3, 31)    # ... and ends March 31
APPROVAL_THRESHOLD = 500000.00      # Rs 5,00,000 approval limit

ACCOUNTS = [
    ("6100", "Travel & Conveyance"),
    ("6200", "Office Supplies & Stationery"),
    ("6300", "Professional & Legal Fees"),
    ("6400", "IT & Software Licenses"),
    ("6500", "Marketing & Advertising"),
    ("6600", "Electricity & Utilities"),
    ("6700", "Repairs & Maintenance"),
    ("6800", "Consulting Services"),
    ("7100", "Equipment Purchases (Capex)"),
    ("7200", "Rent & Facilities"),
]

VENDOR_POOL = [
    "Shree Balaji Logistics Pvt Ltd", "Nexgen Consulting LLP", "Crescent Office Supplies",
    "Om Facilities Management", "Evergreen IT Solutions Pvt Ltd", "Falcon Stationery Mart",
    "Sundaram Professional Services", "Horizon Marketing Solutions", "Ironclad Security Systems",
    "Bharat Freight Carriers", "Keystone Repairs & Services", "Lumen Software Systems India",
    "Meridian Consulting Partners", "Northstar Power & Utilities", "Orion Equipment Rentals",
    "Pinecrest Travel & Tours", "Quantum Data Services Pvt Ltd", "Radiance Advertising Group",
    "Silverline Facility Services", "Titan Hardware Suppliers", "Union Print & Design Studio",
    "Vertex Engineering Co Pvt Ltd", "Westfield Catering Services", "Yellowstone Cleaning Services",
    "Zenith Professional Consultants",
]

EMPLOYEES = [f"EMP{100+i:04d}" for i in range(1, 61)]  # 60 approving employees


def random_business_date(start, end, rng, weekend_bias=0.0):
    """Return a random date; weekend_bias in [0,1] increases chance of Sat/Sun."""
    span = (end - start).days
    for _ in range(20):
        d = start + timedelta(days=int(rng.integers(0, span)))
        is_weekend = d.weekday() >= 5
        if is_weekend and rng.random() < (1 - weekend_bias):
            continue
        return d
    return d


def gen_normal_amount(rng):
    # log-normal gives a realistic right-skewed spend distribution
    # tuned so median transaction ~ Rs 13,000-15,000, tail runs up to ~Rs 49 lakh
    amt = rng.lognormal(mean=9.5, sigma=1.1)
    return round(min(amt, 4900000), 2)


def build_rows():
    rows = []
    txn_id = 100000

    # ---------- 1. Normal transactions ----------
    for _ in range(N_NORMAL):
        txn_id += 1
        acct_code, acct_name = ACCOUNTS[RNG.integers(0, len(ACCOUNTS))]
        vendor = VENDOR_POOL[RNG.integers(0, len(VENDOR_POOL))]
        emp = EMPLOYEES[RNG.integers(0, len(EMPLOYEES))]
        date = random_business_date(START_DATE, END_DATE, RNG, weekend_bias=0.02)
        amount = gen_normal_amount(RNG)
        rows.append([txn_id, date.strftime("%Y-%m-%d"), acct_code, acct_name,
                     vendor, emp, amount, "invoice", "none"])

    # ---------- 2. Round-number anomalies ----------
    round_values = [100000, 200000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 2500000, 3000000]
    for _ in range(N_ROUND):
        txn_id += 1
        acct_code, acct_name = ACCOUNTS[RNG.integers(0, len(ACCOUNTS))]
        vendor = VENDOR_POOL[RNG.integers(0, len(VENDOR_POOL))]
        emp = EMPLOYEES[RNG.integers(0, len(EMPLOYEES))]
        date = random_business_date(START_DATE, END_DATE, RNG, weekend_bias=0.02)
        amount = float(round_values[RNG.integers(0, len(round_values))])
        rows.append([txn_id, date.strftime("%Y-%m-%d"), acct_code, acct_name,
                     vendor, emp, amount, "invoice", "round_number"])

    # ---------- 3. Just-under-approval-threshold anomalies ----------
    for _ in range(N_THRESHOLD):
        txn_id += 1
        acct_code, acct_name = ACCOUNTS[RNG.integers(0, len(ACCOUNTS))]
        vendor = VENDOR_POOL[RNG.integers(0, len(VENDOR_POOL))]
        emp = EMPLOYEES[RNG.integers(0, len(EMPLOYEES))]
        date = random_business_date(START_DATE, END_DATE, RNG, weekend_bias=0.02)
        # amount deliberately sits 1-3% below the approval threshold
        amount = round(APPROVAL_THRESHOLD * (1 - RNG.uniform(0.005, 0.03)), 2)
        rows.append([txn_id, date.strftime("%Y-%m-%d"), acct_code, acct_name,
                     vendor, emp, amount, "invoice", "just_under_threshold"])

    # ---------- 4. Weekend posting anomalies ----------
    for _ in range(N_WEEKEND):
        txn_id += 1
        acct_code, acct_name = ACCOUNTS[RNG.integers(0, len(ACCOUNTS))]
        vendor = VENDOR_POOL[RNG.integers(0, len(VENDOR_POOL))]
        emp = EMPLOYEES[RNG.integers(0, len(EMPLOYEES))]
        date = random_business_date(START_DATE, END_DATE, RNG, weekend_bias=1.0)
        amount = gen_normal_amount(RNG)
        rows.append([txn_id, date.strftime("%Y-%m-%d"), acct_code, acct_name,
                     vendor, emp, amount, "invoice", "weekend_posting"])

    # ---------- 5. Duplicate invoice pairs ----------
    for _ in range(N_DUPLICATE_PAIRS):
        acct_code, acct_name = ACCOUNTS[RNG.integers(0, len(ACCOUNTS))]
        vendor = VENDOR_POOL[RNG.integers(0, len(VENDOR_POOL))]
        emp = EMPLOYEES[RNG.integers(0, len(EMPLOYEES))]
        base_date = random_business_date(START_DATE, END_DATE, RNG, weekend_bias=0.02)
        amount = gen_normal_amount(RNG)
        for offset in (0, RNG.integers(0, 5)):  # same day or within a few days
            txn_id += 1
            d = base_date + timedelta(days=int(offset))
            rows.append([txn_id, d.strftime("%Y-%m-%d"), acct_code, acct_name,
                         vendor, emp, amount, "invoice", "duplicate_invoice"])

    # ---------- 6. Benford's-Law-breaking amounts ----------
    # Force leading digits toward 7/8/9 (rare in natural data) to simulate
    # fabricated numbers, which is exactly what the Benford test should catch.
    for _ in range(N_BENFORD_BREAK):
        txn_id += 1
        acct_code, acct_name = ACCOUNTS[RNG.integers(0, len(ACCOUNTS))]
        vendor = VENDOR_POOL[RNG.integers(0, len(VENDOR_POOL))]
        emp = EMPLOYEES[RNG.integers(0, len(EMPLOYEES))]
        date = random_business_date(START_DATE, END_DATE, RNG, weekend_bias=0.02)
        leading_digit = RNG.choice([7, 8, 9])
        rest = RNG.integers(0, 999)
        amount = (float(f"{leading_digit}{rest:03d}") + round(RNG.uniform(0, 0.99), 2)) * 10
        rows.append([txn_id, date.strftime("%Y-%m-%d"), acct_code, acct_name,
                     vendor, emp, round(amount, 2), "invoice", "benford_break"])

    df = pd.DataFrame(rows, columns=[
        "txn_id", "txn_date", "account_code", "account_name",
        "vendor_name", "approved_by", "amount", "txn_type", "true_anomaly_flag"
    ])

    # Shuffle so anomalies aren't clustered at the end (more realistic + fair for SQL testing)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def format_inr(n):
    """Format a number in Indian digit grouping with lakh/crore commas, e.g. 1,96,48,586.43"""
    s = f"{n:,.2f}"
    # Python's comma grouping is Western (thousands); redo as Indian grouping
    int_part, dec_part = f"{n:.2f}".split(".")
    neg = int_part.startswith("-")
    if neg:
        int_part = int_part[1:]
    if len(int_part) > 3:
        last3 = int_part[-3:]
        rest = int_part[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        int_part = ",".join(parts) + "," + last3
    return ("-" if neg else "") + int_part + "." + dec_part


if __name__ == "__main__":
    df = build_rows()
    out_path = Path(__file__).resolve().parents[1] / "data" / "gl_transactions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} transactions -> {out_path}")
    print("\nAnomaly breakdown:")
    print(df["true_anomaly_flag"].value_counts())
    print(f"\nDate range (Indian FY 2024-25): {df['txn_date'].min()} to {df['txn_date'].max()}")
    print(f"Total transaction value: Rs {format_inr(df['amount'].sum())}")
    print(f"Approval threshold: Rs {format_inr(APPROVAL_THRESHOLD)}")
