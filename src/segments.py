"""Data-driven segment names.

K-Means cluster IDs are arbitrary and can change whenever the model is retrained,
so segment names must not be hard-coded by cluster ID. This module names each
cluster from its average profile (compared with the overall customer average)
and saves the result to data/processed/segment_names.csv, which the API and the
dashboard read.
"""
import os

import pandas as pd

SEGMENT_FILE = "data/processed/segment_names.csv"
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def name_segments(profiles: pd.DataFrame, sizes: pd.Series,
                  purchase_share: pd.Series, overall: pd.Series) -> pd.DataFrame:
    """Assign a business name, description and suggested strategy to each cluster.

    profiles        index = cluster id, columns = mean feature values (original scale)
    sizes           number of customers per cluster
    purchase_share  percent of total purchases per cluster
    overall         mean feature values across all customers
    """
    p = profiles
    remaining = list(p.index)
    chosen = {}

    def ratio(cid, col):
        return p.at[cid, col] / overall[col] if overall[col] else float("nan")

    def take(cid, name, description, strategy):
        chosen[cid] = (name, description, strategy)
        remaining.remove(cid)

    # 1. Highest average purchases
    cid = p.loc[remaining, "PURCHASES"].idxmax()
    take(cid, "Big Spenders",
         f"Average purchases ${p.at[cid, 'PURCHASES']:,.0f} ({ratio(cid, 'PURCHASES'):.1f}x the overall average) "
         f"and the highest spending of any segment.",
         "Protect and grow: premium rewards, credit-limit reviews, and retention offers.")

    # 2. Heavy cash-advance use
    if remaining:
        cid = p.loc[remaining, "CASH_ADVANCE"].idxmax()
        if ratio(cid, "CASH_ADVANCE") >= 2:
            take(cid, "Cash-Advance Heavy",
                 f"Average cash advance ${p.at[cid, 'CASH_ADVANCE']:,.0f} ({ratio(cid, 'CASH_ADVANCE'):.1f}x average) "
                 f"with a high balance of ${p.at[cid, 'BALANCE']:,.0f}.",
                 "Manage risk: monitor utilization and offer installment or balance-transfer plans instead of cash advances.")

    # 3. Pay the balance in full
    if remaining:
        cid = p.loc[remaining, "PRC_FULL_PAYMENT"].idxmax()
        if ratio(cid, "PRC_FULL_PAYMENT") >= 2:
            take(cid, "Full Payers",
                 f"Pay the full balance {p.at[cid, 'PRC_FULL_PAYMENT']:.0%} of the time "
                 f"({ratio(cid, 'PRC_FULL_PAYMENT'):.1f}x average) and keep low balances.",
                 "Deepen spending: cashback and category rewards; they generate little interest income.")

    # 4. Newest accounts
    if remaining:
        cid = p.loc[remaining, "TENURE"].idxmin()
        if p.at[cid, "TENURE"] <= overall["TENURE"] - 2:
            take(cid, "New Customers",
                 f"Average tenure {p.at[cid, 'TENURE']:.1f} months vs {overall['TENURE']:.1f} overall.",
                 "Activate early: onboarding campaigns and first-purchase incentives.")

    # 5. Unusually high minimum payments (small outlier group)
    if remaining:
        cid = p.loc[remaining, "MINIMUM_PAYMENTS"].idxmax()
        if ratio(cid, "MINIMUM_PAYMENTS") >= 5:
            take(cid, "High Minimum Payments",
                 f"Average minimum payment ${p.at[cid, 'MINIMUM_PAYMENTS']:,.0f} "
                 f"({ratio(cid, 'MINIMUM_PAYMENTS'):.0f}x average); a small group worth reviewing individually.",
                 "Review individually: may signal credit stress or data-quality issues.")

    # 6. Everyone else: split by how often they buy
    for cid in list(remaining):
        if p.at[cid, "PURCHASES_FREQUENCY"] >= overall["PURCHASES_FREQUENCY"]:
            take(cid, "Active Revolvers",
                 f"Buy often (purchase frequency {p.at[cid, 'PURCHASES_FREQUENCY']:.2f}) but rarely pay in full "
                 f"({p.at[cid, 'PRC_FULL_PAYMENT']:.0%}).",
                 "Offer installment plans and promotional rates; they are a core interest-revenue segment.")
        else:
            take(cid, "Low Activity",
                 f"Low purchase frequency ({p.at[cid, 'PURCHASES_FREQUENCY']:.2f}) and average purchases of "
                 f"${p.at[cid, 'PURCHASES']:,.0f}.",
                 "Re-engage with targeted spend incentives and watch for dormancy.")

    # Keep names unique if two clusters land in the same rule
    seen = {}
    rows = []
    for i, cid in enumerate(sorted(chosen)):
        name, description, strategy = chosen[cid]
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name} {seen[name]}"
        rows.append({
            "cluster": int(cid),
            "name": name,
            "description": description,
            "strategy": strategy,
            "color": PALETTE[i % len(PALETTE)],
            "n_customers": int(sizes[cid]),
            "pct_customers": round(float(sizes[cid]) / float(sizes.sum()) * 100, 2),
            "pct_purchases": round(float(purchase_share[cid]), 2),
        })
    return pd.DataFrame(rows)


def load_segment_info(path: str = SEGMENT_FILE) -> dict:
    """Return {cluster_id: {name, description, strategy, color, ...}} for the API and dashboard."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {int(r["cluster"]): r.drop(labels="cluster").to_dict() for _, r in df.iterrows()}
