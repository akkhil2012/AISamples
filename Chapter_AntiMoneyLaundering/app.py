# aml_agents_app.py
"""
Streamlit app: Demonstrates simple AI Agents in the context of Anti-Money-Laundering (AML).

Agents included:
 - TransactionAgent: computes rule-based risk scores with explainability (feature contributions)
 - GraphAgent: builds a transaction graph, highlights hubs / beneficiary clusters
 - ComplianceAgent: applies thresholds + business logic to decide "FLAG" vs "APPROVE"
 - SARAgent: generates a SAR draft (templated text) for flagged cases

Run:
    pip install streamlit pandas networkx matplotlib numpy
    streamlit run aml_agents_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime, timedelta

st.set_page_config(page_title="AML Agents Playground", layout="wide")

# -------------------------
# Utilities & Sample Data
# -------------------------
def sample_transactions():
    # Small sample dataset - mix of normal and high-risk patterns
    now = datetime.utcnow()
    rows = [
        {"txn_id": "T001", "account_id": "A100", "beneficiary_id": "B200", "amount": 2500, "currency": "INR", "country": "IN", "method": "POS", "timestamp": now - timedelta(days=1)},
        {"txn_id": "T002", "account_id": "A101", "beneficiary_id": "B201", "amount": 98000, "currency": "INR", "country": "IN", "method": "WIRE", "timestamp": now - timedelta(hours=5)},
        {"txn_id": "T003", "account_id": "A102", "beneficiary_id": "B300", "amount": 1200000, "currency": "USD", "country": "KY", "method": "TRANSFER", "timestamp": now - timedelta(days=2)},
        {"txn_id": "T004", "account_id": "A103", "beneficiary_id": "B202", "amount": 15000, "currency": "USD", "country": "US", "method": "POS", "timestamp": now - timedelta(minutes=90)},
        {"txn_id": "T005", "account_id": "A101", "beneficiary_id": "B300", "amount": 500000, "currency": "USD", "country": "KY", "method": "WIRE", "timestamp": now - timedelta(hours=2)},
        # Add some smaller structured payments to same beneficiary
        {"txn_id": "T006", "account_id": "A104", "beneficiary_id": "B500", "amount": 4900, "currency": "USD", "country": "US", "method": "CARD", "timestamp": now - timedelta(days=6)},
        {"txn_id": "T007", "account_id": "A105", "beneficiary_id": "B500", "amount": 4900, "currency": "USD", "country": "US", "method": "CARD", "timestamp": now - timedelta(days=5)},
        {"txn_id": "T008", "account_id": "A106", "beneficiary_id": "B500", "amount": 4900, "currency": "USD", "country": "US", "method": "CARD", "timestamp": now - timedelta(days=4)},
    ]
    return pd.DataFrame(rows)

# -------------------------
# Agents (simple, explainable)
# -------------------------
class TransactionAgent:
    """
    Rule-based scoring agent with explainability (returns score and contribution breakdown).
    Weights are tunable from UI for demo.
    """
    def __init__(self, weights):
        self.weights = weights

    def analyze(self, txn):
        # Baseline score and contributions
        score = 0.0
        contrib = {}

        # rule: large amount
        amt = txn["amount"]
        if amt >= self.weights["large_amount_threshold"]:
            contrib["large_amount"] = self.weights["large_amount_weight"]
            score += contrib["large_amount"]

        # rule: high-risk country
        if txn["country"] in self.weights["high_risk_countries"]:
            contrib["high_risk_country"] = self.weights["high_risk_country_weight"]
            score += contrib["high_risk_country"]

        # rule: payment method risk
        if txn["method"] in self.weights["risky_methods"]:
            contrib["risky_method"] = self.weights["risky_method_weight"]
            score += contrib["risky_method"]

        # rule: rapid repeat by same account or beneficiary (simple proxy: recent txn count)
        if txn.get("recent_count", 0) >= self.weights["rapid_repeat_threshold"]:
            contrib["rapid_repeat"] = self.weights["rapid_repeat_weight"]
            score += contrib["rapid_repeat"]

        # rule: beneficiary concentration (structuring) - added externally as "beneficiary_sum_contrib"
        if "beneficiary_sum_contrib" in txn and txn["beneficiary_sum_contrib"] > 0:
            contrib["beneficiary_sum"] = txn["beneficiary_sum_contrib"]
            score += contrib["beneficiary_sum"]

        return score, contrib


class GraphAgent:
    """
    Simple graph agent that builds a bi-partite graph (accounts <-> beneficiaries).
    Highlights hubs and returns a figure for visualization.
    """
    def __init__(self, df):
        self.df = df.copy()

    def build_graph(self):
        G = nx.Graph()
        # Add nodes & edges
        for _, r in self.df.iterrows():
            acc = f"acc:{r['account_id']}"
            ben = f"ben:{r['beneficiary_id']}"
            G.add_node(acc, bipartite=0, type="account")
            G.add_node(ben, bipartite=1, type="beneficiary")
            G.add_edge(acc, ben, amount=r["amount"], txn_id=r["txn_id"])
        return G

    def plot_graph(self, G, highlight_beneficiaries=None):
        plt.figure(figsize=(9,6))
        pos = nx.spring_layout(G, seed=42)
        account_nodes = [n for n,d in G.nodes(data=True) if d.get("type") == "account"]
        ben_nodes = [n for n,d in G.nodes(data=True) if d.get("type") == "beneficiary"]

        nx.draw_networkx_nodes(G, pos, nodelist=account_nodes, node_color="#1f77b4", node_size=400, label="accounts")
        nx.draw_networkx_nodes(G, pos, nodelist=ben_nodes, node_color="#ff7f0e", node_size=600, label="beneficiaries")

        # highlight special beneficiaries
        if highlight_beneficiaries:
            highlight_nodes = [f"ben:{b}" for b in highlight_beneficiaries]
            nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color="#d62728", node_size=900)

        nx.draw_networkx_edges(G, pos, alpha=0.6)
        labels = {n: n.split(":",1)[1] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        plt.axis("off")
        plt.tight_layout()
        return plt.gcf()


class ComplianceAgent:
    """Applies decision logic based on aggregated score + business rules"""
    def __init__(self, flag_threshold, escalate_threshold):
        self.flag_threshold = flag_threshold
        self.escalate_threshold = escalate_threshold

    def decide(self, score):
        if score >= self.escalate_threshold:
            return "ESCALATE"   # highest priority
        if score >= self.flag_threshold:
            return "FLAG"
        return "APPROVE"


class SARAgent:
    """Simple templated SAR drafter (explainability-friendly)"""
    def draft(self, txn, score, contributions):
        lines = []
        lines.append(f"Suspicious Activity Report (Draft) - Transaction {txn['txn_id']}")
        lines.append(f"Date/Time: {txn['timestamp']}")
        lines.append(f"Account: {txn['account_id']}  |  Beneficiary: {txn['beneficiary_id']}")
        lines.append(f"Amount: {txn['amount']} {txn.get('currency','')}")
        lines.append("")
        lines.append("Summary of Reasons:")
        for k,v in contributions.items():
            lines.append(f" - {k}: +{v:.2f} risk")
        lines.append(f"\nTotal Risk Score: {score:.2f}")
        lines.append("\nRecommended Action: Conduct enhanced due diligence, review KYC, check settlement partners and recent inbound/outbound flows.")
        return "\n".join(str(x) for x in lines)

# -------------------------
# Streamlit UI
# -------------------------
st.title("AML Agents Playground — GenAI / Agent Concept (Demo)")
st.markdown(
    "This interactive demo showcases a simple agent-based AML pipeline: **TransactionAgent → GraphAgent → ComplianceAgent → SARAgent**. "
    "Tune weights and thresholds, run agents, and inspect explainability outputs."
)

# Sidebar: settings
st.sidebar.header("Agent Settings")
weights = {
    "large_amount_threshold": st.sidebar.number_input("Large amount threshold", value=100000, step=1000),
    "large_amount_weight": st.sidebar.number_input("Large amount weight", value=5.0, step=0.5),
    "high_risk_countries": st.sidebar.text_input("High-risk countries (comma-separated)", value="KY,IR").split(","),
    "high_risk_country_weight": st.sidebar.number_input("High-risk country weight", value=4.0, step=0.5),
    "risky_methods": st.sidebar.text_input("Risky methods (comma-separated)", value="WIRE,TRANSFER").split(","),
    "risky_method_weight": st.sidebar.number_input("Risky method weight", value=2.0, step=0.5),
    "rapid_repeat_threshold": st.sidebar.number_input("Rapid repeat txn threshold (count)", value=3, step=1),
    "rapid_repeat_weight": st.sidebar.number_input("Rapid repeat weight", value=1.5, step=0.5),
}
# normalize lists trimming whitespace
weights["high_risk_countries"] = [c.strip() for c in weights["high_risk_countries"] if c.strip()]
weights["risky_methods"] = [c.strip() for c in weights["risky_methods"] if c.strip()]

flag_threshold = st.sidebar.number_input("Flag threshold (score)", value=4.0, step=0.5)
escalate_threshold = st.sidebar.number_input("Escalate threshold (score)", value=7.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("Upload CSV (optional)")
uploaded = st.sidebar.file_uploader("Transactions CSV", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded, parse_dates=["timestamp"])
    except Exception:
        st.sidebar.error("CSV must include timestamp column in ISO format. Falling back to sample data.")
        df = sample_transactions()
else:
    df = sample_transactions()

# show transactions
st.subheader("Transaction Data")
st.write("You can upload a CSV with columns: txn_id, account_id, beneficiary_id, amount, currency, country, method, timestamp")
st.dataframe(df.style.format({"amount":"{:.2f}"}), height=250)

# Button to run agents
if st.button("Run Agents"):
    # Precompute simple recent counts per account and beneficiary (last 7 days)
    df = df.copy()
    cutoff = datetime.utcnow() - timedelta(days=7)
    recent = df[df["timestamp"] >= cutoff]
    acct_counts = recent.groupby("account_id").size().to_dict()
    ben_counts = recent.groupby("beneficiary_id").size().to_dict()
    ben_sums = recent.groupby("beneficiary_id")["amount"].sum().to_dict()

    # Compute beneficiary_sum_contrib if cumulative > threshold (structuring)
    structuring_threshold = st.sidebar.number_input("Structuring window sum threshold", value=20000, step=1000)
    structuring_weight_per_10000 = st.sidebar.number_input("Structuring weight per 10k", value=0.5, step=0.1)

    txn_agent = TransactionAgent(weights)
    comp_agent = ComplianceAgent(flag_threshold=flag_threshold, escalate_threshold=escalate_threshold)
    sar_agent = SARAgent()

    results = []
    flagged_df_rows = []

    for _, row in df.iterrows():
        r = row.to_dict()
        r["recent_count"] = acct_counts.get(r["account_id"], 0)
        ben_sum = ben_sums.get(r["beneficiary_id"], 0)
        # beneficiary sum contribution (scaled)
        if ben_sum >= structuring_threshold:
            r["beneficiary_sum_contrib"] = (ben_sum / 10000.0) * structuring_weight_per_10000
        else:
            r["beneficiary_sum_contrib"] = 0.0

        score, contrib = txn_agent.analyze(r)
        decision = comp_agent.decide(score)
        r["score"] = score
        r["decision"] = decision
        r["contrib"] = contrib
        results.append(r)

        if decision in ("FLAG", "ESCALATE"):
            flagged_df_rows.append(r)

    results_df = pd.DataFrame(results)

    # Layout: left = table + details, right = graph + SAR drafts
    left, right = st.columns((2,1))

    with left:
        st.subheader("Risk Scored Transactions")
        st.dataframe(results_df[["txn_id","account_id","beneficiary_id","amount","country","method","score","decision"]].sort_values("score", ascending=False), height=300)

        st.markdown("### Click a transaction to inspect explainability")
        txn_choice = st.selectbox("Select txn_id", options=results_df["txn_id"].tolist())
        txn_row = results_df[results_df["txn_id"] == txn_choice].iloc[0]
        st.write("**Transaction details**")
        st.json({
            "txn_id": txn_row["txn_id"],
            "account_id": txn_row["account_id"],
            "beneficiary_id": txn_row["beneficiary_id"],
            "amount": txn_row["amount"],
            "country": txn_row["country"],
            "method": txn_row["method"],
            "timestamp": str(txn_row["timestamp"]),
            "score": float(txn_row["score"]),
            "decision": txn_row["decision"]
        })

        st.write("**Score contributions (explainability)**")
        contrib = txn_row["contrib"]
        if contrib:
            contrib_items = [(k, f"{v:.2f}") for k,v in contrib.items()]
            st.table(pd.DataFrame(contrib_items, columns=["driver","score_contribution"]))
        else:
            st.info("No rule drivers for this transaction (low score).")

    with right:
        st.subheader("Transaction Graph (Accounts ↔ Beneficiaries)")
        graph_agent = GraphAgent(pd.DataFrame(results))
        G = graph_agent.build_graph()

        # identify beneficiaries with high incoming sum
        ben_total = {}
        for _, r in results_df.iterrows():
            ben_total[r["beneficiary_id"]] = ben_total.get(r["beneficiary_id"], 0) + r["amount"]
        # choose highlight beneficiaries above 100k (demo)
        highlight_bens = [b for b,s in ben_total.items() if s >= 100000]
        fig = graph_agent.plot_graph(G, highlight_beneficiaries=highlight_bens)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Flagged cases**")
        if flagged_df_rows:
            flagged_df = pd.DataFrame(flagged_df_rows)
            st.dataframe(flagged_df[["txn_id","account_id","beneficiary_id","amount","score","decision"]].sort_values("score", ascending=False), height=200)

            # SAR drafts for each flagged tx (collapsible)
            for _, fr in flagged_df.iterrows():
                with st.expander(f"SAR Draft - {fr['txn_id']} (decision: {fr['decision']})"):
                    sar_text = sar_agent.draft(fr, fr["score"], fr["contrib"])
                    st.code(sar_text)
        else:
            st.info("No flagged transactions with current settings. Try lowering thresholds or increasing weights.")

    st.success("Agents run complete. Inspect explainability, graph, and SAR drafts.")
else:
    st.info("Press **Run Agents** to execute the TransactionAgent → ComplianceAgent pipeline on the sample dataset.")
    st.markdown("You can upload your own transaction CSV with columns: `txn_id, account_id, beneficiary_id, amount, currency, country, method, timestamp`")

# Footer: short explanation
st.markdown("---")
st.markdown(
    "**Notes & next steps (for a production design):**\n\n"
    "- Replace rule-based TransactionAgent with hybrid models (rules + learned features).  \n"
    "- Use a proper graph DB (Neo4j, TigerGraph) + Graph Neural Networks for deep network detection.  \n"
    "- Add audit logs, model versioning, and human-in-the-loop feedback for governance.  \n"
    "- Integrate a GenAI SAR Agent only after strong red-team evaluation and manual-review gating."
)

