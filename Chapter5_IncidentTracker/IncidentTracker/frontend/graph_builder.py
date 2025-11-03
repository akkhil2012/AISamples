"""
GraphBuilder utility – converts backend session data to a JSON graph that
Streamlit can render with vis.js (or any front-end library).

Nodes carry these keys ::
    id          – unique id string
    label       – display text
    type        – service|process|result|meta
    status      – pending|running|done|error
    metadata    – arbitrary details (dict)

Edges carry ::
    source, target, type, weight
"""
from __future__ import annotations

import hashlib
from typing import Dict, Any, List

import networkx as nx


class GraphBuilder:
    """Create a topological workflow graph for a search session."""

    # Color map (front-end picks color by node.type)
    NODE_TYPES = {
        "service": "#3b82f6",  # blue
        "process": "#14b8a6",  # teal
        "result": "#ec4899",   # pink
        "meta": "#9ca3af",     # gray
    }

    EDGE_TYPES = {
        "data": "#c084fc",      # purple
        "control": "#f97316",   # orange
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def create_workflow_graph(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Return vis.js-compatible graph dict from stored session data."""
        
        g = nx.DiGraph()

        # 1. Core services – always four
        services = [
            ("confluence", "Confluence Search"),
            ("teams", "Teams Search"),
            ("outlook", "Outlook Search"),
            ("local_disk", "Local Disk Search"),
        ]
        for svc_id, svc_label in services:
            g.add_node(
                svc_id,
                label=svc_label,
                type="service",
                status="done",   # could be dynamic
            )

        # 2. Query process node
        qp_id = "query_proc"
        g.add_node(qp_id, label="Query Processor", type="process", status="done")

        # 3. Edges from process → services (control) and back (data)
        for svc_id, _ in services:
            g.add_edge(qp_id, svc_id, type="control", weight=1.0)
            g.add_edge(svc_id, qp_id, type="data", weight=1.0)

        # 4. Result nodes – top 3 recommendations
        for rec in session.get("analysis", {}).get("top_recommendations", [])[:3]:
            rec_id = self._hash(rec["title"])[:10]
            g.add_node(
                rec_id,
                label=f"{rec['title'][:40]}…",  # truncate label
                type="result",
                status="done",
                metadata={
                    "confidence": rec.get("confidence_score"),
                    "source": rec.get("source"),
                },
            )
            # Edge from processor to result
            g.add_edge(qp_id, rec_id, type="data", weight=1.0)

        # 5. Meta node – expanded query
        meta_id = "expanded_q"
        g.add_node(
            meta_id,
            label="Expanded Query",  # could include keywords
            type="meta",
            status="done",
        )
        g.add_edge(meta_id, qp_id, type="data", weight=0.5)
        g.add_edge(qp_id, meta_id, type="control", weight=0.2)

        # Layout – simple spring layout for ~10 nodes
        pos = nx.spring_layout(g, seed=42, k=0.9)
        # Attach coords so the front-end can skip layouting
        for nid, (x, y) in pos.items():
            g.nodes[nid]["x"] = float(x * 400)  # scale for vis.js
            g.nodes[nid]["y"] = float(y * 400)

        # Build plain JSON lists
        nodes = [
            {
                "id": nid,
                "label": data["label"],
                "group": data["type"],
                "status": data.get("status"),
                "x": data.get("x"),
                "y": data.get("y"),
            }
            for nid, data in g.nodes(data=True)
        ]
        edges = [
            {
                "from": src,
                "to": tgt,
                "type": data.get("type", "data"),
                "weight": data.get("weight", 1.0),
            }
            for src, tgt, data in g.edges(data=True)
        ]

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()
