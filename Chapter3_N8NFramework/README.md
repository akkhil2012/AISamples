## Chapter 3 – n8n Workflow Framework

This chapter contains ready-to-import n8n workflows for data comparison and email/Gmail automations, plus an orchestrator flow for chaining tasks.

### Folder Structure

```
Chapter3_N8NFramework/
├── n8n/
│   ├── datacomparator_WF/
│   │   └── n8n-workflow.json       # Full workflow export for Data Comparator
│   ├── datacomparator.json         # Standalone Data Comparator workflow
│   ├── DataComparatorSteps.txt     # Step-by-step notes for Data Comparator
│   ├── gmailDelete.json            # Gmail delete automation
│   ├── orchestrator.json           # Orchestrates/links multiple workflows
│   ├── testgmailSpam.json          # Gmail spam test workflow (variant 1)
│   ├── testgmailSpam1.json         # Gmail spam test workflow (variant 2)
│   ├── testgmailSpam2.json         # Gmail spam test workflow (variant 3)
```

### Prerequisites

- n8n installed locally or hosted
  - Docker (recommended): `docker run -it --rm -p 5678:5678 n8nio/n8n`
  - Or see n8n docs: `https://docs.n8n.io/`
- Required credentials configured inside n8n (e.g., Gmail OAuth2) for email flows

### Importing a Workflow

1. Open n8n at `http://localhost:5678`.
2. Click “Workflows” → “Import from File”.
3. Select any of the provided `.json` files (e.g., `n8n/datacomparator.json`).
4. Review nodes, set credentials (e.g., Gmail), and save.

### Workflows Overview

- Data Comparator (`n8n/datacomparator.json` or `n8n/datacomparator_WF/n8n-workflow.json`)
  - Compares two data sources (files/APIs/arrays) to surface differences.
  - See `DataComparatorSteps.txt` for the node-by-node logic and configuration hints.

- Gmail Automation
  - `gmailDelete.json`: Deletes or labels emails based on rules/filters.
  - `testgmailSpam*.json`: Variants to detect, label, or move spam-like emails for testing.
  - Ensure Gmail credentials are set in Credentials → Google → Gmail API.

- Orchestrator (`orchestrator.json`)
  - A control workflow to trigger/sequence other workflows (via Webhook/HTTP Request nodes or n8n Execute Workflow node).
  - Useful for batching, scheduling, or conditional execution of the above flows.

### How to Run (Typical)

1. Import a workflow JSON.
2. Open the workflow → configure nodes (paths, queries, labels, credentials).
3. “Execute Workflow” to test manually.
4. Optionally add a Trigger (Cron/Webhook) to automate.

### Tips

- Always validate node inputs/outputs with the “Execute Node” button before full runs.
- Keep credentials in n8n’s Credentials manager; avoid hardcoding secrets in nodes.
- For Gmail flows, start with “read-only” actions (list/search) before enabling delete/move.

### Troubleshooting

- Missing credentials: Set them in Credentials and rebind nodes with the key icon.
- Gmail auth errors: Reconnect OAuth with correct scopes (read, modify, delete if needed).
- Data Comparator mismatches: Inspect input schema with Code/Function nodes; normalize keys and types first.

### License

Provided as educational examples. Review and adapt before use in production.


