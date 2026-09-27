"""Online path: query-time safety, routing and refusal.

Nodes [11] pii, [12] classifier, [13] refusal and [14] rewriter
(architecture.md sections 5.11-5.14). Nothing in this package generates an
answer; see ``mf_facts.ask`` for why that boundary is deliberate.
"""

from __future__ import annotations

__all__ = ["classifier", "pii", "refusal", "rewriter"]
