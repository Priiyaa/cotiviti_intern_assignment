"""
Agentic investigation step. This is a real tool-calling agent, not a
single fixed-template call: Claude receives the flagged provider's stats
and a `lookup_peer_benchmark` tool, and decides on its own whether it
needs peer-comparison context before rendering a verdict.

Requires an ANTHROPIC_API_KEY environment variable.
"""

import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

TOOLS = [
    {
        "name": "lookup_peer_benchmark",
        "description": (
            "Look up average billing volume, patient count, and case-mix "
            "stats for a peer group (cluster) of similar providers, to "
            "judge whether a flagged provider's numbers are unusual "
            "relative to peers rather than in isolation. Peer groups are "
            "from a real KMeans clustering on billing volume, patient "
            "count, claim count, inpatient ratio, physician count, and "
            "chronic condition burden."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster": {
                    "type": "string",
                    "description": "The flagged provider's cluster ID, as given in the provider stats (e.g. '0', '1', '2')",
                }
            },
            "required": ["cluster"],
        },
    }
]

SYSTEM_PROMPT = """You are a payment integrity investigator reviewing a provider
flagged by an unsupervised time-series anomaly detector.

You are given ONLY the precomputed statistics in the user message. Do not
invent any numbers not listed there. If comparing this provider to its peer
group would help you judge whether the numbers are actually unusual, use the
lookup_peer_benchmark tool. You are not required to use it if the stats are
already clear.

Respond in this exact format:

STEP 1 - Signal: What does the anomaly score and aggregation method indicate?
STEP 2 - Volume check: Does this provider's claim volume/amount look unusual on its own?
STEP 3 - Peer context: How does it compare to its peer cluster? (use the tool if needed)
STEP 4 - Alternative explanation: Could this be legitimate (e.g. a larger practice, sicker patient mix) rather than fraud?
STEP 5 - Conclusion: Weigh steps 1-4 together.

VERDICT: Approve / Escalate for manual review / Deny

Keep each step to 1-2 sentences.
"""


def investigate(provider_stats: dict, peer_benchmarks: dict) -> str:
    """
    peer_benchmarks: dict like {"Low": {...}, "Mid": {...}, "High": {...}},
    computed from the real dataset (see app.py) rather than hardcoded, so
    the tool's answers are grounded in the same data the demo presents.
    """

    def lookup_peer_benchmark(cluster: str):
        return peer_benchmarks.get(str(cluster), {})

    messages = [
        {
            "role": "user",
            "content": (
                f"Flagged provider stats:\n{json.dumps(provider_stats, indent=2)}\n\n"
                "Investigate and give a verdict."
            ),
        }
    ]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "lookup_peer_benchmark":
                    result = lookup_peer_benchmark(block.input.get("cluster", "Mid"))
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            continue

        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_blocks)