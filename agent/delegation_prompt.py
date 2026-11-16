"""Shared prompt guidance for workers that coordinate with other models."""

CHILD_PROCESS_GUIDANCE = (
    "When coordinating with agents or models, request only evidence or results: logs, file paths, commands run, "
    "message IDs, or the final answer. Never ask them to describe, restate, summarize or explain their own or another "
    "model's internal process. Never paste a raw transcript or hidden or private message blocks into a prompt; "
    "pass only the model's final answer."
)
