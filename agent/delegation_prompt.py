"""Shared prompt guidance for workers that coordinate with other models."""

CHILD_PROCESS_GUIDANCE = (
    "When coordinating with agents or models, request only evidence or results: logs, file paths, commands, message "
    "IDs, or final summaries. Do not ask them to describe their work process. Never put raw transcripts or private "
    "blocks into a prompt; pass only a model's final answer."
)
