---
sidebar_position: 17
title: "Heartbeats"
description: "A recurring prompt for a conversation route — /heartbeat every 10m Check the deployment."
---

# Heartbeats (`/heartbeat`)

`/heartbeat` gives a conversation route one recurring instruction. When the interval elapses, the prompt runs in the route's current session, with its conversation context.

```
/heartbeat every 10m Check the deployment and report meaningful changes
```

Gateway heartbeats reach busy sessions by default, after the next tool result. Use `--idle` to wait until the session is idle. The classic CLI remains idle-only.

## Heartbeat vs cron: which one do I want?

They look similar but serve different jobs:

| | `/heartbeat` | [`hermes cron`](./cron) |
|---|---|---|
| Runs in | **This conversation route** — the current session and its context | A fresh isolated session per tick |
| Survives process restart | Yes; gateway watches resume automatically after restart | Yes — fully durable scheduler |
| How many | One per route | Unlimited jobs |
| Best for | "Keep an eye on X *in this thread* while we work" | Standing jobs, reports, watchdogs, deliveries |

Rule of thumb: if the recurring prompt needs the conversation's context, use `/heartbeat`. If it's a self-contained job, use cron.

## Commands

| Command | What it does |
|---|---|
| `/heartbeat every <interval> [--idle or --steer] <prompt>` | Set (or replace) the route's heartbeat. Intervals: `90s`, `10m`, `2h`, `1d` (minimum 60s). Gateway delivery steers by default; `--idle` waits for idle. |
| `/heartbeat` or `/heartbeat status` | Show the heartbeat, its delivery mode, interval, and time to next fire. |
| `/heartbeat pause` | Stop firing without clearing. |
| `/heartbeat resume` | Resume (re-anchors the timer — no instant stale fire). |
| `/heartbeat clear` | Remove the heartbeat. |
| `hermes heartbeat list` | List heartbeats in the active profile, with route, status, interval, last fire time, fire count, and last error. |

On messaging platforms, send a normal message in the chat before setting or resuming a heartbeat. Local CLI, TUI, and Desktop sessions do not need this step.

`/hb` is an alias. Works on the CLI, the TUI / Desktop app, and gateway platforms (on Slack, use `/hermes heartbeat …`).

## Behavior details

- **Busy delivery by default.** A due gateway heartbeat reaches the parent agent after the next tool result, not its delegated subagents. At most one heartbeat steer is accepted per running turn; an unread steer is returned to the heartbeat schedule. The classic CLI waits for idle.
- **Idle mode.** `--idle` waits until the session is idle. An idle watched gateway route wakes proactively.
- **Missed ticks coalesce.** If several intervals pass, the next fire represents one tick, not a backlog. The timer re-anchors on every fire.
- **User messages win by default.** In idle mode, a queued user message takes priority and the heartbeat waits for the queue to drain.
- **Owned by the route.** A messaging heartbeat replies into its chat or thread even while that conversation is open in the TUI / Desktop app; the viewer never claims the tick.
- **Cache-safe.** The injected prompt is an ordinary user message. No system-prompt mutation, no toolset change.
- **Gateway recovery.** Startup restores active heartbeats in the owning profile. Paused and cleared heartbeats do not restart, and no new chat message is required.
- **Persistence and conversation boundaries.** A gateway heartbeat belongs to its canonical route, not a session ID. Resetting or starting a new session, and restarting the gateway, do not remove it: the next fire uses the route's current session. Use `/heartbeat clear` to remove it. Firing requires the owning process to be running.
- **Execution accounting.** The fire count records admitted attempts, not proof of a successful model response or outbound delivery.
- **Quiet when there is nothing to say (gateway).** A scheduled heartbeat turn may end with a bare silence marker (`NO_REPLY` / `[SILENT]`): the gateway sends nothing, unlike a human message that returns only a marker (that still gets the visible "try again" notice). While the heartbeat works, no typing indicator, tool-progress, streamed draft or "still working" bubble is posted, and a result it does deliver is routed to the chat/topic without quoting the message that set the heartbeat. Approval prompts and failures stay visible.
- **Don't-invent-work guard.** The injected prompt tells the agent to reply briefly and stop when nothing meaningful changed, so an idle heartbeat doesn't generate busywork.

## Example

```
You: /heartbeat every 15m Check whether the CI run for PR #1234 finished; summarize the result when it does

  ♥ Heartbeat set (every 15m): Check whether the CI run for PR #1234 finished; ...

[15 minutes of you working on other things in the same session]

Hermes: [Heartbeat — recurring instruction, fires every 15m]
  💻 gh pr checks 1234   (1.2s)
  CI is still running (14/37 checks complete). Nothing to report yet.
```

When the answer stops changing, `/heartbeat clear` it — or let it keep watch.
