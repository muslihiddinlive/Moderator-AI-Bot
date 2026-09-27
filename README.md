# tg-mod-functions

AI function-calling tools for **Telegram group moderation**, built on
[aiogram](https://github.com/aiogram/aiogram) 3.x.

Give an LLM (OpenAI, Anthropic, or anything else that speaks JSON-schema
tools) a set of ready-made moderation functions — `ban_user`, `mute_user`,
`warn_user`, `pin_message`, `promote_user`, etc. — and let it actually
**execute** them against your group, not just describe them.

## Install

```bash
pip install tg-mod-functions
```

## Quick start

```python
import asyncio
from aiogram import Bot
from tg_mod_functions import ModerationToolkit

async def main():
    bot = Bot(token="YOUR_BOT_TOKEN")
    toolkit = ModerationToolkit(bot, max_warns=3)

    # Hand these straight to your AI provider's `tools=` param:
    openai_tools = toolkit.as_openai_tools()
    claude_tools = toolkit.as_anthropic_tools()

    # When the model returns a tool call, dispatch it by name:
    result = await toolkit.call("mute_user", chat_id=-100123456789, user_id=987654321, minutes=15)
    print(result.message)  # "User 987654321 muted (15 daqiqaga)."

asyncio.run(main())
```

## Wiring it into an Anthropic tool-use loop

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=toolkit.as_anthropic_tools(),
    messages=[{"role": "user", "content": "Bu foydalanuvchini 10 daqiqaga sustlashtir"}],
)

for block in response.content:
    if block.type == "tool_use":
        result = await toolkit.call(block.name, **block.input)
        # send result.as_dict() back as a tool_result content block
```

## Built-in tools

| Tool | What it does |
|---|---|
| `ban_user` | Ban (permanent or timed), optional message purge |
| `unban_user` | Lift a ban |
| `kick_user` | Remove without a permanent ban |
| `mute_user` | Restrict sending messages/media (timed or indefinite) |
| `unmute_user` | Restore default permissions |
| `warn_user` | Add a warning; auto-mutes at the configured `max_warns` |
| `reset_warns` | Clear a user's warning count |
| `delete_message` | Delete a message |
| `pin_message` / `unpin_message` | Pin/unpin |
| `promote_user` / `demote_user` | Grant/revoke admin rights |
| `get_member_info` | Current status + warning count |
| `approve_join_request` / `decline_join_request` | Classic join-request approval (public groups with "Approve new members") |
| `verify_join_via_webapp` | **New (Bot API 10.1+)**: show a joining user a Mini App (CAPTCHA, rules screen, etc.) before deciding — only when this bot is set as the group's *guard bot* |
| `answer_join_query` | **New (Bot API 10.1+)**: resolve a guard-bot join query directly — approve / decline / queue |
| `ban_channel_sender` / `unban_channel_sender` | Ban/unban a channel posting as an anonymous sender (channel-based spam) |
| `lockdown_chat` / `unlock_chat` | Anti-raid: restrict everyone at once, then restore the exact prior permissions |
| `clear_user_reactions` | Bulk-remove a user's recent reactions across the group (reaction-spam cleanup) |
| `remove_message_reaction` | Remove reaction(s) on a single message |
| `get_chat_admins` | List current administrators — context for the AI before granting rights or deciding |
| `get_member_count` | Total member count |

### Anti-raid lockdown

`lockdown_chat` reads the group's *current* default permissions and remembers them in memory before locking everyone out, so `unlock_chat` restores exactly what was there — not a guessed "normal" state. This memory is per-process and not persisted, so if the bot restarts mid-lockdown, `unlock_chat` falls back to a sensible default (messages + photos + videos allowed) instead of failing.

### New: Mini-App join verification (Bot API 10.1+)

Telegram now lets a group designate one bot as its **guard bot**, which
screens every join request. When that's this bot, incoming
`chat_join_request` updates carry a `query_id`, and you have 10 seconds to
either:

- call `verify_join_via_webapp` to pop open a Mini App for the user (your
  own hosted page — a CAPTCHA, a short questionnaire, "accept the rules",
  whatever you build) and later call `answer_join_query` once the Mini App
  reports back its result, or
- skip the Mini App and call `answer_join_query` straight away with
  `"approve"`, `"decline"`, or `"queue"` (leave it for a human admin).

```python
@dp.chat_join_request()
async def on_join_request(update: types.ChatJoinRequest):
    if update.query_id:
        # this bot is the group's guard bot — screen before deciding
        await toolkit.call(
            "verify_join_via_webapp",
            query_id=update.query_id,
            web_app_url="https://yourdomain.com/verify",
        )
        # ...then, once your Mini App posts its result back to your
        # server, resolve it:
        # await toolkit.call("answer_join_query", query_id=update.query_id, result="approve")
    else:
        # classic flow — e.g. let an LLM look at update.bio / the invite
        # link used and decide
        await toolkit.call("approve_join_request", chat_id=update.chat.id, user_id=update.from_user.id)
```

Note: being appointed guard bot is a per-group admin setting made in the
Telegram client (group admins choose the bot); this library only covers
responding to the queries once they arrive.

## Protecting accounts from the AI agent

An LLM driving these tools can be prompt-injected or simply mistaken —
pass `protected_user_ids` so it can never ban/mute/kick/warn/demote those
accounts (your own superadmin ID, sub-bots, etc.), regardless of what a
model decides to call:

```python
toolkit = ModerationToolkit(bot, protected_user_ids={123456789})
```

The group's **creator** is always protected automatically, even if you
don't list them — the toolkit checks the live member status before every
restrictive action.

## Warnings need storage — bring your own

Telegram has no native concept of "warnings", so the bot has to remember
them itself. Three options ship in the box:

- `InMemoryWarnStorage` (default) — a plain dict, lost on restart.
- `JSONFileWarnStorage(path)` — persists to a local JSON file, survives a
  process restart. Writes are atomic and lock-serialized, so it's safe
  under concurrent handlers. Note: Render's free tier uses an *ephemeral*
  disk, so this survives a crash/restart but **not** a redeploy — for
  that, use an external backend (Postgres/Redis add-on, or a Telegram
  supergroup as a DB) instead.
- Anything else — implement the tiny `WarnStorage` interface:

```python
from tg_mod_functions import WarnStorage, JSONFileWarnStorage

# Local file, survives restarts:
toolkit = ModerationToolkit(bot, warn_storage=JSONFileWarnStorage("warns.json"))

# Or bring your own backend:
class RedisWarnStorage(WarnStorage):
    async def add_warn(self, chat_id, user_id) -> int: ...
    async def get_warns(self, chat_id, user_id) -> int: ...
    async def reset_warns(self, chat_id, user_id) -> None: ...

toolkit = ModerationToolkit(bot, warn_storage=RedisWarnStorage())
```

## Changelog

**0.4.0**
- Added `protected_user_ids` + automatic group-creator protection: no tool
  can ban/mute/kick/warn/demote a protected account, even under a
  malicious or mistaken AI tool call.
- Added `JSONFileWarnStorage`, an atomic, restart-persistent file backend.
- `InMemoryWarnStorage` is now safe under concurrent `asyncio` tasks.
- (from 0.3.0) Added join-request handling, guard-bot Mini App
  verification, channel-sender bans, anti-raid lockdown, reaction
  moderation, and admin/member-count lookups.
- `promote_user` now exposes `can_manage_chat`, `can_manage_video_chats`,
  and `can_promote_members` (mirroring what `demote_user` already clears).

## What's *not* included (yet)

**Slow mode** (limiting how often members can post) isn't exposed by
Telegram's Bot API at all — it's only settable via a user account over
MTProto (Pyrogram/Telethon). If you need it, a `pyrogram`-backed extra
backend can be added later; it's out of scope for the Bot-API-only v1.

## Custom tools

Register your own alongside the defaults:

```python
from tg_mod_functions import Tool, ToolResult

async def my_handler(chat_id: int, user_id: int) -> ToolResult:
    ...
    return ToolResult(ok=True, message="done")

toolkit.register(Tool(
    name="my_custom_action",
    description="...",
    parameters={"chat_id": {"type": "integer"}, "user_id": {"type": "integer"}},
    required=["chat_id", "user_id"],
    handler=my_handler,
))
```

## License

MIT
