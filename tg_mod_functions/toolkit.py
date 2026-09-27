"""tg_mod_functions.toolkit

A registry of Telegram group-moderation "tools" that can be handed straight
to an LLM's function-calling / tool-use API (OpenAI or Anthropic format), and
that also know how to execute themselves against a live aiogram ``Bot``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from aiogram import Bot
from aiogram.types import ChatPermissions

from .storage import InMemoryWarnStorage, WarnStorage


@dataclass
class ToolResult:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Handy for feeding straight back into an LLM as a tool result."""
        return {"ok": self.ok, "message": self.message, **self.data}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON-schema "properties"
    required: list[str]
    handler: Callable[..., Awaitable[ToolResult]]

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


class ModerationToolkit:
    """Wraps an aiogram ``Bot`` with a set of moderation tools an AI agent can call.

    Example
    -------
    ::

        bot = Bot(token=BOT_TOKEN)
        toolkit = ModerationToolkit(bot)

        # hand these straight to your AI provider's `tools=` param:
        openai_tools = toolkit.as_openai_tools()
        claude_tools = toolkit.as_anthropic_tools()

        # when the model returns a tool call, dispatch it:
        result = await toolkit.call("mute_user", chat_id=-100123, user_id=456, minutes=10)
        print(result.message)
    """

    def __init__(
        self,
        bot: Bot,
        warn_storage: Optional[WarnStorage] = None,
        max_warns: int = 3,
        protected_user_ids: Optional[set[int]] = None,
    ) -> None:
        self.bot = bot
        self.warn_storage = warn_storage or InMemoryWarnStorage()
        self.max_warns = max_warns
        # Owner/superadmin IDs that no tool call is allowed to touch (ban,
        # mute, kick, demote, ...), even if an LLM is instructed or tricked
        # into targeting them. Checked against every destructive action.
        self.protected_user_ids = set(protected_user_ids or ())
        self._tools: dict[str, Tool] = {}
        self._pre_lockdown_permissions: dict[int, ChatPermissions] = {}
        self._register_defaults()

    # -- registry ----------------------------------------------------------
    def register(self, tool: Tool) -> None:
        """Add a custom tool, or override a built-in one by reusing its name."""
        self._tools[tool.name] = tool

    def as_openai_tools(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def as_anthropic_tools(self) -> list[dict]:
        return [t.to_anthropic_schema() for t in self._tools.values()]

    async def call(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self._tools:
            return ToolResult(ok=False, message=f"Unknown tool: {name}")
        try:
            return await self._tools[name].handler(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surface Telegram API errors to the caller
            return ToolResult(ok=False, message=f"{type(exc).__name__}: {exc}")

    # -- safety --------------------------------------------------------------
    async def _guard(self, chat_id: int, user_id: int) -> Optional[ToolResult]:
        """Refuse an action against a protected account or the group's creator.

        Returns a ToolResult to short-circuit the caller, or None to proceed.
        Call this at the top of every handler that can restrict/remove/demote
        a user, so a misbehaving or manipulated AI agent can't be talked into
        banning the bot's own owner or the group's creator.
        """
        if user_id in self.protected_user_ids:
            return ToolResult(ok=False, message=f"User {user_id} is protected; refusing to act on this account.")
        try:
            member = await self.bot.get_chat_member(chat_id=chat_id, user_id=user_id)
        except Exception:  # noqa: BLE001 - if we can't check, fail safe below via handler's own error
            return None
        if getattr(member, "status", None) == "creator":
            return ToolResult(ok=False, message=f"User {user_id} is the group's creator; refusing to act on this account.")
        return None


    # -- default tool definitions -------------------------------------------
    def _register_defaults(self) -> None:
        self.register(Tool(
            name="ban_user",
            description=(
                "Permanently or temporarily ban a user from the group, "
                "optionally deleting their recent messages."
            ),
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID to ban"},
                "minutes": {"type": "integer", "description": "Ban duration in minutes; omit or 0 for a permanent ban"},
                "revoke_messages": {"type": "boolean", "description": "Also delete the user's recent messages"},
            },
            required=["chat_id", "user_id"],
            handler=self._ban_user,
        ))
        self.register(Tool(
            name="unban_user",
            description="Lift a ban, allowing the user to rejoin the group.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID to unban"},
            },
            required=["chat_id", "user_id"],
            handler=self._unban_user,
        ))
        self.register(Tool(
            name="kick_user",
            description="Remove a user from the group without a permanent ban (they can rejoin via invite link).",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID to kick"},
            },
            required=["chat_id", "user_id"],
            handler=self._kick_user,
        ))
        self.register(Tool(
            name="mute_user",
            description="Restrict a user from sending messages/media for a given duration (or indefinitely).",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID to mute"},
                "minutes": {"type": "integer", "description": "Mute duration in minutes; omit or 0 for indefinite"},
            },
            required=["chat_id", "user_id"],
            handler=self._mute_user,
        ))
        self.register(Tool(
            name="unmute_user",
            description="Restore a user's default permission to send messages/media.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID to unmute"},
            },
            required=["chat_id", "user_id"],
            handler=self._unmute_user,
        ))
        self.register(Tool(
            name="warn_user",
            description=(
                "Add a warning to a user. If their warning count reaches the "
                "configured max, they are auto-muted for 60 minutes and the "
                "count resets."
            ),
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID to warn"},
                "reason": {"type": "string", "description": "Why the user is being warned"},
            },
            required=["chat_id", "user_id"],
            handler=self._warn_user,
        ))
        self.register(Tool(
            name="reset_warns",
            description="Reset a user's warning count back to zero.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID"},
            },
            required=["chat_id", "user_id"],
            handler=self._reset_warns,
        ))
        self.register(Tool(
            name="delete_message",
            description="Delete a specific message in the group.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "message_id": {"type": "integer", "description": "ID of the message to delete"},
            },
            required=["chat_id", "message_id"],
            handler=self._delete_message,
        ))
        self.register(Tool(
            name="pin_message",
            description="Pin a message in the group.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "message_id": {"type": "integer", "description": "ID of the message to pin"},
                "silent": {"type": "boolean", "description": "Pin without sending a notification (default true)"},
            },
            required=["chat_id", "message_id"],
            handler=self._pin_message,
        ))
        self.register(Tool(
            name="unpin_message",
            description="Unpin a message (or the most recently pinned message if message_id is omitted).",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "message_id": {"type": "integer", "description": "ID of the message to unpin; omit to unpin the latest"},
            },
            required=["chat_id"],
            handler=self._unpin_message,
        ))
        self.register(Tool(
            name="promote_user",
            description="Grant a user admin rights in the group (moderation-focused rights by default).",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID to promote"},
                "can_delete_messages": {"type": "boolean", "description": "Default true"},
                "can_restrict_members": {"type": "boolean", "description": "Default true"},
                "can_pin_messages": {"type": "boolean", "description": "Default true"},
                "can_invite_users": {"type": "boolean", "description": "Default true"},
                "can_manage_chat": {"type": "boolean", "description": "Default false"},
                "can_manage_video_chats": {"type": "boolean", "description": "Default false"},
                "can_promote_members": {"type": "boolean", "description": "Let this admin promote others too. Default false — grant with care."},
            },
            required=["chat_id", "user_id"],
            handler=self._promote_user,
        ))
        self.register(Tool(
            name="demote_user",
            description="Remove all admin rights from a user.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID to demote"},
            },
            required=["chat_id", "user_id"],
            handler=self._demote_user,
        ))
        self.register(Tool(
            name="approve_join_request",
            description=(
                "Approve a pending request to join the group (classic flow, for groups "
                "with 'Approve new members' enabled on their invite link)."
            ),
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID whose join request to approve"},
            },
            required=["chat_id", "user_id"],
            handler=self._approve_join_request,
        ))
        self.register(Tool(
            name="decline_join_request",
            description="Decline a pending request to join the group (classic flow).",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID whose join request to decline"},
            },
            required=["chat_id", "user_id"],
            handler=self._decline_join_request,
        ))
        self.register(Tool(
            name="verify_join_via_webapp",
            description=(
                "For groups where this bot is set as the 'guard bot' (Bot API 10.1+): show the "
                "joining user a Mini App (e.g. a CAPTCHA or a rules-acceptance screen) before "
                "deciding on their join request. Only usable when the incoming join request "
                "carried a query_id — must be answered (via answer_join_query or this call) "
                "within 10 seconds of receiving it."
            ),
            parameters={
                "query_id": {"type": "string", "description": "The join request's query_id (present only in guard-bot mode)"},
                "web_app_url": {"type": "string", "description": "HTTPS URL of the Mini App to show the user"},
            },
            required=["query_id", "web_app_url"],
            handler=self._verify_join_via_webapp,
        ))
        self.register(Tool(
            name="answer_join_query",
            description=(
                "For guard-bot mode (Bot API 10.1+): directly resolve a join request query "
                "without a Mini App — approve it, decline it, or queue it for another admin "
                "to decide. Must be called within 10 seconds of receiving the query_id."
            ),
            parameters={
                "query_id": {"type": "string", "description": "The join request's query_id"},
                "result": {"type": "string", "enum": ["approve", "decline", "queue"], "description": "The decision"},
            },
            required=["query_id", "result"],
            handler=self._answer_join_query,
        ))
        self.register(Tool(
            name="ban_channel_sender",
            description=(
                "Ban a channel from posting in the group as an anonymous sender. Use this "
                "when a channel (not a regular user) is spamming the group by posting on "
                "behalf of itself."
            ),
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "sender_chat_id": {"type": "integer", "description": "The channel's chat ID to ban"},
            },
            required=["chat_id", "sender_chat_id"],
            handler=self._ban_channel_sender,
        ))
        self.register(Tool(
            name="unban_channel_sender",
            description="Lift a ban on a channel that was posting in the group as an anonymous sender.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "sender_chat_id": {"type": "integer", "description": "The channel's chat ID to unban"},
            },
            required=["chat_id", "sender_chat_id"],
            handler=self._unban_channel_sender,
        ))
        self.register(Tool(
            name="lockdown_chat",
            description=(
                "Anti-raid: temporarily restrict EVERYONE in the group from sending "
                "messages/media (the group's own default permissions are saved first, so "
                "unlock_chat can restore them exactly)."
            ),
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
            },
            required=["chat_id"],
            handler=self._lockdown_chat,
        ))
        self.register(Tool(
            name="unlock_chat",
            description="Lift a lockdown, restoring the group's default permissions from before lockdown_chat was called.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
            },
            required=["chat_id"],
            handler=self._unlock_chat,
        ))
        self.register(Tool(
            name="clear_user_reactions",
            description="Remove up to 10,000 recent message reactions added by a specific user across the group (e.g. reaction-spam cleanup).",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID whose reactions to remove"},
            },
            required=["chat_id", "user_id"],
            handler=self._clear_user_reactions,
        ))
        self.register(Tool(
            name="remove_message_reaction",
            description="Remove the reaction(s) on one specific message, optionally limited to a specific user.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "message_id": {"type": "integer", "description": "Target message ID"},
                "user_id": {"type": "integer", "description": "Only remove this user's reaction; omit to remove all"},
            },
            required=["chat_id", "message_id"],
            handler=self._remove_message_reaction,
        ))
        self.register(Tool(
            name="get_chat_admins",
            description="List the group's current administrators (and owner), useful context before granting rights or making moderation decisions.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "include_bots": {"type": "boolean", "description": "Include bot administrators (default false)"},
            },
            required=["chat_id"],
            handler=self._get_chat_admins,
        ))
        self.register(Tool(
            name="get_member_count",
            description="Get the total number of members in the group.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
            },
            required=["chat_id"],
            handler=self._get_member_count,
        ))
        self.register(Tool(
            name="get_member_info",
            description="Get a user's current status in the group (member/administrator/kicked/restricted) plus their warning count.",
            parameters={
                "chat_id": {"type": "integer", "description": "Telegram group chat ID"},
                "user_id": {"type": "integer", "description": "Telegram user ID"},
            },
            required=["chat_id", "user_id"],
            handler=self._get_member_info,
        ))

    # -- handlers ------------------------------------------------------------
    @staticmethod
    def _until(minutes: Optional[int]) -> Optional[int]:
        if not minutes:
            return None
        return int(time.time()) + minutes * 60

    async def _ban_user(self, chat_id: int, user_id: int, minutes: int = 0, revoke_messages: bool = False) -> ToolResult:
        if blocked := await self._guard(chat_id, user_id):
            return blocked
        await self.bot.ban_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            until_date=self._until(minutes),
            revoke_messages=revoke_messages,
        )
        duration = f"{minutes} daqiqaga" if minutes else "butunlay"
        return ToolResult(ok=True, message=f"User {user_id} banned ({duration}).", data={"minutes": minutes})

    async def _unban_user(self, chat_id: int, user_id: int) -> ToolResult:
        await self.bot.unban_chat_member(chat_id=chat_id, user_id=user_id, only_if_banned=True)
        return ToolResult(ok=True, message=f"User {user_id} unbanned.")

    async def _kick_user(self, chat_id: int, user_id: int) -> ToolResult:
        if blocked := await self._guard(chat_id, user_id):
            return blocked
        await self.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
        await self.bot.unban_chat_member(chat_id=chat_id, user_id=user_id, only_if_banned=True)
        return ToolResult(ok=True, message=f"User {user_id} kicked.")

    async def _mute_user(self, chat_id: int, user_id: int, minutes: int = 0) -> ToolResult:
        if blocked := await self._guard(chat_id, user_id):
            return blocked
        await self.bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=ChatPermissions(
                can_send_messages=False,
                can_send_audios=False,
                can_send_documents=False,
                can_send_photos=False,
                can_send_videos=False,
                can_send_video_notes=False,
                can_send_voice_notes=False,
                can_send_polls=False,
                can_send_other_messages=False,
                can_add_web_page_previews=False,
            ),
            until_date=self._until(minutes),
        )
        duration = f"{minutes} daqiqaga" if minutes else "muddatsiz"
        return ToolResult(ok=True, message=f"User {user_id} muted ({duration}).", data={"minutes": minutes})

    async def _unmute_user(self, chat_id: int, user_id: int) -> ToolResult:
        # Restore the GROUP's actual default permissions, not a hardcoded
        # "everything allowed" set — some groups restrict e.g. polls or media
        # for everyone by default, and giving this one user more than that
        # would be a privilege escalation, not an unmute.
        chat = await self.bot.get_chat(chat_id)
        permissions = chat.permissions or ChatPermissions(can_send_messages=True)
        await self.bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=permissions,
        )
        return ToolResult(ok=True, message=f"User {user_id} unmuted.")

    async def _warn_user(self, chat_id: int, user_id: int, reason: str = "") -> ToolResult:
        if blocked := await self._guard(chat_id, user_id):
            return blocked
        count = await self.warn_storage.add_warn(chat_id, user_id)
        if count >= self.max_warns:
            await self._mute_user(chat_id, user_id, minutes=60)
            await self.warn_storage.reset_warns(chat_id, user_id)
            return ToolResult(
                ok=True,
                message=f"User {user_id} reached {count}/{self.max_warns} warns and was auto-muted for 60 min.",
                data={"warns": count, "auto_muted": True},
            )
        return ToolResult(
            ok=True,
            message=f"User {user_id} warned ({count}/{self.max_warns}). Reason: {reason or 'none given'}",
            data={"warns": count, "auto_muted": False},
        )

    async def _reset_warns(self, chat_id: int, user_id: int) -> ToolResult:
        await self.warn_storage.reset_warns(chat_id, user_id)
        return ToolResult(ok=True, message=f"Warns reset for user {user_id}.")

    async def _delete_message(self, chat_id: int, message_id: int) -> ToolResult:
        await self.bot.delete_message(chat_id=chat_id, message_id=message_id)
        return ToolResult(ok=True, message=f"Message {message_id} deleted.")

    async def _pin_message(self, chat_id: int, message_id: int, silent: bool = True) -> ToolResult:
        await self.bot.pin_chat_message(chat_id=chat_id, message_id=message_id, disable_notification=silent)
        return ToolResult(ok=True, message=f"Message {message_id} pinned.")

    async def _unpin_message(self, chat_id: int, message_id: Optional[int] = None) -> ToolResult:
        await self.bot.unpin_chat_message(chat_id=chat_id, message_id=message_id)
        return ToolResult(ok=True, message="Message unpinned.")

    async def _promote_user(
        self,
        chat_id: int,
        user_id: int,
        can_delete_messages: bool = True,
        can_restrict_members: bool = True,
        can_pin_messages: bool = True,
        can_invite_users: bool = True,
        can_manage_chat: bool = False,
        can_manage_video_chats: bool = False,
        can_promote_members: bool = False,
    ) -> ToolResult:
        await self.bot.promote_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            can_delete_messages=can_delete_messages,
            can_restrict_members=can_restrict_members,
            can_pin_messages=can_pin_messages,
            can_invite_users=can_invite_users,
            can_manage_chat=can_manage_chat,
            can_manage_video_chats=can_manage_video_chats,
            can_promote_members=can_promote_members,
        )
        return ToolResult(ok=True, message=f"User {user_id} promoted to admin.")

    async def _demote_user(self, chat_id: int, user_id: int) -> ToolResult:
        if blocked := await self._guard(chat_id, user_id):
            return blocked
        await self.bot.promote_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            can_delete_messages=False,
            can_restrict_members=False,
            can_pin_messages=False,
            can_invite_users=False,
            can_manage_chat=False,
            can_manage_video_chats=False,
            can_promote_members=False,
        )
        return ToolResult(ok=True, message=f"User {user_id} demoted.")

    async def _approve_join_request(self, chat_id: int, user_id: int) -> ToolResult:
        await self.bot.approve_chat_join_request(chat_id=chat_id, user_id=user_id)
        return ToolResult(ok=True, message=f"Join request from user {user_id} approved.")

    async def _decline_join_request(self, chat_id: int, user_id: int) -> ToolResult:
        await self.bot.decline_chat_join_request(chat_id=chat_id, user_id=user_id)
        return ToolResult(ok=True, message=f"Join request from user {user_id} declined.")

    async def _verify_join_via_webapp(self, query_id: str, web_app_url: str) -> ToolResult:
        await self.bot.send_chat_join_request_web_app(
            chat_join_request_query_id=query_id,
            web_app_url=web_app_url,
        )
        return ToolResult(ok=True, message="Mini App verification sent to the joining user.")

    async def _answer_join_query(self, query_id: str, result: str) -> ToolResult:
        if result not in ("approve", "decline", "queue"):
            return ToolResult(ok=False, message="result must be one of: approve, decline, queue")
        await self.bot.answer_chat_join_request_query(
            chat_join_request_query_id=query_id,
            result=result,
        )
        return ToolResult(ok=True, message=f"Join request query resolved: {result}.")

    async def _ban_channel_sender(self, chat_id: int, sender_chat_id: int) -> ToolResult:
        await self.bot.ban_chat_sender_chat(chat_id=chat_id, sender_chat_id=sender_chat_id)
        return ToolResult(ok=True, message=f"Channel {sender_chat_id} banned from posting.")

    async def _unban_channel_sender(self, chat_id: int, sender_chat_id: int) -> ToolResult:
        await self.bot.unban_chat_sender_chat(chat_id=chat_id, sender_chat_id=sender_chat_id)
        return ToolResult(ok=True, message=f"Channel {sender_chat_id} unbanned.")

    async def _lockdown_chat(self, chat_id: int) -> ToolResult:
        chat = await self.bot.get_chat(chat_id)
        # Save whatever permissions are currently in effect so unlock_chat can
        # restore them exactly, rather than guessing at "normal" permissions.
        self._pre_lockdown_permissions[chat_id] = chat.permissions or ChatPermissions(can_send_messages=True)
        await self.bot.set_chat_permissions(
            chat_id=chat_id,
            permissions=ChatPermissions(
                can_send_messages=False,
                can_send_audios=False,
                can_send_documents=False,
                can_send_photos=False,
                can_send_videos=False,
                can_send_video_notes=False,
                can_send_voice_notes=False,
                can_send_polls=False,
                can_send_other_messages=False,
                can_add_web_page_previews=False,
                can_invite_users=False,
                can_pin_messages=False,
                can_change_info=False,
            ),
        )
        return ToolResult(ok=True, message=f"Chat {chat_id} locked down — everyone restricted.")

    async def _unlock_chat(self, chat_id: int) -> ToolResult:
        permissions = self._pre_lockdown_permissions.pop(chat_id, None)
        if permissions is None:
            # No lockdown on record (e.g. process restarted) — fall back to a
            # sensible default rather than doing nothing.
            permissions = ChatPermissions(can_send_messages=True, can_send_photos=True, can_send_videos=True)
        await self.bot.set_chat_permissions(chat_id=chat_id, permissions=permissions)
        return ToolResult(ok=True, message=f"Chat {chat_id} unlocked.")

    async def _clear_user_reactions(self, chat_id: int, user_id: int) -> ToolResult:
        await self.bot.delete_all_message_reactions(chat_id=chat_id, user_id=user_id)
        return ToolResult(ok=True, message=f"Cleared recent reactions by user {user_id}.")

    async def _remove_message_reaction(self, chat_id: int, message_id: int, user_id: Optional[int] = None) -> ToolResult:
        await self.bot.delete_message_reaction(chat_id=chat_id, message_id=message_id, user_id=user_id)
        return ToolResult(ok=True, message=f"Reaction(s) removed from message {message_id}.")

    async def _get_chat_admins(self, chat_id: int, include_bots: bool = False) -> ToolResult:
        admins = await self.bot.get_chat_administrators(chat_id=chat_id, return_bots=include_bots)
        summary = [
            {
                "user_id": a.user.id,
                "name": a.user.full_name,
                "username": a.user.username,
                "status": a.status,
                "is_anonymous": getattr(a, "is_anonymous", False),
            }
            for a in admins
        ]
        return ToolResult(ok=True, message=f"{len(summary)} administrator(s) found.", data={"admins": summary})

    async def _get_member_count(self, chat_id: int) -> ToolResult:
        count = await self.bot.get_chat_member_count(chat_id=chat_id)
        return ToolResult(ok=True, message=f"{count} member(s) in the chat.", data={"count": count})

    async def _get_member_info(self, chat_id: int, user_id: int) -> ToolResult:
        member = await self.bot.get_chat_member(chat_id=chat_id, user_id=user_id)
        warns = await self.warn_storage.get_warns(chat_id, user_id)
        return ToolResult(
            ok=True,
            message=f"Status: {member.status}, warns: {warns}/{self.max_warns}",
            data={"status": member.status, "warns": warns},
        )
