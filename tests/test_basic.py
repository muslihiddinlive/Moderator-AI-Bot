import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tg_mod_functions import JSONFileWarnStorage, ModerationToolkit


def make_toolkit(max_warns: int = 3, protected_user_ids=None) -> ModerationToolkit:
    bot = AsyncMock()
    # Default: get_chat_member reports a normal, non-creator member so
    # existing tests don't need to know about the guard.
    bot.get_chat_member.return_value = SimpleNamespace(status="member")
    return ModerationToolkit(bot, max_warns=max_warns, protected_user_ids=protected_user_ids)


def test_schemas_have_all_tools():
    toolkit = make_toolkit()
    openai_tools = toolkit.as_openai_tools()
    claude_tools = toolkit.as_anthropic_tools()
    names = {t["function"]["name"] for t in openai_tools}
    assert names == {t["name"] for t in claude_tools}
    assert "ban_user" in names
    assert "mute_user" in names
    assert "warn_user" in names


@pytest.mark.asyncio
async def test_ban_user_calls_bot_api():
    toolkit = make_toolkit()
    result = await toolkit.call("ban_user", chat_id=-100, user_id=1, minutes=10)
    assert result.ok
    toolkit.bot.ban_chat_member.assert_awaited_once()
    _, kwargs = toolkit.bot.ban_chat_member.await_args
    assert kwargs["chat_id"] == -100
    assert kwargs["user_id"] == 1
    assert kwargs["until_date"] is not None


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    toolkit = make_toolkit()
    result = await toolkit.call("does_not_exist", chat_id=1, user_id=2)
    assert not result.ok
    assert "Unknown tool" in result.message


@pytest.mark.asyncio
async def test_warn_user_auto_mutes_at_max():
    toolkit = make_toolkit(max_warns=2)
    r1 = await toolkit.call("warn_user", chat_id=-1, user_id=5, reason="spam")
    assert r1.data["warns"] == 1
    assert r1.data["auto_muted"] is False

    r2 = await toolkit.call("warn_user", chat_id=-1, user_id=5, reason="spam again")
    assert r2.data["warns"] == 2
    assert r2.data["auto_muted"] is True
    toolkit.bot.restrict_chat_member.assert_awaited()

    # warns reset after auto-mute
    remaining = await toolkit.warn_storage.get_warns(-1, 5)
    assert remaining == 0


@pytest.mark.asyncio
async def test_classic_join_request_flow():
    toolkit = make_toolkit()
    approve = await toolkit.call("approve_join_request", chat_id=-1, user_id=42)
    assert approve.ok
    toolkit.bot.approve_chat_join_request.assert_awaited_once_with(chat_id=-1, user_id=42)

    decline = await toolkit.call("decline_join_request", chat_id=-1, user_id=42)
    assert decline.ok
    toolkit.bot.decline_chat_join_request.assert_awaited_once_with(chat_id=-1, user_id=42)


@pytest.mark.asyncio
async def test_guard_bot_join_query_flow():
    toolkit = make_toolkit()
    verify = await toolkit.call("verify_join_via_webapp", query_id="q1", web_app_url="https://example.com/verify")
    assert verify.ok
    toolkit.bot.send_chat_join_request_web_app.assert_awaited_once_with(
        chat_join_request_query_id="q1", web_app_url="https://example.com/verify"
    )

    answer = await toolkit.call("answer_join_query", query_id="q1", result="approve")
    assert answer.ok
    toolkit.bot.answer_chat_join_request_query.assert_awaited_once_with(
        chat_join_request_query_id="q1", result="approve"
    )

    bad = await toolkit.call("answer_join_query", query_id="q1", result="banana")
    assert not bad.ok


@pytest.mark.asyncio
async def test_ban_and_unban_channel_sender():
    toolkit = make_toolkit()
    result = await toolkit.call("ban_channel_sender", chat_id=-1, sender_chat_id=-100999)
    assert result.ok
    toolkit.bot.ban_chat_sender_chat.assert_awaited_once_with(chat_id=-1, sender_chat_id=-100999)

    result = await toolkit.call("unban_channel_sender", chat_id=-1, sender_chat_id=-100999)
    assert result.ok
    toolkit.bot.unban_chat_sender_chat.assert_awaited_once_with(chat_id=-1, sender_chat_id=-100999)


@pytest.mark.asyncio
async def test_lockdown_saves_and_restores_original_permissions():
    toolkit = make_toolkit()
    original_permissions = SimpleNamespace(can_send_messages=True, can_send_polls=False)
    toolkit.bot.get_chat.return_value = SimpleNamespace(permissions=original_permissions)

    lock_result = await toolkit.call("lockdown_chat", chat_id=-1)
    assert lock_result.ok
    # everyone restricted
    _, lock_kwargs = toolkit.bot.set_chat_permissions.await_args
    assert lock_kwargs["permissions"].can_send_messages is False

    unlock_result = await toolkit.call("unlock_chat", chat_id=-1)
    assert unlock_result.ok
    _, unlock_kwargs = toolkit.bot.set_chat_permissions.await_args
    # restored the exact original permissions object, not a hardcoded default
    assert unlock_kwargs["permissions"] is original_permissions


@pytest.mark.asyncio
async def test_unlock_without_prior_lockdown_falls_back_gracefully():
    toolkit = make_toolkit()
    result = await toolkit.call("unlock_chat", chat_id=-999)
    assert result.ok


@pytest.mark.asyncio
async def test_reaction_moderation():
    toolkit = make_toolkit()
    r1 = await toolkit.call("clear_user_reactions", chat_id=-1, user_id=5)
    assert r1.ok
    toolkit.bot.delete_all_message_reactions.assert_awaited_once_with(chat_id=-1, user_id=5)

    r2 = await toolkit.call("remove_message_reaction", chat_id=-1, message_id=77, user_id=5)
    assert r2.ok
    toolkit.bot.delete_message_reaction.assert_awaited_once_with(chat_id=-1, message_id=77, user_id=5)


@pytest.mark.asyncio
async def test_get_chat_admins_and_member_count():
    toolkit = make_toolkit()
    fake_admin = SimpleNamespace(
        user=SimpleNamespace(id=10, full_name="Test Admin", username="testadmin"),
        status="administrator",
        is_anonymous=False,
    )
    toolkit.bot.get_chat_administrators.return_value = [fake_admin]
    toolkit.bot.get_chat_member_count.return_value = 42

    admins = await toolkit.call("get_chat_admins", chat_id=-1)
    assert admins.ok
    assert admins.data["admins"] == [
        {"user_id": 10, "name": "Test Admin", "username": "testadmin", "status": "administrator", "is_anonymous": False}
    ]

    count = await toolkit.call("get_member_count", chat_id=-1)
    assert count.ok
    assert count.data["count"] == 42


@pytest.mark.asyncio
async def test_bot_exception_is_surfaced_not_raised():
    toolkit = make_toolkit()
    toolkit.bot.ban_chat_member.side_effect = RuntimeError("Telegram says no")
    result = await toolkit.call("ban_user", chat_id=-1, user_id=1)
    assert not result.ok
    assert "Telegram says no" in result.message


@pytest.mark.asyncio
async def test_protected_user_id_cannot_be_banned():
    toolkit = make_toolkit(protected_user_ids={999})
    result = await toolkit.call("ban_user", chat_id=-1, user_id=999)
    assert not result.ok
    assert "protected" in result.message
    toolkit.bot.ban_chat_member.assert_not_awaited()


@pytest.mark.asyncio
async def test_protected_user_id_cannot_be_muted_or_demoted():
    toolkit = make_toolkit(protected_user_ids={999})
    mute_result = await toolkit.call("mute_user", chat_id=-1, user_id=999, minutes=5)
    demote_result = await toolkit.call("demote_user", chat_id=-1, user_id=999)
    assert not mute_result.ok
    assert not demote_result.ok
    toolkit.bot.restrict_chat_member.assert_not_awaited()
    toolkit.bot.promote_chat_member.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_creator_cannot_be_banned_even_if_not_in_protected_list():
    toolkit = make_toolkit()
    toolkit.bot.get_chat_member.return_value = SimpleNamespace(status="creator")
    result = await toolkit.call("ban_user", chat_id=-1, user_id=42)
    assert not result.ok
    assert "creator" in result.message
    toolkit.bot.ban_chat_member.assert_not_awaited()


@pytest.mark.asyncio
async def test_promote_then_demote_user():
    toolkit = make_toolkit()
    promote_result = await toolkit.call("promote_user", chat_id=-1, user_id=1)
    assert promote_result.ok
    demote_result = await toolkit.call("demote_user", chat_id=-1, user_id=1)
    assert demote_result.ok
    _, promote_kwargs = toolkit.bot.promote_chat_member.await_args_list[0]
    _, demote_kwargs = toolkit.bot.promote_chat_member.await_args_list[1]
    assert promote_kwargs["can_delete_messages"] is True
    assert demote_kwargs["can_delete_messages"] is False


@pytest.mark.asyncio
async def test_get_member_info_reports_status_and_warns():
    toolkit = make_toolkit()
    toolkit.bot.get_chat_member.return_value = SimpleNamespace(status="member")
    await toolkit.call("warn_user", chat_id=-1, user_id=7, reason="test")
    result = await toolkit.call("get_member_info", chat_id=-1, user_id=7)
    assert result.ok
    assert result.data["status"] == "member"
    assert result.data["warns"] == 1


@pytest.mark.asyncio
async def test_json_file_warn_storage_persists_across_instances(tmp_path):
    path = tmp_path / "warns.json"
    storage1 = JSONFileWarnStorage(path)
    count = await storage1.add_warn(-1, 5)
    assert count == 1

    # Simulate a process restart: a brand-new instance pointed at the same file.
    storage2 = JSONFileWarnStorage(path)
    assert await storage2.get_warns(-1, 5) == 1

    await storage2.add_warn(-1, 5)
    assert await storage1.get_warns(-1, 5) == 2

    await storage2.reset_warns(-1, 5)
    assert await storage1.get_warns(-1, 5) == 0


@pytest.mark.asyncio
async def test_json_file_warn_storage_isolates_chats_and_users(tmp_path):
    storage = JSONFileWarnStorage(tmp_path / "warns.json")
    await storage.add_warn(chat_id=-1, user_id=1)
    await storage.add_warn(chat_id=-1, user_id=2)
    await storage.add_warn(chat_id=-2, user_id=1)

    assert await storage.get_warns(-1, 1) == 1
    assert await storage.get_warns(-1, 2) == 1
    assert await storage.get_warns(-2, 1) == 1
    assert await storage.get_warns(-2, 2) == 0
