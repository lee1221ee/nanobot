from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.discord import DiscordChannel, DiscordConfig


def _make_channel(**overrides: object) -> DiscordChannel:
    cfg = DiscordConfig(enabled=True, token="fake-token", allow_from=["*"], **overrides)
    ch = DiscordChannel(cfg, MessageBus())
    ch._http = AsyncMock(spec=httpx.AsyncClient)
    ch._bot_user_id = "BOT123"
    return ch


def _guild_message_payload(
    *,
    content: str = "hello",
    channel_id: str = "CH100",
    message_id: str = "MSG200",
    guild_id: str | None = "GUILD1",
    author_id: str = "USER1",
    mentions_bot: bool = True,
) -> dict:
    mentions = [{"id": "BOT123"}] if mentions_bot else []
    return {
        "id": message_id,
        "channel_id": channel_id,
        "content": content,
        "guild_id": guild_id,
        "author": {"id": author_id, "bot": False},
        "mentions": mentions,
        "attachments": [],
    }


# -- Config defaults ---------------------------------------------------------


def test_config_reply_in_thread_defaults_true() -> None:
    cfg = DiscordConfig()
    assert cfg.reply_in_thread is True


# -- Send to thread channel --------------------------------------------------


@pytest.mark.asyncio
async def test_send_to_thread_channel() -> None:
    ch = _make_channel()
    ch._http.post = AsyncMock(
        return_value=AsyncMock(status_code=200, raise_for_status=lambda: None)
    )

    await ch.send(
        OutboundMessage(
            channel="discord",
            chat_id="THREAD999",
            content="reply in thread",
        )
    )

    call_args = ch._http.post.call_args
    assert "/channels/THREAD999/messages" in call_args.args[0]


# -- Thread creation for guild messages --------------------------------------


@pytest.mark.asyncio
async def test_handle_creates_thread_for_guild_message() -> None:
    ch = _make_channel()
    ch._http.post = AsyncMock(
        return_value=AsyncMock(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"id": "THREAD_NEW"},
        )
    )

    payload = _guild_message_payload()

    with (
        patch.object(ch, "_handle_message", new_callable=AsyncMock) as mock_handle,
        patch.object(ch, "_start_typing", new_callable=AsyncMock) as mock_typing,
    ):
        await ch._handle_message_create(payload)

    # _create_thread should have been called via _http.post
    create_call = ch._http.post.call_args_list[0]
    assert "/threads" in create_call.args[0]

    # _handle_message should receive thread ID as chat_id
    mock_handle.assert_called_once()
    kwargs = mock_handle.call_args.kwargs
    assert kwargs["chat_id"] == "THREAD_NEW"
    assert kwargs["session_key"] == "discord:THREAD_NEW"

    # Typing indicator should be sent to the thread, not the original channel
    mock_typing.assert_called_once_with("THREAD_NEW")

    # Thread should be tracked
    assert "THREAD_NEW" in ch._known_threads


# -- DM skips thread --------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_skips_thread_for_dm() -> None:
    ch = _make_channel()
    payload = _guild_message_payload(guild_id=None)

    with patch.object(ch, "_handle_message", new_callable=AsyncMock) as mock_handle:
        await ch._handle_message_create(payload)

    mock_handle.assert_called_once()
    kwargs = mock_handle.call_args.kwargs
    assert kwargs["chat_id"] == "CH100"
    assert kwargs["session_key"] is None


# -- Disabled reply_in_thread ------------------------------------------------


@pytest.mark.asyncio
async def test_handle_skips_thread_when_disabled() -> None:
    ch = _make_channel(reply_in_thread=False)
    payload = _guild_message_payload()

    with patch.object(ch, "_handle_message", new_callable=AsyncMock) as mock_handle:
        await ch._handle_message_create(payload)

    mock_handle.assert_called_once()
    kwargs = mock_handle.call_args.kwargs
    assert kwargs["chat_id"] == "CH100"
    assert kwargs["session_key"] is None


# -- Reuses existing thread --------------------------------------------------


@pytest.mark.asyncio
async def test_handle_reuses_existing_thread() -> None:
    ch = _make_channel()
    ch._known_threads.add("CH100")
    payload = _guild_message_payload(channel_id="CH100")

    with patch.object(ch, "_handle_message", new_callable=AsyncMock) as mock_handle:
        await ch._handle_message_create(payload)

    mock_handle.assert_called_once()
    kwargs = mock_handle.call_args.kwargs
    assert kwargs["chat_id"] == "CH100"
    assert kwargs["session_key"] == "discord:CH100"

    # No HTTP call for thread creation
    ch._http.post.assert_not_called()


# -- Fallback on thread creation failure -------------------------------------


@pytest.mark.asyncio
async def test_handle_fallback_on_thread_creation_failure() -> None:
    ch = _make_channel()
    ch._http.post = AsyncMock(side_effect=Exception("API error"))
    payload = _guild_message_payload()

    with (
        patch.object(ch, "_handle_message", new_callable=AsyncMock) as mock_handle,
        patch("nanobot.channels.discord.asyncio.sleep", new_callable=AsyncMock),
    ):
        await ch._handle_message_create(payload)

    mock_handle.assert_called_once()
    kwargs = mock_handle.call_args.kwargs
    # Falls back to original channel_id
    assert kwargs["chat_id"] == "CH100"
    assert kwargs["session_key"] is None

    # All 3 retry attempts should have been made
    assert ch._http.post.call_count == 3


# -- GUILD_CREATE populates known threads ------------------------------------


@pytest.mark.asyncio
async def test_guild_create_populates_known_threads() -> None:
    ch = _make_channel()
    assert len(ch._known_threads) == 0

    payload = {
        "threads": [
            {"id": "T1"},
            {"id": "T2"},
            {"id": "T3"},
        ]
    }
    await ch._dispatch_event(0, "GUILD_CREATE", payload)

    assert ch._known_threads == {"T1", "T2", "T3"}


# -- THREAD_CREATE updates known threads -------------------------------------


@pytest.mark.asyncio
async def test_thread_create_updates_known_threads() -> None:
    ch = _make_channel()
    assert "T_NEW" not in ch._known_threads

    await ch._dispatch_event(0, "THREAD_CREATE", {"id": "T_NEW"})

    assert "T_NEW" in ch._known_threads


# -- THREAD_DELETE removes known threads -------------------------------------


@pytest.mark.asyncio
async def test_thread_delete_removes_known_thread() -> None:
    ch = _make_channel()
    ch._known_threads.add("T_OLD")

    await ch._dispatch_event(0, "THREAD_DELETE", {"id": "T_OLD"})

    assert "T_OLD" not in ch._known_threads


# -- Rate-limit retry on thread creation -------------------------------------


@pytest.mark.asyncio
async def test_create_thread_retries_on_rate_limit() -> None:
    ch = _make_channel()
    rate_limit_resp = AsyncMock(
        status_code=429,
        json=lambda: {"retry_after": 0.01},
    )
    success_resp = AsyncMock(
        status_code=200,
        raise_for_status=lambda: None,
        json=lambda: {"id": "THREAD_RL"},
    )
    ch._http.post = AsyncMock(side_effect=[rate_limit_resp, success_resp])

    result = await ch._create_thread("CH1", "MSG1", "test")

    assert result == "THREAD_RL"
    assert ch._http.post.call_count == 2
    assert "THREAD_RL" in ch._known_threads
