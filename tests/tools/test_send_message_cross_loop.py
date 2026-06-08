"""Regression tests for the cross-event-loop deadlock fix in send_message.

When the agent's tool worker thread calls _send_via_adapter() while the
adapter's queues live on the gateway's main event loop, the send must be
dispatched via run_coroutine_threadsafe to the gateway loop — NOT awaited
directly on the worker loop (which would deadlock due to the selector never
being woken by cross-thread future.set_result).
"""

import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from gateway.config import Platform


class TestSendViaAdapterCrossLoopDispatch:

    @pytest.mark.asyncio
    async def test_cross_loop_dispatches_to_gateway_loop(self, monkeypatch):
        """adapter.send() runs on gateway loop, not the caller's loop."""
        from tools.send_message_tool import _send_via_adapter

        send_loop_id = {}
        platform = Platform("wecom")

        class FakeAdapter:
            async def send(self, *, chat_id, content, metadata=None):
                send_loop_id["loop"] = id(asyncio.get_running_loop())
                return SimpleNamespace(success=True, message_id="cross-ok")

        gateway_loop = asyncio.new_event_loop()
        started = threading.Event()

        def run_gateway():
            asyncio.set_event_loop(gateway_loop)
            started.set()
            gateway_loop.run_forever()

        t = threading.Thread(target=run_gateway, daemon=True)
        t.start()
        started.wait(timeout=2)

        try:
            runner = SimpleNamespace(
                adapters={platform: FakeAdapter()},
                _gateway_loop=gateway_loop,
            )
            fake_gateway_run = ModuleType("gateway.run")
            fake_gateway_run._gateway_runner_ref = lambda: runner
            monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

            result = await _send_via_adapter(
                platform,
                SimpleNamespace(extra={}),
                "wr_group_123",
                "hello from worker",
            )

            assert result == {"success": True, "message_id": "cross-ok"}
            # Verify send() ran on the gateway loop, not our current loop
            assert send_loop_id["loop"] == id(gateway_loop)
        finally:
            gateway_loop.call_soon_threadsafe(gateway_loop.stop)
            t.join(timeout=2)
            gateway_loop.close()

    @pytest.mark.asyncio
    async def test_same_loop_uses_direct_await(self, monkeypatch):
        """When current loop IS the gateway loop, adapter.send() is awaited
        directly — no run_coroutine_threadsafe (which would self-lock)."""
        from tools.send_message_tool import _send_via_adapter

        current_loop = asyncio.get_running_loop()
        platform = Platform("wecom")
        called_directly = {}

        class FakeAdapter:
            async def send(self, *, chat_id, content, metadata=None):
                called_directly["loop"] = id(asyncio.get_running_loop())
                return SimpleNamespace(success=True, message_id="direct-ok")

        runner = SimpleNamespace(
            adapters={platform: FakeAdapter()},
            _gateway_loop=current_loop,
        )
        fake_gateway_run = ModuleType("gateway.run")
        fake_gateway_run._gateway_runner_ref = lambda: runner
        monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

        result = await _send_via_adapter(
            platform,
            SimpleNamespace(extra={}),
            "wr_group_456",
            "direct send",
        )

        assert result == {"success": True, "message_id": "direct-ok"}
        assert called_directly["loop"] == id(current_loop)

    @pytest.mark.asyncio
    async def test_gateway_loop_not_running_returns_error(self, monkeypatch):
        """When gateway loop exists but is stopped, return an error rather
        than attempting direct await on a loop-bound adapter."""
        from tools.send_message_tool import _send_via_adapter

        stopped_loop = asyncio.new_event_loop()
        stopped_loop.close()
        platform = Platform("wecom")

        class FakeAdapter:
            async def send(self, *, chat_id, content, metadata=None):
                raise AssertionError("should not be called")

        runner = SimpleNamespace(
            adapters={platform: FakeAdapter()},
            _gateway_loop=stopped_loop,
        )
        fake_gateway_run = ModuleType("gateway.run")
        fake_gateway_run._gateway_runner_ref = lambda: runner
        monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

        result = await _send_via_adapter(
            platform,
            SimpleNamespace(extra={}),
            "wr_group_789",
            "should fail",
        )

        assert "error" in result
        assert "not running" in result["error"]

    @pytest.mark.asyncio
    async def test_shield_prevents_cancel_of_enqueued_send(self, monkeypatch):
        """asyncio.shield ensures that cancelling the caller does NOT cancel
        the already-dispatched send on the gateway loop."""
        from tools.send_message_tool import _send_via_adapter

        send_completed = asyncio.Event()
        send_result_holder = {}
        platform = Platform("wecom")

        class FakeAdapter:
            async def send(self, *, chat_id, content, metadata=None):
                # Simulate a slow send (token bucket wait)
                await asyncio.sleep(0.3)
                send_result_holder["sent"] = True
                send_completed.set()
                return SimpleNamespace(success=True, message_id="shielded")

        gateway_loop = asyncio.new_event_loop()
        started = threading.Event()

        def run_gateway():
            asyncio.set_event_loop(gateway_loop)
            started.set()
            gateway_loop.run_forever()

        t = threading.Thread(target=run_gateway, daemon=True)
        t.start()
        started.wait(timeout=2)

        try:
            runner = SimpleNamespace(
                adapters={platform: FakeAdapter()},
                _gateway_loop=gateway_loop,
            )
            fake_gateway_run = ModuleType("gateway.run")
            fake_gateway_run._gateway_runner_ref = lambda: runner
            monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

            # Start the send, then cancel the caller task after a short delay
            async def do_send():
                return await _send_via_adapter(
                    platform,
                    SimpleNamespace(extra={}),
                    "wr_group_shield",
                    "shielded msg",
                )

            task = asyncio.create_task(do_send())
            await asyncio.sleep(0.1)  # let it dispatch to gateway loop
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            # The send on the gateway loop should still complete despite cancel
            fut = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(send_completed.wait(), timeout=1.0),
                gateway_loop,
            )
            fut.result(timeout=2)
            assert send_result_holder.get("sent") is True
        finally:
            gateway_loop.call_soon_threadsafe(gateway_loop.stop)
            t.join(timeout=2)
            gateway_loop.close()

    @pytest.mark.asyncio
    async def test_cross_loop_dispatches_send_media_to_gateway_loop(
        self, monkeypatch, tmp_path
    ):
        """plugin send_media() must hop to the gateway loop, same as send()."""
        from tools.send_message_tool import _send_via_adapter

        send_loop_id = {}
        recorded = {}
        platform = SimpleNamespace(value="wecom_stream")
        adapter_key = platform.value
        media_path = tmp_path / "photo.png"
        media_path.write_bytes(b"png")

        class FakeAdapter:
            async def send_media(self, *, chat_id, file_path, media_type, metadata=None):
                send_loop_id["loop"] = id(asyncio.get_running_loop())
                recorded["chat_id"] = chat_id
                recorded["file_path"] = file_path
                recorded["media_type"] = media_type
                return SimpleNamespace(success=True, message_id="media-ok")

            async def send(self, *, chat_id, content, metadata=None):
                raise AssertionError("text send should not run for media-only")

        gateway_loop = asyncio.new_event_loop()
        started = threading.Event()

        def run_gateway():
            asyncio.set_event_loop(gateway_loop)
            started.set()
            gateway_loop.run_forever()

        t = threading.Thread(target=run_gateway, daemon=True)
        t.start()
        started.wait(timeout=2)

        try:
            runner = SimpleNamespace(
                adapters={adapter_key: FakeAdapter()},
                _gateway_loop=gateway_loop,
            )
            fake_gateway_run = ModuleType("gateway.run")
            fake_gateway_run._gateway_runner_ref = lambda: runner
            monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

            result = await _send_via_adapter(
                platform,
                SimpleNamespace(extra={}),
                "wr_group_media",
                "",
                media_files=[(str(media_path), False)],
            )

            assert result == {
                "success": True,
                "message_id": "media-ok",
                "media_delivered": True,
            }
            assert send_loop_id["loop"] == id(gateway_loop)
            assert recorded["media_type"] == "image"
            assert recorded["file_path"] == str(media_path)
        finally:
            gateway_loop.call_soon_threadsafe(gateway_loop.stop)
            t.join(timeout=2)
            gateway_loop.close()

    @pytest.mark.asyncio
    async def test_native_send_image_file_when_adapter_has_no_send_media(
        self, monkeypatch, tmp_path
    ):
        """Adapters without send_media use BasePlatformAdapter native methods."""
        from tools.send_message_tool import _send_via_adapter

        recorded = {}
        platform = Platform("wecom")
        image_path = tmp_path / "shot.jpg"
        image_path.write_bytes(b"jpg")

        class FakeAdapter:
            async def send_image_file(self, chat_id, image_path, **kwargs):
                recorded["image_path"] = image_path
                recorded["chat_id"] = chat_id
                recorded["kwargs"] = kwargs
                return SimpleNamespace(success=True, message_id="img-ok")

            async def send(self, *, chat_id, content, metadata=None):
                raise AssertionError(
                    "caption should ride on the media bubble, not a separate send"
                )

        runner = SimpleNamespace(adapters={platform: FakeAdapter()})
        fake_gateway_run = ModuleType("gateway.run")
        fake_gateway_run._gateway_runner_ref = lambda: runner
        monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

        result = await _send_via_adapter(
            platform,
            SimpleNamespace(extra={}),
            "wr_group_native",
            "caption",
            media_files=[(str(image_path), False)],
        )

        assert result == {
            "success": True,
            "message_id": "img-ok",
            "media_delivered": True,
        }
        assert recorded["image_path"] == str(image_path)
        assert recorded["kwargs"]["caption"] == "caption"


def test_plugin_platform_media_only_is_not_rejected(monkeypatch, tmp_path):
    """Plugin platforms (not in Platform enum) must not hit the builtin MEDIA gate."""
    from tools.send_message_tool import _send_to_platform

    platform = SimpleNamespace(value="wecom_stream")
    recorded = {}
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"mp4")

    class FakeAdapter:
        async def send_media(self, *, chat_id, file_path, media_type, metadata=None):
            recorded["file_path"] = file_path
            recorded["media_type"] = media_type
            return SimpleNamespace(success=True, message_id="plugin-media")

        async def send(self, *, chat_id, content, metadata=None):
            raise AssertionError("media-only send should not call send()")

    runner = SimpleNamespace(adapters={platform.value: FakeAdapter()})
    fake_gateway_run = ModuleType("gateway.run")
    fake_gateway_run._gateway_runner_ref = lambda: runner
    monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

    pconfig = SimpleNamespace(enabled=True, token=None, extra={})
    result = asyncio.run(
        _send_to_platform(
            platform,
            pconfig,
            "wo_user_1",
            "",
            media_files=[(str(clip_path), False)],
        )
    )

    assert result == {
        "success": True,
        "message_id": "plugin-media",
        "media_delivered": True,
    }
    assert recorded["media_type"] == "video"


def test_media_type_for_plugin_send_uses_shared_extension_sets():
    from tools.send_message_tool import _media_type_for_plugin_send

    assert _media_type_for_plugin_send("/x.png", False) == "image"
    assert _media_type_for_plugin_send("/x.mp4", False) == "video"
    assert _media_type_for_plugin_send("/x.ogg", True) == "voice"
    assert _media_type_for_plugin_send("/x.pdf", False) == "file"
