"""Async fan-out event broadcaster.

Routes RealtimeEvents from the ContextManager to multiple consumers
(WebSocket clients, TUI, etc.) without coupling them to each other.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from autonote.realtime.models import RealtimeEvent

logger = logging.getLogger(__name__)

EventCallback = Callable[[RealtimeEvent], Awaitable[None]]


class EventBus:
    """Fan-out broadcaster: one publisher, many async subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[EventCallback] = []
        self._lock = asyncio.Lock()

    async def subscribe(self, callback: EventCallback) -> None:
        async with self._lock:
            self._subscribers.append(callback)

    async def unsubscribe(self, callback: EventCallback) -> None:
        async with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    async def publish(self, event: RealtimeEvent) -> None:
        """Send *event* to every subscriber. Errors in one don't affect others."""
        async with self._lock:
            targets = list(self._subscribers)
        for cb in targets:
            try:
                await cb(event)
            except Exception:
                logger.warning("EventBus: subscriber %r failed", cb, exc_info=True)
