from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from ..models import RouteScope


_ROUTE_SCOPE: ContextVar[RouteScope | None] = ContextVar("route_scope", default=None)


def current_route_scope() -> Optional[RouteScope]:
    return _ROUTE_SCOPE.get()


@contextmanager
def route_scope(scope: RouteScope) -> Iterator[RouteScope]:
    token = _ROUTE_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _ROUTE_SCOPE.reset(token)
