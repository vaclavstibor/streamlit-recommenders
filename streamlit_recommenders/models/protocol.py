from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RecommenderProtocol(Protocol):
    def get_recommendations(
        self,
        user_id: str | int,
        k: int,
        session_items: list[str | int] | None = None,
        selections: list[dict] | None = None,
        **params: Any,
    ) -> list[str | int]: ...
