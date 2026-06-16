from typing import Protocol, runtime_checkable


@runtime_checkable
class RecommenderProtocol(Protocol):
    def recommend(
        self,
        user_id: str | int,
        k: int,
        **params: float | int | str,
    ) -> list[str | int]: ...
