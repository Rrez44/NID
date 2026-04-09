"""
Thin httpx wrapper around the IDS FastAPI server.

Provides three methods that match the endpoints in api/serve.py exactly:
    - health()  -> dict from GET /health
    - classes() -> list[str] from GET /classes
    - predict(features) -> dict from POST /predict

Errors are raised as ``IDSAPIError`` subclasses with actionable messages so
cli.py can map them to exit codes without parsing httpx exceptions itself.
"""

from __future__ import annotations

import time
from typing import Any

import httpx


class IDSAPIError(Exception):
    """Base for all IDS API client errors."""


class APIUnreachable(IDSAPIError):
    """Connection refused / DNS / timeout — server probably not running."""


class APIModelNotLoaded(IDSAPIError):
    """HTTP 503 from /predict — server is up but artifacts are missing."""


class APIBadRequest(IDSAPIError):
    """HTTP 422 from /predict — typically a missing-features schema mismatch."""

    def __init__(self, message: str, missing: list[str] | None = None):
        super().__init__(message)
        self.missing = missing or []


class APIUnexpectedError(IDSAPIError):
    """5xx after retries, or any unparseable response."""


class IDSClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "IDSClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----- endpoints -----

    def health(self) -> dict[str, Any]:
        try:
            r = self._client.get("/health")
        except httpx.ConnectError as e:
            raise APIUnreachable(f"IDS API unreachable at {self.base_url}: {e}") from e
        except httpx.HTTPError as e:
            raise APIUnreachable(f"IDS API request failed at {self.base_url}: {e}") from e
        if r.status_code != 200:
            raise APIUnexpectedError(f"/health returned HTTP {r.status_code}: {r.text}")
        return r.json()

    def classes(self) -> list[str]:
        try:
            r = self._client.get("/classes")
        except httpx.ConnectError as e:
            raise APIUnreachable(f"IDS API unreachable at {self.base_url}: {e}") from e
        except httpx.HTTPError as e:
            raise APIUnreachable(f"IDS API request failed at {self.base_url}: {e}") from e
        if r.status_code == 503:
            raise APIModelNotLoaded("Model not loaded on server")
        if r.status_code != 200:
            raise APIUnexpectedError(f"/classes returned HTTP {r.status_code}: {r.text}")
        return list(r.json().get("classes", []))

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """POST /predict with retry on 5xx / connection errors."""
        body = {"features": features}
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post("/predict", json=body)
            except httpx.ConnectError as e:
                last_exc = e
                if attempt < self.retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise APIUnreachable(
                    f"IDS API unreachable at {self.base_url}: {e}"
                ) from e
            except httpx.HTTPError as e:
                last_exc = e
                if attempt < self.retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise APIUnexpectedError(
                    f"IDS API request failed: {e}"
                ) from e

            if r.status_code == 200:
                return r.json()
            if r.status_code == 503:
                raise APIModelNotLoaded("Model not loaded on server")
            if r.status_code == 422:
                detail = ""
                missing: list[str] = []
                try:
                    body_json = r.json()
                    detail = str(body_json.get("detail", ""))
                    # Server format: "Missing features: ['Foo', 'Bar']"
                    if detail.startswith("Missing features:"):
                        # Best-effort parse — exact list isn't critical, the
                        # message text is what users see.
                        import ast
                        try:
                            missing = ast.literal_eval(detail.split(":", 1)[1].strip())
                        except (ValueError, SyntaxError):
                            missing = []
                except ValueError:
                    detail = r.text
                raise APIBadRequest(detail or "HTTP 422", missing=missing)
            if 500 <= r.status_code < 600 and attempt < self.retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise APIUnexpectedError(
                f"/predict returned HTTP {r.status_code}: {r.text}"
            )

        # Should be unreachable, but mypy/typecheckers like a fallback.
        raise APIUnexpectedError(f"predict failed after retries: {last_exc}")
