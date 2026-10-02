# llm/cortex_messages.py — Minimal client for the Snowflake Cortex Messages API
# (Anthropic Messages format, Claude models only) with JSON-schema structured output.
#
#   POST https://<host>/api/v2/cortex/v1/messages
#   body: model, max_tokens, system, messages, output_config.format = json_schema
#   reply: content[0].text is JSON matching the schema
#
# Auth, either:
#   - an existing snowflake.connector connection (session token, same as the
#     REST example: Authorization: Snowflake Token="<con.rest.token>"), or
#   - a programmatic access token (Authorization: Bearer <PAT>).

import os

import requests

ENDPOINT = "/api/v2/cortex/v1/messages"
DEFAULT_MODEL = os.getenv("CORTEX_MESSAGES_MODEL", "claude-sonnet-4-5")


class CortexMessagesError(RuntimeError):
    pass


class CortexMessagesClient:
    def __init__(self, host: str, token: str, token_type: str = "PROGRAMMATIC_ACCESS_TOKEN",
                 model: str = DEFAULT_MODEL, timeout: float = 50.0):
        if not host or not token:
            raise CortexMessagesError("Cortex Messages API needs a host and a token")
        self.host = host.replace("https://", "").rstrip("/")
        self.token, self.token_type = token, token_type
        self.model, self.timeout = model, timeout

    @classmethod
    def from_connection(cls, con, model: str = DEFAULT_MODEL) -> "CortexMessagesClient":
        """Reuse the app's Snowflake connection (no extra secret needed)."""
        token = getattr(getattr(con, "rest", None), "token", None)
        return cls(host=getattr(con, "host", None), token=token, token_type="SESSION", model=model)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json", "anthropic-version": "2023-06-01"}
        if self.token_type == "SESSION":
            h["Authorization"] = f'Snowflake Token="{self.token}"'
        else:
            h["Authorization"] = f"Bearer {self.token}"
            h["X-Snowflake-Authorization-Token-Type"] = self.token_type
        return h

    def build_body(self, system: str, user: str, schema: dict | None, max_tokens: int) -> dict:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if schema:
            body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return body

    def create(self, system: str, user: str, schema: dict | None = None, max_tokens: int = 2048) -> str:
        resp = requests.post(f"https://{self.host}{ENDPOINT}", headers=self._headers(),
                             json=self.build_body(system, user, schema, max_tokens), timeout=self.timeout)
        if resp.status_code >= 400:
            raise CortexMessagesError(f"Cortex Messages API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        if not texts:
            raise CortexMessagesError("Cortex Messages API returned no text content")
        return texts[0]
