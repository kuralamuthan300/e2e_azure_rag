from __future__ import annotations

import asyncio

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from openai import AsyncOpenAI

_AZURE_AI_SCOPE = "https://ai.azure.com/.default"


class FoundryLLM:
    """Small direct Foundry Responses client used by graph nodes.

    Calling the model directly, rather than through LangChain, keeps
    intermediate model output out of the Foundry host's message stream.
    """

    def __init__(
        self,
        project_endpoint: str,
        model_deployment: str,
        *,
        credential: TokenCredential | None = None,
    ) -> None:
        self._project_endpoint = project_endpoint.rstrip("/")
        self._model_deployment = model_deployment
        self._credential = credential or DefaultAzureCredential()

    async def complete(self, prompt: str) -> str:
        token = await asyncio.to_thread(
            self._credential.get_token,
            _AZURE_AI_SCOPE,
        )
        client = AsyncOpenAI(
            base_url=f"{self._project_endpoint}/openai/v1",
            api_key=token.token,
        )
        response = await client.responses.create(
            model=self._model_deployment,
            input=prompt,
        )
        return response.output_text.strip()
