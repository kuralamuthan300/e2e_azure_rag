# Copyright (c) Microsoft. All rights reserved.

"""Deterministic LangGraph RAG agent hosted on Microsoft Foundry.

The graph retrieves from Azure AI Search through a Foundry Toolbox, grades the
retrieved context, answers from the documents or the web, and appends
structured citations. The graph uses LangGraph's ``MessagesState`` so the
Foundry Responses host can invoke it without custom protocol handling.
"""

from __future__ import annotations

import logging

from config import AgentSettings
from dotenv import load_dotenv
from grader import ContextGrader
from graph import build_graph
from llm import FoundryLLM
from toolbox import ToolboxClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_graph():
    """Create the graph used by the configuration-driven hosting entrypoint."""
    settings = AgentSettings.from_env()
    logger.info(
        "Starting rag-agent with model=%s toolbox=%s project=%s",
        settings.model_deployment,
        settings.toolbox_name,
        settings.project_endpoint,
    )
    llm = FoundryLLM(
        project_endpoint=settings.project_endpoint,
        model_deployment=settings.model_deployment,
    )
    grader = ContextGrader(
        project_endpoint=settings.project_endpoint,
        model_deployment=settings.model_deployment,
    )
    toolbox = ToolboxClient(
        project_endpoint=settings.project_endpoint,
        toolbox_name=settings.toolbox_name,
    )
    return build_graph(llm, toolbox, grader)
