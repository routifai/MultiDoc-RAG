# file: rag_chatbot/agents/routing_agent.py

import instructor
import openai
from typing import List, Optional
from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from rag_chatbot.config import ChatConfig

class DocMetadata(BaseIOSchema):
    """Metadata for a document."""
    doc_id: str = Field(..., description="Unique ID for the document")
    title: Optional[str] = Field(None, description="Descriptive title")
    summary: Optional[str] = Field(None, description="Brief summary or key points")

class RoutingAgentInputSchema(BaseIOSchema):
    """Input schema for the Routing Agent."""
    user_message: str = Field(..., description="The user's question or request.")
    docs_metadata: List[DocMetadata] = Field(..., description="List of documents with metadata.")

class RoutingAgentOutputSchema(BaseIOSchema):
    """Output schema for the Routing Agent."""
    relevant_docs: List[str] = Field(..., description="List of relevant doc_ids.")
    reasoning: Optional[str] = Field(None, description="Explanation or chain-of-thought")

SYSTEM_PROMPT = """
You are a Document Routing Agent.
Given a list of documents (doc_id, title, summary), determine which are relevant to the user's question.
Documents: {{docs_metadata}}
User's question: {{user_message}}
"""

routing_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(openai.OpenAI(api_key=ChatConfig.api_key)),
        model=ChatConfig.model,
        system_prompt_generator=SystemPromptGenerator(
            background=[SYSTEM_PROMPT],
            steps=[
                "1. Read the user's question.",
                "2. Evaluate each document's title and summary.",
                "3. Select relevant doc_ids."
            ],
            output_instructions=[
                "Return relevant doc_ids in 'relevant_docs'.",
                "Optionally, explain in 'reasoning'."
            ],
        ),
        input_schema=RoutingAgentInputSchema,
        output_schema=RoutingAgentOutputSchema,
    )
)