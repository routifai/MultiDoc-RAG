# file: rag_chatbot/agents/planner_agent.py

from typing import List
from pydantic import Field
from openai import OpenAI
import instructor

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from .routing_agent import DocMetadata
from ..config import ChatConfig

client = instructor.from_openai(OpenAI(api_key=ChatConfig.api_key))

class RAGPlannerAgentInputSchema(BaseIOSchema):
    """
    Input schema for the RAG Planner Agent.

    Contains the user's query (user_message) and metadata for
    any uploaded or relevant documents (docs_metadata).
    """
    user_message: str = Field(..., description="The user's input message or query")
    docs_metadata: List[DocMetadata] = Field(..., description="List of document metadata with doc_id, title, summary")

class RAGPlannerAgentOutputSchema(BaseIOSchema):
    """
    Output schema for the RAG Planner Agent.

    Contains the chain-of-thought reasoning about how to break down
    the user query, and the sub_queries to be used in retrieval (if needed).
    """
    reasoning: str = Field(..., description="The agent's reasoning for how to decompose the query")
    sub_queries: List[str] = Field(..., description="List of up to 3 sub-queries or an empty list if no retrieval is needed")

planner_agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model=ChatConfig.model,
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a RAG Planner Agent.",
                "Break down complex queries into prioritized sub-queries or detect if metadata alone can answer."
            ],
            steps=[
                "1. Analyze the user's query: {user_message}.",
                "2. Review docs_metadata: {docs_metadata}.",
                "3. If query asks for titles/summaries only, return empty sub_queries and explain in reasoning.",
                "4. Otherwise, break into up to 3 sub-queries, prioritizing specificity and relevance.",
                "5. Merge overlapping sub-queries and rank by importance."
            ],
            output_instructions=[
                "Return 'reasoning' (string) and 'sub_queries' (list of strings, max 3).",
                "If metadata alone suffices, set sub_queries = [] and explain in 'reasoning'."
            ]
        ),
        input_schema=RAGPlannerAgentInputSchema,
        output_schema=RAGPlannerAgentOutputSchema,
        temperature=0.3
    )
)
