import instructor
import openai
from typing import List
from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

from rag_chatbot.config import ChatConfig


class RAGQueryAgentInputSchema(BaseIOSchema):
    """Input schema for the RAG query agent."""

    user_message: str = Field(
        ...,
        description="The user's question or message to generate sub-queries for"
    )


class RAGQueryAgentOutputSchema(BaseIOSchema):
    """
    Output schema for the RAG query agent when generating
    multiple sub-queries.
    """
    sub_queries: List[str] = Field(
        ...,
        description="List of simpler sub-queries to retrieve relevant chunks"
    )
    reasoning: str = Field(
        None,
        description="Explanation of how these sub-queries were derived from the user's question"
    )


query_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(openai.OpenAI(api_key=ChatConfig.api_key)),
        model=ChatConfig.model,
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are an expert at formulating search queries for RAG systems.",
                "Your role is to break down the user's question into multiple simpler sub-questions that will collectively address the user's main information need.",
            ],
            steps=[
                "1. Analyze the user's question for distinct topics or pieces of information.",
                "2. Create multiple simpler sub-questions that capture each key concept.",
                "3. Ensure each sub-question is general enough to match relevant content but specific enough to retrieve useful chunks.",
            ],
            output_instructions=[
                "Provide a list of sub-queries in the `sub_queries` field (as many as needed).",
                "Explain your reasoning for how you derived these sub-queries.",
                "Aim for clear, concise sub-questions that will surface relevant context.",
            ],
        ),
        input_schema=RAGQueryAgentInputSchema,
        output_schema=RAGQueryAgentOutputSchema,
    )
)
