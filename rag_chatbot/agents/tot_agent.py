# file: rag_chatbot/agents/tot_agent.py

from typing import List
from pydantic import Field
from openai import OpenAI
import instructor

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from rag_chatbot.agents.routing_agent import DocMetadata
from rag_chatbot.config import ChatConfig

client = instructor.from_openai(OpenAI(api_key=ChatConfig.api_key))

class ToTAgentState(BaseIOSchema):
    """State tracking for the Tree-of-Thought Agent, including reasoning paths and progress."""
    chain_of_thought: str = Field(default="", description="Current chain of thought")
    sub_queries: List[str] = Field(default_factory=list, description="Generated sub-queries")
    steps: int = Field(default=0, description="Steps taken")
    is_final: bool = Field(default=False, description="Whether this is the final state")
    paths: List[str] = Field(default_factory=list, description="Alternative reasoning paths explored")

class ToTAgentInputSchema(BaseIOSchema):
    """Input schema for the Tree-of-Thought Agent."""
    partial_state: ToTAgentState = Field(..., description="Current ToT state")
    user_message: str = Field(..., description="Original user query")
    docs_metadata: List[DocMetadata] = Field(..., description="Document metadata")

class ToTAgentOutputSchema(BaseIOSchema):
    """Output schema for the Tree-of-Thought Agent."""
    updated_state: ToTAgentState = Field(..., description="Updated ToT state")

tot_agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="gpt-4o-mini",
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a Tree-of-Thought Agent.",
                "You have access to multiple documents (with doc_id, title, summary): {docs_metadata}.",
                "Explore multiple reasoning paths and stop adaptively when no new insights are gained."
            ],
            steps=[
                "1. Review the user query: {user_message} and current state: {partial_state}.",
                "2. If metadata alone or general knowledge suffices, set is_final=True and produce no sub-queries.",
                "3. Otherwise, propose up to 3 new reasoning paths (e.g., financial metrics, trends, comparisons).",
                "4. Evaluate these paths, refine sub-queries, and update chain_of_thought. The sub_queries should be directly useful for retrieval or deeper analysis.",
                "5. If no new sub-queries or insights are added, set is_final=True to stop. Otherwise keep exploring until satisfied."
            ],
            output_instructions=[
                "Update chain_of_thought with your reasoning steps.",
                "Potentially add up to 3 sub_queries that will help answer the user's question.",
                "Increment steps. If done, set is_final=True.",
                "Store any alternative lines of reasoning in paths if relevant."
            ]
        ),
        input_schema=ToTAgentInputSchema,
        output_schema=ToTAgentOutputSchema,
        temperature=0.5
    )
)
