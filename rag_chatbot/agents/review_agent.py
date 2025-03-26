# file: rag_chatbot/agents/review_agent.py

import instructor
import openai
from typing import List
from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from ..config import ChatConfig

class RAGReviewAgentInputSchema(BaseIOSchema):
    """Input schema for the Review Agent."""
    user_message: str = Field(..., description="The original user question.")
    preliminary_answer: str = Field(..., description="The preliminary answer (could be fused from multiple sub-answers).")
    plan: str = Field(..., description="Chain-of-thought or reasoning from the Planner/ToT agent.")
    chunks: List[str] = Field(default_factory=list, description="Retrieved context chunks as text.")

class RAGReviewAgentOutputSchema(BaseIOSchema):
    """Output schema for the Review Agent."""
    final_answer: str = Field(..., description="The refined or confirmed final answer.")
    reasoning: str = Field(None, description="Explanation of any refinements or validation performed.")

review_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(openai.OpenAI(api_key=ChatConfig.api_key)),
        model=ChatConfig.model,
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a Review Agent in a RAG system.",
                "Your role is to review the preliminary answer along with the user's question, the plan, and the retrieved context.",
                "Ensure the final answer is accurate, addresses all aspects of the question, and is well-supported by the provided chunks.",
                "If multiple partial answers were fused, integrate them consistently and check for contradictions or gaps."
            ],
            steps=[
                "1. Examine user_message, preliminary_answer, plan, and chunks.",
                "2. Identify any inconsistencies, missing info, or errors in the preliminary answer.",
                "3. Refine or correct the answer using the provided chunks if needed.",
                "4. If the answer is already correct and sufficient, confirm it."
            ],
            output_instructions=[
                "Output the final refined answer in 'final_answer'.",
                "Optionally provide 'reasoning' to explain any adjustments or validation you performed."
            ],
        ),
        input_schema=RAGReviewAgentInputSchema,
        output_schema=RAGReviewAgentOutputSchema,
    )
)
