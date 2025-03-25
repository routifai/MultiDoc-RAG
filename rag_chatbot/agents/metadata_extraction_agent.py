# file: rag_chatbot/agents/metadata_extraction_agent.py

import instructor
import openai
from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from rag_chatbot.config import ChatConfig
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
MAX_INPUT_TOKENS = 1000  # Limit input to 1000 tokens for efficiency

class MetadataExtractionAgentInputSchema(BaseIOSchema):
    """Input schema for the Metadata Extraction Agent."""
    doc_id: str = Field(..., description="Unique ID of the document")
    doc_text: str = Field(..., description=f"First {MAX_INPUT_TOKENS} tokens of document text for metadata extraction")

class MetadataExtractionAgentOutputSchema(BaseIOSchema):
    """Output schema for the Metadata Extraction Agent."""
    doc_id: str = Field(..., description="Unique ID of the document")
    title: str = Field(..., description="Generated or extracted title (max 50 characters)")
    summary: str = Field(..., description="Brief summary of the document (max 100 words)")

metadata_extraction_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(openai.OpenAI(api_key=ChatConfig.api_key)),
        model=ChatConfig.model,
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a Metadata Extraction Agent.",
                f"Generate a title and summary from the first {MAX_INPUT_TOKENS} tokens of document text."
            ],
            steps=[
                f"1. Analyze the provided text (limited to {MAX_INPUT_TOKENS} tokens).",
                "2. Extract or generate a title from the first 500 characters if possible, otherwise infer from content.",
                "3. Generate a concise summary capturing the main topic or key points."
            ],
            output_instructions=[
                "Return 'doc_id', 'title', and 'summary'.",
                "Title: Short and descriptive (max 50 characters).",
                "Summary: Concise (max 100 words) and informative."
            ]
        ),
        input_schema=MetadataExtractionAgentInputSchema,
        output_schema=MetadataExtractionAgentOutputSchema,
    )
)

def truncate_text(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    """Truncate text to a maximum token count."""
    tokens = tokenizer.encode(text)[:max_tokens]
    return tokenizer.decode(tokens)