# file: rag_chatbot/context_providers.py

from dataclasses import dataclass
from typing import List
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
from rag_chatbot.config import MAX_TOTAL_TOKENS
import tiktoken

@dataclass
class ChunkItem:
    content: str
    metadata: dict

class RAGContextProvider(SystemPromptContextProviderBase):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.chunks: List[ChunkItem] = []
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    def _count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tiktoken."""
        return len(self.tokenizer.encode(text))

    def get_info(self) -> str:
        """Return context, truncated if exceeding MAX_TOTAL_TOKENS."""
        if not self.chunks:
            return "No context chunks available."
        
        total_tokens = 0
        truncated_chunks = []
        for idx, item in enumerate(self.chunks, 1):
            chunk_tokens = self._count_tokens(item.content)
            if total_tokens + chunk_tokens > MAX_TOTAL_TOKENS:
                remaining = MAX_TOTAL_TOKENS - total_tokens
                if remaining > 0:
                    tokens = self.tokenizer.encode(item.content)[:remaining]
                    truncated_text = self.tokenizer.decode(tokens)
                    truncated_chunks.append(f"Chunk {idx} (truncated):\nMetadata: {item.metadata}\nContent:\n{truncated_text}\n{'-' * 80}")
                break
            truncated_chunks.append(f"Chunk {idx}:\nMetadata: {item.metadata}\nContent:\n{item.content}\n{'-' * 80}")
            total_tokens += chunk_tokens
        
        return "\n\n".join(truncated_chunks)