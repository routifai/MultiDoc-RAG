import os
from typing import List, Dict
import wget
import PyPDF2
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
import tiktoken

from atomic_agents.agents.base_agent import BaseAgentConfig
from rag_chatbot.agents.planner_agent import planner_agent, RAGPlannerAgentInputSchema, RAGPlannerAgentOutputSchema
from rag_chatbot.agents.query_agent import query_agent, RAGQueryAgentInputSchema, RAGQueryAgentOutputSchema
from rag_chatbot.agents.qa_agent import qa_agent, RAGQuestionAnsweringAgentInputSchema, RAGQuestionAnsweringAgentOutputSchema
from rag_chatbot.agents.review_agent import review_agent, RAGReviewAgentInputSchema, RAGReviewAgentOutputSchema
from rag_chatbot.agents.routing_agent import routing_agent, RoutingAgentInputSchema, RoutingAgentOutputSchema, DocMetadata
from rag_chatbot.agents.metadata_extraction_agent import metadata_extraction_agent, MetadataExtractionAgentInputSchema, MetadataExtractionAgentOutputSchema
from rag_chatbot.context_providers import RAGContextProvider, ChunkItem
from rag_chatbot.services.chroma_db import ChromaDBService
from rag_chatbot.config import CHUNK_SIZE, CHUNK_OVERLAP, NUM_CHUNKS_TO_RETRIEVE, CHROMA_PERSIST_DIR, MAX_CONTEXT_LENGTH, MAX_TOTAL_TOKENS, ChatConfig

MAX_SUB_QUERIES = 4

console = Console()

DOCUMENTS = {
    "TSLA_Gen": "https://ir.tesla.com/_flysystem/s3/sec/000162828024002390/tsla-20231231-gen.pdf",
    "Doc_CIK1759509": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001759509/88d50a11-dfdc-4442-b36f-a0abebf3f034.pdf",
    "Doc_CIK1543151": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001543151/6fabd79a-baa9-4b08-84fe-deab4ef8415f.pdf",
    "Doc_CIK1018724": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/e42c2068-bad5-4ab6-ae57-36ff8b2aeffd.pdf"
}

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

# Utility Functions
def download_pdf(doc_id: str, url: str) -> str:
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{doc_id}.pdf")
    if os.path.exists(output_path):
        console.print(f"[bold green]{doc_id} already exists, skipping download.[/bold green]")
    else:
        console.print(f"[bold yellow]Downloading {doc_id}...[/bold yellow]")
        wget.download(url, output_path)
        console.print(f"[bold green]{doc_id} downloaded![/bold green]")
    return output_path

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        console.print(f"[bold red]Error extracting text from {file_path}: {e}[/bold red]")
    return text

def get_title_from_metadata(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            info = reader.metadata
            return info.get("/Title", None) if info else None
    except Exception:
        return None

def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    paragraphs = text.split("\n\n") if len(text.split("\n\n")) > 1 else [text]
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end].strip())
                start += chunk_size - overlap
            continue
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_text = chunks[-1][-overlap:] if overlap > 0 and chunks else ""
            current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
        else:
            current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def limit_chunks(chunks: List[ChunkItem], max_tokens: int = MAX_CONTEXT_LENGTH) -> List[ChunkItem]:
    selected = []
    total_tokens = 0
    for chunk in chunks:
        tcount = count_tokens(chunk.content)
        if total_tokens + tcount > max_tokens:
            remaining = max_tokens - total_tokens
            if remaining > 0:
                tokens = tokenizer.encode(chunk.content)[:remaining]
                trimmed_text = tokenizer.decode(tokens)
                selected.append(ChunkItem(content=trimmed_text, metadata=chunk.metadata))
            break
        selected.append(chunk)
        total_tokens += tcount
    return selected

def truncate_text(text: str, max_tokens: int) -> str:
    tokens = tokenizer.encode(text)[:max_tokens]
    return tokenizer.decode(tokens)

# Display Functions
def display_welcome() -> None:
    welcome_msg = """
Welcome to the Multi-Document RAG Chatbot!
Ask questions about the following predefined documents:
- TSLA_Gen: Tesla Inc. Annual Report 2023
- Doc_CIK1759509: Lyft Inc. Registration Statement S-8
- Doc_CIK1543151: Uber Technologies Annual Report 2023
- Doc_CIK1018724: Amazon.com Annual Report 2024
Or ask anything else—I’ll answer directly!
Type 'exit' to quit.
"""
    console.print("\n")
    console.print(Panel(welcome_msg, title="[bold blue]Multi-Document RAG Chatbot[/bold blue]", border_style="blue", padding=(1, 2)))
    console.print("\n" + "─" * 80 + "\n")

def display_chunks(chunks: List[ChunkItem]) -> None:
    console.print("\n[bold cyan]📚 Retrieved Chunks:[/bold cyan]")
    for i, chunk in enumerate(chunks, 1):
        distance = chunk.metadata.get("distance", 0.0)
        console.print(Panel(Markdown(chunk.content), title=f"[bold]Chunk {i} (Distance: {distance:.4f})[/bold]", border_style="blue", padding=(1, 2)))

def display_panel(title: str, content: str) -> None:
    console.print(Panel.fit(content, title=title, border_style="magenta", padding=(1, 2)))

# Modular Functions
def fuse_answers(sub_answers: List[str], user_message: str) -> str:
    if len(sub_answers) <= 1:
        return sub_answers[0] if sub_answers else "No sufficient context retrieved."
    combined = "\n\n".join(sub_answers)
    input_data = RAGReviewAgentInputSchema(
        user_message=user_message,
        preliminary_answer=combined,
        plan="Synthesize sub-answers into a coherent response.",
        chunks=[]
    )
    review_output = review_agent.run(input_data)
    return review_output.final_answer.strip()

def process_planner(user_message: str, docs_metadata: List[DocMetadata], progress: Progress) -> RAGPlannerAgentOutputSchema:
    try:
        task = progress.add_task("[cyan]Planning...", total=None)
        planner_input = RAGPlannerAgentInputSchema(user_message=user_message, docs_metadata=docs_metadata)
        planner_output = planner_agent.run(planner_input)
        progress.remove_task(task)
        display_panel("Planner Agent", f"[bold yellow]Reasoning:[/bold yellow]\n{planner_output.reasoning}\n\n[bold yellow]Sub-Queries:[/bold yellow] {planner_output.sub_queries}")
        return planner_output
    except Exception as e:
        console.print(f"[bold red]Planner Error: {e}[/bold red]")
        raise

def process_sub_queries(sub_queries: List[str], chroma_db_dict: Dict[str, ChromaDBService], rag_context: RAGContextProvider, progress: Progress, docs_metadata: List[DocMetadata]) -> tuple[List[str], int, List[str]]:
    try:
        sub_answers = []
        total_tokens = 0
        used_doc_ids = set()
        
        for sq in sub_queries:
            task = progress.add_task(f"[cyan]Routing for '{sq}'...", total=None)
            routing_output = routing_agent.run(RoutingAgentInputSchema(user_message=sq, docs_metadata=docs_metadata))
            progress.remove_task(task)
            display_panel(f"Routing Agent ({sq})", 
                          f"[bold green]Docs:[/bold green] {routing_output.relevant_docs}\n\n[bold green]Reasoning:[/bold green]\n{routing_output.reasoning or 'N/A'}")

            if "title" in sq.lower() and ("document" in sq.lower() or "report" in sq.lower()):
                for doc_id in routing_output.relevant_docs:
                    title = next((meta.title for meta in docs_metadata if meta.doc_id == doc_id), doc_id)
                    sub_answers.append(f"Sub-Query: {sq}\nAnswer: {title}")
                    used_doc_ids.add(doc_id)
                continue

            all_chunks: List[ChunkItem] = []
            for doc_id in routing_output.relevant_docs:
                results = chroma_db_dict[doc_id].query(query_text=sq, n_results=NUM_CHUNKS_TO_RETRIEVE)
                for doc_text, idx, dist in zip(results["documents"], results["ids"], results["distances"]):
                    if dist < 0.7:
                        all_chunks.append(ChunkItem(content=doc_text, metadata={"doc_id": doc_id, "chunk_index": idx, "distance": dist}))
                        used_doc_ids.add(doc_id)

            if not all_chunks:
                sub_answers.append(f"Sub-Query: {sq}\nAnswer: Insufficient information.")
                continue

            limited_chunks = limit_chunks(all_chunks, MAX_CONTEXT_LENGTH)
            chunk_tokens = sum(count_tokens(c.content) for c in limited_chunks)
            if total_tokens + chunk_tokens > MAX_TOTAL_TOKENS:
                console.print(f"[bold yellow]Warning: Skipping sub-query '{sq}' due to token limit[/bold yellow]")
                break
            rag_context.chunks = limited_chunks
            total_tokens += chunk_tokens
            display_chunks(limited_chunks)

            task = progress.add_task(f"[cyan]Answering '{sq}'...", total=None)
            qa_output = qa_agent.run(RAGQuestionAnsweringAgentInputSchema(question=sq))
            progress.remove_task(task)
            sub_answer = f"Sub-Query: {sq}\nAnswer: {qa_output.answer}"
            sub_answers.append(sub_answer)
            total_tokens += count_tokens(sub_answer)
            display_panel(f"QA Agent ({sq})", f"[bold green]Answer:[/bold green]\n{qa_output.answer}\n\n[bold green]Reasoning:[/bold green]\n{qa_output.reasoning}")
        
        return sub_answers, total_tokens, list(used_doc_ids)
    except Exception as e:
        console.print(f"[bold red]Sub-Query Processing Error: {e}[/bold red]")
        raise

def review_answer(user_message: str, combined_answer: str, planner_reasoning: str, chunks: List[str], progress: Progress, docs_metadata: List[DocMetadata]) -> RAGReviewAgentOutputSchema:
    try:
        task = progress.add_task("[cyan]Reviewing...", total=None)
        if "document" in user_message.lower() and "4" in user_message.lower():
            doc_titles = [f"{meta.title}" for meta in docs_metadata][:4]
            combined_answer = "You have attached the following documents:\n" + "\n".join([f"{i+1}. {title}" for i, title in enumerate(doc_titles)])
        review_input = RAGReviewAgentInputSchema(
            user_message=user_message,
            preliminary_answer=combined_answer,
            plan=planner_reasoning or "",
            chunks=chunks
        )
        review_output = review_agent.run(review_input)
        progress.remove_task(task)
        display_panel("Final Answer (Pre-Source)", f"[bold cyan]Answer:[/bold cyan]\n{review_output.final_answer}\n\n[bold cyan]Reasoning:[/bold cyan]\n{review_output.reasoning or 'N/A'}")
        return review_output
    except Exception as e:
        console.print(f"[bold red]Review Error: {e}[/bold red]")
        raise

def chat_loop(chroma_db_dict: Dict[str, ChromaDBService], chunks_dict: Dict[str, List[str]], docs_metadata: List[DocMetadata], rag_context: RAGContextProvider) -> None:
    display_welcome()
    while True:
        try:
            user_message = console.input("\n[bold blue]Your question:[/bold blue] ").strip()
            if user_message.lower() in ChatConfig.exit_commands:
                console.print("\n[bold]Goodbye![/bold]")
                break

            console.print("\n" + "─" * 80)
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                planner_output = process_planner(user_message, docs_metadata, progress)

                if not planner_output.sub_queries and ("document" in user_message.lower() or "attached" in user_message.lower()):
                    doc_titles = [f"{meta.title}" for meta in docs_metadata]
                    answer = "You have attached the following documents:\n" + "\n".join([f"{i+1}. {title}" for i, title in enumerate(doc_titles)])
                    sources = "\n".join([f"- {title}" for title in doc_titles])
                    display_panel("Final Answer", f"[bold cyan]Answer:[/bold cyan]\n{answer}\n\nSources:\n{sources}\n\n[bold cyan]Reasoning:[/bold cyan]\nThe query asked for attached documents, so the titles were retrieved directly from the metadata.")
                    continue

                sub_queries = planner_output.sub_queries[:MAX_SUB_QUERIES] if planner_output.sub_queries else []
                console.print(f"[dim]Processing {len(sub_queries)} sub-queries[/dim]")

                if sub_queries:
                    sub_answers, total_tokens, used_doc_ids = process_sub_queries(sub_queries, chroma_db_dict, rag_context, progress, docs_metadata)
                    combined_answer = fuse_answers(sub_answers, user_message)
                    if total_tokens + count_tokens(combined_answer) > MAX_TOTAL_TOKENS:
                        combined_answer = truncate_text(combined_answer, MAX_TOTAL_TOKENS - total_tokens)

                    review_output = review_answer(user_message, combined_answer, planner_output.reasoning, [c.content for c in rag_context.chunks], progress, docs_metadata)
                    doc_id_to_title = {meta.doc_id: meta.title for meta in docs_metadata}
                    used_titles = [doc_id_to_title[doc_id] for doc_id in used_doc_ids if doc_id in doc_id_to_title]
                    sources_text = "Sources:\n" + "\n".join([f"- {title}" for title in used_titles]) if used_titles else "Sources: None"
                    final_output = f"{review_output.final_answer}\n\n{sources_text}"
                    display_panel("Final Answer", f"[bold cyan]Answer:[/bold cyan]\n{final_output}\n\n[bold cyan]Reasoning:[/bold cyan]\n{review_output.reasoning or 'N/A'}")
                    console.print(f"[dim]Total tokens used: {total_tokens}[/dim]")
                else:
                    task = progress.add_task("[cyan]Answering directly...", total=None)
                    original_chunks = rag_context.chunks
                    rag_context.chunks = []
                    qa_output = qa_agent.run(RAGQuestionAnsweringAgentInputSchema(question=user_message))
                    rag_context.chunks = original_chunks
                    progress.remove_task(task)
                    display_panel("Final Answer", f"[bold cyan]Answer:[/bold cyan]\n{qa_output.answer}\n\n[bold cyan]Reasoning:[/bold cyan]\n{qa_output.reasoning or 'N/A'}")

            console.print("\n" + "─" * 80)
        except Exception as e:
            console.print(f"\n[bold red]Chat Loop Error: {str(e)}[/bold red]")

def initialize_indexes(force_reindex: bool = False) -> tuple[Dict[str, ChromaDBService], Dict[str, List[str]], List[DocMetadata], RAGContextProvider]:
    chroma_db_dict = {}
    chunks_dict = {}
    docs_metadata = []

    for doc_id, url in DOCUMENTS.items():
        pdf_path = download_pdf(doc_id, url)
        text = extract_text_from_pdf(pdf_path)
        pdf_title = get_title_from_metadata(pdf_path)
        
        truncated_text = truncate_text(text, max_tokens=1000)
        input_data = MetadataExtractionAgentInputSchema(doc_id=doc_id, doc_text=truncated_text)
        output = metadata_extraction_agent.run(input_data)
        extracted_title = output.title.strip()
        final_title = extracted_title if extracted_title and extracted_title != doc_id else (pdf_title or doc_id)
        docs_metadata.append(DocMetadata(doc_id=doc_id, title=final_title, summary=output.summary))
        
        console.print(f"[bold cyan]Metadata for {doc_id}:[/bold cyan]")
        console.print(f"[cyan]Title:[/cyan] {final_title}")
        console.print(f"[cyan]Summary:[/cyan] {output.summary}")
        console.print("─" * 80)

        chunks = chunk_document(text)
        chunks_dict[doc_id] = chunks
        console.print(f"[dim]• {doc_id}: Created {len(chunks)} chunks[/dim]")

        persist_dir = os.path.join(CHROMA_PERSIST_DIR, doc_id)
        collection_exists = os.path.exists(persist_dir) and os.listdir(persist_dir)
        
        if collection_exists and not force_reindex:
            console.print(f"[bold green]Loading existing ChromaDB collection for {doc_id}...[/bold green]")
            chroma_db = ChromaDBService(
                collection_name=doc_id,
                persist_directory=persist_dir,
                recreate_collection=False
            )
        else:
            console.print(f"[bold yellow]Creating new ChromaDB collection for {doc_id}...[/bold yellow]")
            chroma_db = ChromaDBService(
                collection_name=doc_id,
                persist_directory=persist_dir,
                recreate_collection=True
            )
            chroma_db.add_documents(
                documents=chunks,
                metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
            )
            console.print(f"[bold green]ChromaDB collection for {doc_id} created![/bold green]")
        
        chroma_db_dict[doc_id] = chroma_db

    rag_context = RAGContextProvider("RAG Context")
    return chroma_db_dict, chunks_dict, docs_metadata, rag_context

if __name__ == "__main__":
    try:
        chroma_db_dict, chunks_dict, docs_metadata, rag_context = initialize_indexes(force_reindex=False)
        planner_agent.register_context_provider("rag_context", rag_context)
        query_agent.register_context_provider("rag_context", rag_context)
        qa_agent.register_context_provider("rag_context", rag_context)
        review_agent.register_context_provider("rag_context", rag_context)
        routing_agent.register_context_provider("rag_context", rag_context)
        chat_loop(chroma_db_dict, chunks_dict, docs_metadata, rag_context)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {str(e)}[/bold red]")