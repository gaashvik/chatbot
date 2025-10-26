from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
from PyPDF2 import PdfReader
from document_processing.hypothetical_prompt_embeddings import (
    HypeQuestionGenerator,
    HypeEmbeddingSystem,
    HypotheticalQuestion,
)
import traceback

from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
import config


@dataclass
class UnifiedDocumentChunk:
    chunk_id: str
    chunk_content: str  # contains text, Hype Q1-Q3
    source_document: str
    chunk_type: str
    page_number: int
    metadata: Dict[str, Any]
    hypothetical_questions: List[str]
    embedding: Optional[List[float]]


class unifiedDocumentIndexBuilder:
    """
    Builds unified format indexes from documents using HyPE system with single embeddings per chunk
    """

    def __init__(self, embedding_model: str):

        self.question_generator = HypeQuestionGenerator()
        self.embedding_system = HypeEmbeddingSystem(embedding_model)
        self.embedding_dimension = 1024
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " "]
        )
        self.qdrant = QdrantClient(
            url="https://245be38a-1058-482a-9639-4b4ddd802aec.us-east-1-1.aws.cloud.qdrant.io",
            api_key=config.QDRANT_API_KEY,
        )

    def _process_pdf(self, file_path: Path) -> List[UnifiedDocumentChunk]:
        pdf = PdfReader(file_path)
        pages = [page.extract_text() for page in (pdf.pages) if page.extract_text()]
        chunks = []
        for i, p in enumerate(pages):
            docs = self.text_splitter.split_text(p)
            for c_idx, chunk_text in enumerate(docs):
                if chunk_text.strip():
                    chunk = UnifiedDocumentChunk(
                        chunk_id=f"{file_path.stem}_pdf_chunk_page_{i}_{c_idx}",
                        chunk_content=chunk_text.strip(),
                        source_document=str(file_path),
                        page_number=i + 1,
                        chunk_type="pdf_unified",
                        metadata={
                            "file_type": file_path.suffix.lower(),
                            "original_content": chunk_text.strip(),
                            "page_number": i,
                            "chunk_index": c_idx,
                        },
                        hypothetical_questions=[],
                        embedding=None,
                    )
                    chunks.append(chunk)

        return chunks

    def _generate_questions_for_chunk(
        self, chunk_dict: Dict[str, Any]
    ) -> List[HypotheticalQuestion]:
        try:
            return self.question_generator._generate_questions_for_chunk(
                chunk_dict["chunk_text"],
                chunk_id=chunk_dict["chunk_id"],
                source_document=chunk_dict["source_document"],
                page_number=chunk_dict["page_number"],
                chunk_metadata=chunk_dict["metadata"],
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"Error generating questions for chunk {chunk_dict['chunk_id']}: {e}\n{tb}"
            )
            return []

    def _generate_hype_questions_for_chunks(
        self, chunks: List[UnifiedDocumentChunk]
    ) -> List[UnifiedDocumentChunk]:
        chunk_dicts = [
            {
                "chunk_text": c.chunk_content,
                "chunk_id": c.chunk_id,
                "page_number": c.page_number,
                "source_document": c.source_document,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_chunk = {
                executor.submit(self._generate_questions_for_chunk, chunk_dict): i
                for i, chunk_dict in enumerate(chunk_dicts)
            }

            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    questions = future.result()
                    question_text = [q.question for q in questions] if questions else []
                    chunks[chunk_idx].hypothetical_questions = question_text
                except Exception as e:
                    print(f"Failed to generate questions for chunk {chunk_idx}: {e}")
                    chunks[chunk_idx].hypothetical_questions = []
        return chunks

    def _get_unified_chunk_content(self, chunk: UnifiedDocumentChunk) -> str:
        try:
            chunk_original_content = chunk.metadata.get(
                "chunk_content", chunk.chunk_content
            )
            chunk_hype_questions = chunk.hypothetical_questions
            unified_parts = []
            if chunk_original_content:
                unified_parts.append(f"Answer:{chunk_original_content}")
            for i, question in enumerate(chunk_hype_questions[:3]):
                unified_parts.append(f"Q{i+1}. {question}")
            return "\n".join(unified_parts)
        except Exception as e:
            print(f"Failed to create unified text for {chunk.chunk_id}: {e}")
            return chunk.chunk_content

    def _create_embeddings(
        self, chunks: List[UnifiedDocumentChunk]
    ) -> List[UnifiedDocumentChunk]:

        for i, chunk in enumerate(chunks):
            unified_content = self._get_unified_chunk_content(chunk)
            chunk.chunk_content = unified_content
            print("[UNIFIED CONTENT]", unified_content)
            embedding = self.embedding_system._get_single_embedding(unified_content)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            chunk.embedding = embedding
        return chunk

    def _create_index(self, chunks: List[UnifiedDocumentChunk]) -> bool:
        import uuid

        try:
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk.embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "chunk_content": chunk.chunk_content,
                        "chunk_type": chunk.chunk_type,
                        "page_number": chunk.page_number,
                        "metadata": chunk.metadata,
                    },
                )
                for chunk in chunks
            ]
            print(self.qdrant.get_collections())
            self.qdrant.upload_points(collection_name="rag_pdf_chunks", points=points)
            print("index created successfully")
            return True
        except Exception as e:
            print(f"could not create index:{e}")
            return False

    def build_and_upload_index(self, document_dir: str = "documents") -> bool:
        all_chunks = []
        processed_docs = []
        failed_docs = []
        doc_path = Path(document_dir)

        if not doc_path.exists():
            raise ValueError(f"Documents directory {document_dir} does not exist")
        for file_path in doc_path.glob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == ".pdf":
                        print(f"processing pdf{file_path.name}")
                        chunks = self._process_pdf(file_path)
                        all_chunks.extend(chunks)
                        processed_docs.append(file_path.name)
                    else:
                        print(f"unsupported file type")
                except Exception as e:
                    print(f"Error occured while trying to process file")
                    failed_docs.append(file_path.name)

        if not all_chunks:
            print("No chunks extracted from documents")
            return False

        all_chunks = self._generate_hype_questions_for_chunks(chunks)

        all_chunks = self._create_embeddings(chunks)

        outcome = self._create_index(chunks)

        if not outcome:
            print("could not uplaod index to quadrant")
            return False
        print("successfully uplaoded to qdrant")
        return True
