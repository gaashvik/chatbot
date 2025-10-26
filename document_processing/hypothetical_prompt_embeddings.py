#!/usr/bin/env python3
"""
Hypothetical Prompt Embeddings (HyPE) System for Lambda-friendly RAG

This system transforms retrieval from query-document matching to question-question matching
by pre-generating hypothetical questions during indexing phase, eliminating the need for
expensive runtime query expansion.

Key Features:
1. Generates 3-5 hypothetical questions per document chunk during indexing
2. Embeds questions using AWS Titan Embeddings V2 (AWS-only)
3. FAISS vector store for efficient similarity search
4. Parallel processing for question generation
5. Lambda-friendly with no runtime overhead
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from config import LLM_HYPE, EMBEDDING_MODEL_NAME, HUGGING_FACE_API
from huggingface_hub import InferenceClient


@dataclass
class HypotheticalQuestion:
    """Data class for hypothetical questions"""

    question: str
    chunk_id: str
    chunk_text: str
    chunk_metadata: str
    source_document: str
    page_number: int
    confidence_score: float
    question_type: str
    embeddings: Optional[List[float]] = None


class HypeIndex:
    """datas class for Hype index structuring"""

    questions: List[HypotheticalQuestion]
    qdrant_index: Optional[Any] = None


class HypeEmbeddingSystem:
    """Embedding system"""

    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        self.embedding_dimension = 1024
        self.embedder = client = InferenceClient(
            provider="hf-inference",
            model=self.embedding_model,
            token=HUGGING_FACE_API,
        )

    def _get_single_embedding(self, text: str) -> List[float]:
        try:
            embedding_result = self.embedder.feature_extraction(
                text=text, normalize=True, truncate=True
            )
            return embedding_result
        except Exception as e:
            print("Exceptionoccured while embedding:", e)
            return [0.0] * self.embedding_dimension


class HypeQuestionGenerator:
    """
    Generates Hypothetical questions using domain-specific prompts
    """

    def __init__(self):
        self.llm = LLM_HYPE
        self.question_templates = {
            "factual": [
                "What specific information is provided about {topic}?",
                "What are the key details mentioned regarding {topic}?",
                "What facts are stated about {topic}?",
            ],
            "definitional": [
                "How is {concept} defined or explained?",
                "What does {concept} mean in this context?",
                "How would you explain {concept}?",
            ],
            "comparative": [
                "What's the difference between {item1} and {item2}?",
                "How does {item1} compare to {item2}?",
                "What are the advantages of {item1} over {item2}?",
            ],
            "procedural": [
                "How do you {action}?",
                "What are the steps to {action}?",
                "What's the process for {action}?",
            ],
        }

    def _create_question_prompt(self, chunk: str) -> str:
        prompt = f"""
            Analyze the following information and genrate  exactly 4 domain specific essential questions that user might ask to find this specific informations.
            Requirements for each question:
                1. Be specific and directly answerable from the text
                2. Use natural language that real customers would ask
                3. Cover different aspects: factual, procedural, comparative, or definitional
                4. Be one clear sentence each
            Text to analyze:
            ```
            {chunk}
            ```
            """
        return prompt

    def _get_confidence_score(self, question: str, chunk: str):
        confidence_score = 0.7
        question_words = set(question.lower().split())
        chunk_words = set(chunk.lower().split())
        overlap = len(question_words & chunk_words)

        if overlap > 3:
            confidence_score += 0.2
        elif overlap > 1:
            confidence_score += 0.1

        if "?" in question and len(question.split()) > 5:
            confidence_score += 0.1

        return min(1, confidence_score)

    def _clean_question(self, question: str) -> str:
        """
        Clean and validate question text
        """
        # Remove numbering, bullets, etc.
        question = re.sub(r"^[\d\.\-\*\•]\s*", "", question)

        # Remove descriptive prefixes like "Casual/Conversational:", "Formal/Technical:", etc.
        question = re.sub(r"^[A-Za-z][A-Za-z/\s]*:\s*", "", question)

        # Remove leading dots and whitespace that might be left over
        question = re.sub(r"^\.\s*", "", question)

        question = question.strip()

        # Ensure it ends with question mark
        if question and not question.endswith("?"):
            question += "?"

        # Basic validation
        if len(question) < 10 or len(question) > 200:
            return ""

        return question

    def _get_question_type(self, question: str) -> str:
        """
        Classify question type for better organization
        """
        question_lower = question.lower()

        if any(
            word in question_lower
            for word in ["what is", "define", "definition", "meaning"]
        ):
            return "definitional"
        elif any(
            word in question_lower for word in ["how to", "how do", "steps", "process"]
        ):
            return "procedural"
        elif any(
            word in question_lower for word in ["difference", "compare", "versus", "vs"]
        ):
            return "comparative"
        else:
            return "factual"

    def _parse_questions(
        self,
        llm_response,
        chunk,
        chunk_id,
        source_document,
        page_number,
        chunk_metadata,
    ):
        questions = []
        lines = [
            line.strip() for line in llm_response.strip().split("\n") if line.strip()
        ]
        for i, line in enumerate(lines[:5]):
            cleaned_question = self._clean_question(line)
            if not cleaned_question:
                continue

            confidence_score = self._get_confidence_score(cleaned_question, chunk)

            question_type = self._get_question_type(cleaned_question)

            questions.append(
                HypotheticalQuestion(
                    question=cleaned_question,
                    chunk_id=chunk_id,
                    chunk_text=chunk,
                    source_document=source_document,
                    page_number=page_number,
                    chunk_metadata=chunk_metadata,
                    confidence_score=confidence_score,
                    question_type=question_type,
                )
            )
        return questions

    def _create_fallback_questions(
        self,
        chunk_text: str,
        chunk_id: str,
        source_document: str,
        page_number: int,
        chunk_metadata,
    ) -> List[HypotheticalQuestion]:
        """
        Create basic fallback questions if generation fails
        """
        fallback_questions = [
            f"What information is provided about {source_document.replace('.pdf', '').replace('_', ' ')}?",
            "What are the key details mentioned in this content?",
            "What should I know about this topic?",
        ]

        questions = []
        for i, q in enumerate(fallback_questions):
            questions.append(
                HypotheticalQuestion(
                    question=q,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    source_document=source_document,
                    page_number=page_number,
                    confidence_score=0.5,  # Lower confidence for fallback
                    question_type="factual",
                    chunk_metadata=chunk_metadata,
                )
            )

        return questions

    def _generate_questions_for_chunk(
        self,
        chunk,
        chunk_id,
        source_document,
        page_number,
        chunk_metadata: Dict[str, Any] = None,
    ) -> List[HypotheticalQuestion]:
        import traceback

        try:
            model_prompt = self._create_question_prompt(chunk)
            response = self.llm.invoke(model_prompt)
            print("[LLM RESPONSE]:", response)

            questions = self._parse_questions(
                response.content,
                chunk,
                chunk_id,
                source_document,
                page_number,
                chunk_metadata,
            )
            print("Questions generated successfully")
            return questions
        except Exception as e:
            tb = traceback.format_exc()
            print(f"exception occured: {e}\n{tb}")
            fallback_questions = self._create_fallback_questions(
                chunk, chunk_id, source_document, page_number, chunk_metadata
            )
            return fallback_questions
