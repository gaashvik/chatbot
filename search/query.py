from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from huggingface_hub import InferenceClient
from typing import List, Any
import config


class RAGSearch:
    def __init__(self):
        self.qdrant = QdrantClient(
            url="https://245be38a-1058-482a-9639-4b4ddd802aec.us-east-1-1.aws.cloud.qdrant.io",
            api_key=config.QDRANT_API_KEY,
        )
        self.embedding_model_name = config.EMBEDDING_MODEL_NAME
        self.embedder = InferenceClient(
            provider="hf-inference",
            model=self.embedding_model_name,
            token=config.HUGGING_FACE_API,
        )
        self.embedding_dimension = 1024

    def _create_embedding(self, query: str) -> List[float]:
        try:
            result = self.embedder.feature_extraction(
                text=query, normalize=True, truncate=True
            )
            return result
        except Exception as e:
            print("error creating embedding:", e)
            return [0.0] * self.embedding_dimension

    def _unpack_context(self, search_result: List[ScoredPoint]) -> str:
        context = ""
        for i, item in enumerate(search_result):
            context += f"""CONTEXT_INFO #{i+1} \n\n
1.Page Number:{item.payload.get('page_number','no page number found')}\n
2.Source Document:{item.payload.get('page_number','no source document found ')}\n
3.Content:\n
{item.payload.get('chunk_content','no context found')}\n\n"""
        return context

    def _search(self, query: str) -> str:
        # get embedding

        search_vector = self._create_embedding(query)

        result = self.qdrant.search(
            collection_name="rag_pdf_chunks", query_vector=search_vector, limit=5
        )

        context = self._unpack_context(result)

        return context
