from qdrant_client import QdrantClient, models
from qdrant_client.models import ScoredPoint
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from huggingface_hub import InferenceClient
from rank_bm25 import BM25Okapi
from typing import List, Any, Dict
import numpy as np
import config
from pydantic import BaseModel
from rapidfuzz import fuzz
from langchain_aws import ChatBedrockConverse
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForMaskedLM
import json
import torch


@dataclass
class CandidateItem:
    original_score: float
    payload: Dict[str, Any]
    rerank_position: int
    rerank_score: float = 0
    mmr_selected: bool = False


class CrossEncoderScores(BaseModel):
    """Output format for cross encoder scored for a list of candidate text, returns a list of scores"""

    scores: List[str]


class SmartSearch:
    def __init__(self, embedding_model_name: str, bedrock_client=None):
        self.qdrant = QdrantClient(
            url="https://245be38a-1058-482a-9639-4b4ddd802aec.us-east-1-1.aws.cloud.qdrant.io",
            api_key=config.QDRANT_API_KEY,
        )
        self.embedding_model_name = embedding_model_name
        self.embedder = bedrock_client
        self.mmr_lambda = 0.85
        self.embedding_dimension = 1024
        self.llm = ChatBedrockConverse(
            model_id="anthropic.claude-3-haiku-20240307-v1:0", client=bedrock_client
        )
        self.splade_model_id = "naver/splade-cocondenser-ensembledistil"
        self.splade_tokenizer = AutoTokenizer.from_pretrained(self.splade_model_id)
        self.splade_model = AutoModelForMaskedLM.from_pretrained(self.splade_model_id)
        self.llm = self.llm.with_structured_output(CandidateItem)
        print(self.embedder)

    def _unpack_context(self, search_result: List[CandidateItem]) -> str:
        """
        Converts retrieved context results into well‑formatted markdown text
        for LLM consumption and human readability.
        """

        markdown_output = []
        for i, item in enumerate(search_result):
            page = item.payload.get("page_number", "no page number found")
            source = item.payload.get("source_document", "no source document found")
            content = item.payload.get("chunk_content", "no context found")

            context_block = f"""
        ---

        ### 🧩 Context #{i+1}

        **Page Number:** {page}  
        **Source Document:** {source}  

        **Content:**
        > {content.strip()}

        ---
        """
            markdown_output.append(context_block.strip())

        return "\n\n".join(markdown_output)

    def _create_dense_embedding(self, text: str) -> List[float]:
        try:
            body = {
                "inputText": text,
                "dimensions": self.embedding_dimension,
                "normalize": True,
            }

            response = self.embedder.invoke_model(
                modelId=self.embedding_model_name,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            response_body = json.loads(response["body"].read())

            print("[Genertaed embedding]:", response_body)
            print("\n")
            print("\n")
            return response_body["embedding"]
        except Exception as e:
            print("Exception occured while embedding:", e)
            return [0.0] * self.embedding_dimension

    def _create_sparse_embedding(self, text: str):
        tokens = self.splade_tokenizer(text, return_tensors="pt")
        output = self.splade_model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()
        indicies = vec.nonzero().numpy().flatten().tolist()
        values = vec.detach().numpy()[indicies].tolist()

        return indicies, values

    def _search(self, query: str) -> str:

        dense_embedding = self._create_dense_embedding(query)
        sparse_embedding_indicies, sparse_embedding_values = (
            self._create_sparse_embedding(query)
        )

        result = self.qdrant.query_points(
            collection_name="test",
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_embedding_indicies,
                        values=sparse_embedding_values,
                    ),
                    using="sparseV",
                    limit=10,
                ),
                models.Prefetch(
                    query=dense_embedding,  # <-- dense vector
                    using="denseV",
                    limit=10,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )
        reranked_result = self._rerank_search_result(result.points, query)

        context = self._unpack_context(reranked_result)

        return context

    def _rerank_search_result(
        self, candidate_passages: List[ScoredPoint], query: str, top_k: int = 6
    ) -> List[CandidateItem]:

        # 1. calculate relavance score based on cross encoding

        relevance_score = self._calculate_relevance_score(candidate_passages, query)

        print("[RELEVANCE SCORE]", relevance_score)

        # 2. apply MMR

        result = self._apply_mmr_reranking(
            candidate_passages, query, relevance_score, top_k
        )
        print("[MMR RESULT]", result)

        result.sort(key=lambda x: x.rerank_score, reverse=True)

        return result[:top_k]

    def _apply_mmr_reranking(
        self,
        candidate_passages: List[ScoredPoint],
        query: str,
        relevance_scores: List[float],
        top_k: int = 6,
    ) -> List[CandidateItem]:

        if not candidate_passages or not relevance_scores:
            return [
                CandidateItem(original_score=candidate.score, payload=candidate.payload)
                for candidate in candidate_passages[:top_k]
            ]

        candidate_text = [
            item.payload.get("chunk_content") for item in candidate_passages
        ]
        similarity_matrix = self._calculate_similarity_matirx(candidate_text)

        selected_indices = []
        remaining_indices = list(range(len(candidate_passages)))

        first_idx = max(remaining_indices, key=lambda i: relevance_scores[i])

        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:

                relevance = relevance_scores[idx]

                max_similarity = 0.0

                for selected_idx in selected_indices:
                    if idx < len(similarity_matrix) and selected_idx < len(
                        similarity_matrix[0]
                    ):
                        similarity = similarity_matrix[idx][selected_idx]
                        max_similarity = max(max_similarity, similarity)

                    mmr_score = (
                        self.mmr_lambda * relevance
                        - (1 - self.mmr_lambda) * max_similarity
                    )
                    mmr_scores.append((idx, mmr_score))

            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        reranked_results = []
        for rank, idx in enumerate(selected_indices):
            candidate = candidate_passages[idx]

            rerank_score = relevance_scores[idx]
            rerank_position = rank + 1
            mmr_selected = True

            result = CandidateItem(
                candidate.score,
                candidate.payload,
                rerank_position,
                rerank_score,
                mmr_selected,
            )

            reranked_results.append(result)
        return reranked_results

    def _calculate_similarity_matirx(
        self, candidate_text: List[str]
    ) -> List[List[float]]:

        n = len(candidate_text)

        try:
            # identity matrix
            mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    # normalize
                    similarity = (
                        fuzz.token_set_ratio(candidate_text[i], candidate_text[j])
                        / 100.0
                    )
                    mat[i][j] = similarity
                    mat[j][i] = similarity

            return mat
        except Exception as e:
            print("error calculating matrix")
            return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def _calculate_relevance_score(
        self, candidate_passages: List[ScoredPoint], query: str
    ) -> List[float]:

        component_scores = self._calculate_component_score(candidate_passages, query)

        final_score = self._combine_scores(query, candidate_passages, component_scores)

        return final_score

    def _calculate_component_score(
        self, candidate_passages: List[ScoredPoint], query: str
    ) -> List[float]:

        candidate_text = [
            candidate_passage.payload.get("chunk_content", "")
            for candidate_passage in candidate_passages
        ]

        # 1. calculate BM25 score (faster than TF-IDF)

        bm25_score = self._caclulate_bm25_score(query, candidate_text)

        # 2. calculate cross-encoder score

        cross_encoder_score = self._calculate_cross_encoder_score(query, candidate_text)

        return {"bm25_scores": bm25_score, "cross_encoder_scores": cross_encoder_score}

    def _combine_scores(
        self, query: str, candidates: List[ScoredPoint], component_scores: List[float]
    ) -> List[float]:

        bm25_scores = component_scores["bm25_scores"]
        cross_encoder_scores = component_scores["cross_encoder_scores"]

        final_scores = []

        for i, candidate in enumerate(candidates):
            scores = self._get_candidate_component_scores(
                candidate, bm25_scores, cross_encoder_scores, i
            )

            final_score = self._get_combined_score(scores)

            final_scores.append(final_score)

        return final_scores

    def _get_candidate_component_scores(
        self,
        candidate: ScoredPoint,
        bm25_scores: List[float],
        cross_encoder_scores: List[float],
        idx: int,
    ):
        original_score = candidate.score

        bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0

        base_score = (
            max(original_score, bm25_score) if original_score > 0 else bm25_score
        )

        cross_encoder_score = (
            cross_encoder_scores[idx] if len(cross_encoder_scores) else 0.0
        )
        return {
            "original_score": original_score,
            "base_score": base_score,
            "cross_score": cross_encoder_score,
        }

    def _get_combined_score(self, score):

        original_score = score["original_score"]
        base_score = score["base_score"]
        cross_score = score["cross_score"]

        # Perfect scores (>95%) maintain 95% of original quality
        if original_score > 0.95:
            return 0.95 * original_score + 0.05 * cross_score
        # Excellent scores (>90%) maintain 90% quality
        elif original_score > 0.9:
            return 0.9 * original_score + 0.10 * cross_score
        # High-quality scores (>80%) get 85% weight preservation
        elif original_score > 0.8:
            return 0.85 * original_score + +0.15 * cross_score
        else:
            return 0.6 * base_score + 0.35 * cross_score

    def _tokenize(self, query: str) -> List[str]:
        return query.lower().split()

    def _caclulate_bm25_score(
        self, query: str, candidate_text: List[str]
    ) -> List[float]:

        tokenized_text = [self._tokenize(text) for text in candidate_text]
        bm25 = BM25Okapi(tokenized_text)
        tokenized_query = self._tokenize(query)

        scores = np.asanyarray(bm25.get_scores(tokenized_query), dtype=float)

        if len(scores) == 0:
            return [0.0] * len(candidate_text)

        max_score = float(scores.max())

        if max_score <= 1e-12:
            return [0.0] * len(candidate_text)

        normalized_scores = (scores / max_score).tolist()
        return normalized_scores

    def _get_batch_result(self, query: str, candidate_text: List[str]) -> List[float]:
        prompt = (
            "Rate how well each text passage answers this query on a scale of 0.0-1.0.",
            f"query:{query}\n",
            "Rate Each Passage:",
        )
        for i, text in enumerate(candidate_text):
            truncated_text = text[:200] + "..." if len(text) > 200 else text
            prompt += f"\nPassage {i+1}: {truncated_text}"

    def _calculate_cross_encoder_score(
        self, query: str, candidate_text: List[str]
    ) -> List[float]:

        try:
            batch_eval_prompt = self._get_batch_eval_promopt(query, candidate_text)

            result = self.llm.invoke(
                input=batch_eval_prompt, response_format=CrossEncoderScores
            )

            while len(result) < len(candidate_text):
                result.append(0.5)

            return result

        except Exception as e:
            return [0.5] * len(candidate_text)
