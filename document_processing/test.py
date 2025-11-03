from document_processing.index_builder_csv import (
    unifiedDocumentIndexBuilder,
    HypeEmbeddingSystem,
)

import config
from pathlib import Path
import boto3

if __name__ == "__main__":
    try:

        builder = unifiedDocumentIndexBuilder(
            config.EMBEDDING_MODEL_NAME, config.bedrock_client
        )

        success = builder.build_and_upload_index(
            document_dir=Path("/Users/shubhkamra/projects/chatbot/files")
        )

        if not success:
            print("failed")

    except Exception as e:
        print("error occurred while gloabally trying to create index,", e)
