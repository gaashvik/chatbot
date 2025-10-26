from document_processing.index_builder_csv import unifiedDocumentIndexBuilder
import config
from pathlib import Path

if __name__ == "__main__":
    try:

        builder = unifiedDocumentIndexBuilder(config.EMBEDDING_MODEL_NAME)

        success = builder.build_and_upload_index(
            document_dir=Path("/home/shubhk/aivar/generic_chatbot/files")
        )

        if not success:
            print("failed")
    except Exception as e:
        print("error occurred while gloabally trying to create index")
