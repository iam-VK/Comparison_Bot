MODEL = "llama3.1"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 60

QDRANT_KEY = "XSn6uiTT9W9N82v-2ykmiTiARTVOKCCgUvI2ch5ATQ9oZLsXSYxlzA"
# QDRANT_URL = "https://4d07c1f8-9269-410e-89aa-570117a9e30f.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = 'RAG_proto-2_GOT-MEDIUM'

print("Configs:")
print({
    "MODEL":MODEL,
    "CHUNK_SIZE":CHUNK_SIZE,
    "CHUNK_OVERLAP":CHUNK_OVERLAP,
    "QDRANT_URL":QDRANT_URL,
    "COLLECTION_NAME":COLLECTION_NAME,
})