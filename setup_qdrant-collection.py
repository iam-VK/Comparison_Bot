from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
from config import COLLECTION_NAME, QDRANT_URL

client = QdrantClient(url=QDRANT_URL)
vector_param = VectorParams(size=384, distance=Distance.DOT)
client.create_collection(collection_name = COLLECTION_NAME, vectors_config = vector_param)