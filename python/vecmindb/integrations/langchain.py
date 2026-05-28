"""VecminDB VectorStore integration for LangChain."""

from typing import Any, Iterable, List, Optional
import uuid

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    LANGCHAIN_INSTALLED = True
except ImportError:
    LANGCHAIN_INSTALLED = False
    class VectorStore: pass
    class Embeddings: pass
    class Document: pass

from vecmindb.client import VecminClient
from vecmindb.exceptions import VecminError


class VecminDBVectorStore(VectorStore):
    """VecminDB VectorStore integration for LangChain.

    This adapter bridges the LangChain VectorStore interface with the
    commercial-grade VecminDB Python SDK.

    Args:
        client: An initialised :class:`VecminClient` instance.
        embedding: A LangChain Embeddings implementation.
        collection_name: Name of the VecminDB collection to use.
        dim: Vector dimensionality for the collection.
    """

    def __init__(
        self,
        client: VecminClient,
        embedding: Embeddings,
        collection_name: str = "langchain_memory",
        dim: int = 1536,
        **kwargs: Any,
    ) -> None:
        if not LANGCHAIN_INSTALLED:
            raise ImportError("LangChain is not installed. Please install it with `pip install langchain-core`.")
        self.vecmin_client = client
        self.embedding = embedding
        self.collection_name = collection_name

        # Ensure collection exists
        try:
            self.vecmin_client.ensure_collection(self.collection_name, dimension=dim)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to ensure collection '{self.collection_name}': {e}")

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of text strings to embed and store.
            metadatas: Optional list of metadata dicts for each text.

        Returns:
            List of inserted vector identifiers.
        """
        texts = list(texts)
        if len(texts) == 0:
            return []

        embeddings = self.embedding.embed_documents(texts)
        ids: List[str] = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            meta["text"] = {"String": text}  # Conform to VecminDB metadata format

            doc_id = str(uuid.uuid4())
            try:
                res_id = self.vecmin_client.insert(
                    self.collection_name,
                    vector=embeddings[i],
                    id=doc_id,
                    metadata=meta,
                )
                ids.append(res_id)
            except VecminError as e:
                import warnings
                warnings.warn(f"Failed to insert vector for text '{text[:20]}...': {e}")

        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Query string to search for.
            k: Number of documents to return.

        Returns:
            List of LangChain Document objects.
        """
        emb = self.embedding.embed_query(query)
        try:
            response = self.vecmin_client.search(
                self.collection_name,
                query=emb,
                top_k=k,
            )
            docs: List[Document] = []
            for hit in response.results:
                meta = hit.metadata or {}
                # Extract text from VecminDB structured metadata
                text_obj = meta.get("text", {})
                text = text_obj.get("String", "") if isinstance(text_obj, dict) else str(text_obj)

                # Cleanup metadata to pass back to LangChain
                clean_meta: Dict[str, Any] = {}
                for mk, mv in meta.items():
                    if isinstance(mv, dict) and "String" in mv:
                        clean_meta[mk] = mv["String"]
                    elif isinstance(mv, dict) and "Float" in mv:
                        clean_meta[mk] = mv["Float"]
                    elif isinstance(mv, dict) and "Integer" in mv:
                        clean_meta[mk] = mv["Integer"]
                    else:
                        clean_meta[mk] = mv

                docs.append(Document(page_content=text, metadata=clean_meta))
            return docs
        except VecminError as e:
            import warnings
            warnings.warn(f"Similarity search failed: {e}")
            return []

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Optional[VecminClient] = None,
        collection_name: str = "langchain_memory",
        dim: int = 1536,
        **kwargs: Any,
    ) -> "VecminDBVectorStore":
        """Return VectorStore initialized from texts and embeddings.

        Args:
            texts: Texts to add.
            embedding: Embeddings implementation.
            metadatas: Optional metadata for each text.
            client: VecminClient instance (required).
            collection_name: Collection name.
            dim: Vector dimensionality.

        Returns:
            Initialised VecminDBVectorStore.

        Raises:
            ValueError: If *client* is not provided.
        """
        if client is None:
            raise ValueError("client (VecminClient) must be provided")

        store = cls(client, embedding, collection_name, dim, **kwargs)
        store.add_texts(texts, metadatas, **kwargs)
        return store
