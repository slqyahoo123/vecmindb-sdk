"""VecminDB VectorStore integration for LlamaIndex."""

from typing import Any, List, Optional
import uuid

try:
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.schema import TextNode, BaseNode
    LLAMAINDEX_INSTALLED = True
except ImportError:
    LLAMAINDEX_INSTALLED = False
    class BasePydanticVectorStore: pass
    class VectorStoreQuery: pass
    class VectorStoreQueryResult: pass
    class TextNode: pass
    class BaseNode: pass

from vecmindb.client import VecminClient
from vecmindb.exceptions import VecminError


class VecminDBVectorStore(BasePydanticVectorStore):
    """VecminDB VectorStore integration for LlamaIndex.

    This adapter bridges the LlamaIndex VectorStore interface with the
    commercial-grade VecminDB Python SDK.

    Args:
        client: An initialised :class:`VecminClient` instance.
        collection_name: Name of the VecminDB collection to use.
        dim: Vector dimensionality for the collection.
    """

    stores_text: bool = True
    flat_metadata: bool = False

    def __init__(
        self,
        client: VecminClient,
        collection_name: str = "llamaindex_memory",
        dim: int = 1536,
        **kwargs: Any,
    ) -> None:
        if not LLAMAINDEX_INSTALLED:
            raise ImportError("LlamaIndex is not installed. Please install it with `pip install llama-index-core`.")

        super().__init__(**kwargs)
        # Avoid Pydantic validation issues by setting private
        self._vecmin_client = client
        self.collection_name = collection_name

        try:
            self._vecmin_client.ensure_collection(self.collection_name, dimension=dim)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to ensure collection '{self.collection_name}': {e}")

    @property
    def client(self) -> Any:
        """Return the underlying VecminDB client."""
        return self._vecmin_client

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List of LlamaIndex BaseNode objects with embeddings.

        Returns:
            List of inserted node identifiers.
        """
        ids: List[str] = []
        for node in nodes:
            meta = node.metadata or {}
            meta["text"] = {"String": node.get_content()}

            doc_id = node.node_id or str(uuid.uuid4())
            emb = node.get_embedding()
            if not emb:
                continue

            try:
                res_id = self._vecmin_client.insert(
                    self.collection_name,
                    vector=emb,
                    id=doc_id,
                    metadata=meta,
                )
                ids.append(res_id)
            except VecminError as e:
                import warnings
                warnings.warn(f"Failed to add node {doc_id}: {e}")
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using ref_doc_id.

        Args:
            ref_doc_id: Document reference identifier.
        """
        try:
            self._vecmin_client.delete_vector(self.collection_name, ref_doc_id)
        except VecminError:
            pass  # Silently ignore – vector may not exist

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query: LlamaIndex VectorStoreQuery object.

        Returns:
            VectorStoreQueryResult with matching nodes.
        """
        try:
            response = self._vecmin_client.search(
                self.collection_name,
                query=query.query_embedding,
                top_k=query.similarity_top_k,
            )
            nodes: List[BaseNode] = []
            similarities: List[float] = []
            ids: List[str] = []
            for hit in response.results:
                meta = hit.metadata or {}

                text_obj = meta.get("text", {})
                text = text_obj.get("String", "") if isinstance(text_obj, dict) else str(text_obj)

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

                vec_id = hit.id or str(uuid.uuid4())
                node = TextNode(text=text, id_=vec_id, metadata=clean_meta)

                nodes.append(node)
                similarities.append(hit.score)
                ids.append(vec_id)

            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
        except VecminError as e:
            import warnings
            warnings.warn(f"Similarity search failed: {e}")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
