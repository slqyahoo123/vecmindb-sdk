"""VecminDB Synchronous Python SDK Client.

Provides :class:`VecminClient` – a synchronous HTTP client that mirrors the
full API surface of :class:`AsyncVecminClient` using ``httpx``'s synchronous
transport.

Usage::

    with VecminClient("http://localhost:5520", api_key="xxx") as client:
        client.create_collection("docs", dimension=1536)
        results = client.search("docs", query=[0.1, 0.2, ...], top_k=10)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from .auth import AuthManager
from .exceptions import (
    AuthenticationError,
    ConnectionError,
    TimeoutError,
    VecminError,
    exception_from_status,
)
from .models import (
    BatchCreateVectorsRequest,
    BatchDeleteVectorsRequest,
    BatchInsertRequest,
    ClusterJoinRequest,
    ClusterLoginRequest,
    ClusterLoginResponse,
    ClusterNodeInfo,
    ClusterPromoteRequest,
    ClusterStatus,
    CollectionInfo,
    CollectionStats,
    CreateCollectionRequest,
    CreateVectorRequest,
    GlobalStats,
    HealthStatus,
    IndexInfo,
    InsertRequest,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SnapshotInfo,
    SubsystemStatus,
    VecminResponse,
)
from .retry import RetryConfig, retry_sync

logger = logging.getLogger("vecmindb.client")


class VecminClient:
    """Synchronous client for the VecminDB REST API.

    This client provides the same method signatures as
    :class:`AsyncVecminClient` but executes requests synchronously using
    ``httpx.Client``.

    Args:
        base_url: Root URL of the VecminDB server.
        api_key: Long-lived API key for authentication.
        jwt_token: Pre-obtained JWT bearer token.
        admin_password: Admin password – required for JWT auto-refresh.
        connect_timeout: TCP connection timeout in seconds.
        read_timeout: Socket read timeout in seconds.
        write_timeout: Socket write timeout in seconds.
        max_connections: Maximum connections in the pool.
        max_keepalive_connections: Maximum idle keep-alive connections.
        retry_config: Retry policy.  Defaults to 3 retries, 0.5 s back-off.

    Example::

        with VecminClient("http://localhost:5520", api_key="key") as c:
            collections = c.list_collections()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5520",
        *,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        admin_password: Optional[str] = None,
        agent_id: Optional[str] = None,
        sovereignty_token: Optional[str] = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        write_timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_url = f"{self._base_url}/api/v1"
        self._retry_config = retry_config or RetryConfig()
        self.agent_id = agent_id or "default_agent"
        self.sovereignty_token = sovereignty_token or "system"
        self._auth = AuthManager(
            api_key=api_key,
            jwt_token=jwt_token,
            admin_password=admin_password,
        )
        self._auth.set_login_fn(self._login_internal)

        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=connect_timeout,
        )
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            limits=limits,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VecminClient":  # noqa: D105
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D105
        self.close()

    def close(self) -> None:
        """Gracefully close the underlying HTTP connection pool."""
        self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(
        self,
        *,
        agent_id: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        effective_agent_id = agent_id or self.agent_id
        effective_model_id = model_id or self.sovereignty_token
        headers.update(self._auth.auth_headers())
        headers.update(self._auth.extra_headers(agent_id=effective_agent_id, model_id=effective_model_id, request_id=request_id))
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        **header_kwargs,
    ) -> Any:
        headers = self._build_headers(**header_kwargs)

        def _do_request() -> Any:
            try:
                response = self._client.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                    headers=headers,
                )
            except httpx.ConnectError as exc:
                raise ConnectionError(f"Cannot connect to {self._base_url}: {exc}") from exc
            except httpx.TimeoutException as exc:
                raise TimeoutError(f"Request to {path} timed out: {exc}") from exc
            except httpx.HTTPError as exc:
                raise ConnectionError(f"HTTP error on {path}: {exc}") from exc

            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"code": response.status_code, "message": "Success", "data": response.text}

            try:
                body = response.json()
                msg = body.get("message", response.text)
                code = body.get("code", response.status_code)
            except json.JSONDecodeError:
                msg = response.text
                code = response.status_code

            raise exception_from_status(response.status_code, message=msg, code=code)

        return retry_sync(_do_request, self._retry_config)

    def _api_get(self, path: str, *, params: Optional[Dict[str, Any]] = None, **kw) -> Any:
        return self._request("GET", f"{self._api_url}{path}", params=params, **kw)

    def _api_post(self, path: str, body: Optional[Any] = None, **kw) -> Any:
        return self._request("POST", f"{self._api_url}{path}", json_body=body, **kw)

    def _api_delete(self, path: str, **kw) -> Any:
        return self._request("DELETE", f"{self._api_url}{path}", **kw)

    # ------------------------------------------------------------------
    # JWT login (internal)
    # ------------------------------------------------------------------

    def _login_internal(self, password: str) -> tuple[str, int]:
        payload = ClusterLoginRequest(password=password)
        resp = self._client.post(
            f"{self._api_url}/cluster/login",
            json=payload.model_dump(),
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            raise AuthenticationError(f"Login failed: {resp.text}")
        data = resp.json()
        login = ClusterLoginResponse(**data.get("data", data))
        return login.token, login.expires_in

    # ==================================================================
    # Health & System
    # ==================================================================

    def health_check(self, **kw) -> HealthStatus:
        """Check server health (``GET /api/v1/health``)."""
        data = self._api_get("/health", **kw)
        return HealthStatus(**data.get("data", data))

    def status_check(self, **kw) -> SubsystemStatus:
        """Retrieve detailed subsystem status (``GET /api/v1/status``)."""
        data = self._api_get("/status", **kw)
        return SubsystemStatus(**data.get("data", data))

    def liveness(self) -> bool:
        """Kubernetes liveness probe (``GET /healthz/live``)."""
        try:
            resp = self._client.get(f"{self._base_url}/healthz/live")
            return resp.status_code == 200
        except Exception:
            return False

    def readiness(self) -> bool:
        """Kubernetes readiness probe (``GET /healthz/ready``)."""
        try:
            resp = self._client.get(f"{self._base_url}/healthz/ready")
            return resp.status_code == 200
        except Exception:
            return False

    def metrics(self) -> str:
        """Fetch Prometheus metrics (``GET /metrics``)."""
        data = self._request("GET", "/metrics")
        return data.get("data", str(data))

    # ==================================================================
    # Collection Management
    # ==================================================================

    def create_collection(
        self,
        name: str,
        *,
        dimension: int = 1536,
        metric_type: str = "Cosine",
        index_type: str = "HNSW",
        index_params: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> CollectionInfo:
        """Create a new collection (``POST /api/v1/collections``)."""
        payload = CreateCollectionRequest(
            name=name, dimension=dimension, metric_type=metric_type,
            index_type=index_type, index_params=index_params,
        )
        data = self._api_post("/collections", payload.model_dump(exclude_none=True), **kw)
        return CollectionInfo(**data.get("data", data))

    def list_collections(self, **kw) -> List[CollectionInfo]:
        """List all collections (``GET /api/v1/collections``)."""
        data = self._api_get("/collections", **kw)
        items = data.get("data", data)
        if isinstance(items, list):
            return [CollectionInfo(**c) for c in items]
        return [CollectionInfo(**items)]

    def get_collection(self, name: str, **kw) -> CollectionInfo:
        """Get collection details (``GET /api/v1/collections/{name}``)."""
        data = self._api_get(f"/collections/{name}", **kw)
        return CollectionInfo(**data.get("data", data))

    def delete_collection(self, name: str, **kw) -> VecminResponse:
        """Delete a collection (``DELETE /api/v1/collections/{name}``)."""
        data = self._api_delete(f"/collections/{name}", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    def get_collection_stats(self, name: str, **kw) -> CollectionStats:
        """Get collection statistics (``GET /api/v1/collections/{name}/stats``)."""
        data = self._api_get(f"/collections/{name}/stats", **kw)
        return CollectionStats(**data.get("data", data))

    def rebuild_collection_index(self, name: str, **kw) -> VecminResponse:
        """Rebuild the index for a collection (``POST /api/v1/collections/{name}/rebuild``)."""
        data = self._api_post(f"/collections/{name}/rebuild", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    # ==================================================================
    # Collection-level Vector Operations
    # ==================================================================

    def insert(
        self,
        collection: str,
        *,
        vector: List[float],
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> str:
        """Insert a single vector into a collection.

        Args:
            collection: Target collection name.
            vector: Embedding values.
            id: Optional client-assigned vector identifier.
            metadata: Optional key-value metadata.

        Returns:
            The vector identifier.
        """
        payload = InsertRequest(id=id, values=vector, metadata=metadata)
        data = self._api_post(
            f"/collections/{collection}/vectors",
            payload.model_dump(exclude_none=True),
            **kw,
        )
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("id", result.get("data", str(result)))
        return str(result)

    def batch_insert(
        self,
        collection: str,
        *,
        vectors: List[Dict[str, Any]],
        **kw,
    ) -> List[str]:
        """Batch insert vectors into a collection (``POST /api/v1/collections/{name}/batch``)."""
        items = [InsertRequest(**v) for v in vectors]
        payload = BatchInsertRequest(vectors=items)
        data = self._api_post(
            f"/collections/{collection}/batch",
            payload.model_dump(exclude_none=True),
            **kw,
        )
        result = data.get("data", data)
        if isinstance(result, list):
            return [r.get("id", str(r)) if isinstance(r, dict) else str(r) for r in result]
        return [str(result)]

    def search(
        self,
        collection: str,
        *,
        query: List[float],
        top_k: int = 10,
        ef_search: int = 50,
        metric: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> SearchResponse:
        """Search a collection for nearest neighbours (``POST /api/v1/collections/{name}/search``)."""
        payload = SearchRequest(query=query, k=top_k, ef_search=ef_search, metric=metric, filter=filter)
        data = self._api_post(
            f"/collections/{collection}/search",
            payload.model_dump(exclude_none=True),
            **kw,
        )
        raw = data.get("data", data)
        if isinstance(raw, list):
            return SearchResponse(results=[SearchHit(**h) for h in raw if isinstance(h, dict)])
        if isinstance(raw, dict):
            hits = raw.get("results", raw.get("hits", []))
            return SearchResponse(results=[SearchHit(**h) for h in hits], total=raw.get("total", len(hits)))
        return SearchResponse()

    def get_vector(self, collection: str, id: str, **kw) -> Dict[str, Any]:
        """Retrieve a vector by ID (``GET /api/v1/collections/{name}/vectors/{id}``)."""
        data = self._api_get(f"/collections/{collection}/vectors/{id}", **kw)
        return data.get("data", data)

    def delete_vector(self, collection: str, id: str, **kw) -> VecminResponse:
        """Delete a vector by ID (``DELETE /api/v1/collections/{name}/vectors/{id}``)."""
        data = self._api_delete(f"/collections/{collection}/vectors/{id}", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    # ==================================================================
    # Global Vector Operations
    # ==================================================================

    def create_vector(
        self,
        *,
        values: List[float],
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        **kw,
    ) -> str:
        """Create a vector in the global namespace (``POST /api/v1/vectors``)."""
        payload = CreateVectorRequest(id=id, values=values, metadata=metadata, collection=collection)
        data = self._api_post("/vectors", payload.model_dump(exclude_none=True), **kw)
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("id", str(result))
        return str(result)

    def batch_create_vectors(
        self,
        *,
        vectors: List[Dict[str, Any]],
        **kw,
    ) -> List[str]:
        """Batch create vectors (``POST /api/v1/vectors/batch``)."""
        items = [CreateVectorRequest(**v) for v in vectors]
        payload = BatchCreateVectorsRequest(vectors=items)
        data = self._api_post("/vectors/batch", payload.model_dump(exclude_none=True), **kw)
        result = data.get("data", data)
        if isinstance(result, list):
            return [r.get("id", str(r)) if isinstance(r, dict) else str(r) for r in result]
        return [str(result)]

    def get_global_vector(self, id: str, **kw) -> Dict[str, Any]:
        """Get a global vector by ID (``GET /api/v1/vectors/{id}``)."""
        data = self._api_get(f"/vectors/{id}", **kw)
        return data.get("data", data)

    def delete_global_vector(self, id: str, **kw) -> VecminResponse:
        """Delete a global vector (``DELETE /api/v1/vectors/{id}``)."""
        data = self._api_delete(f"/vectors/{id}", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    def batch_delete_vectors(self, *, ids: List[str], **kw) -> VecminResponse:
        """Batch delete vectors (``POST /api/v1/vectors/batch/delete``)."""
        payload = BatchDeleteVectorsRequest(ids=ids)
        data = self._api_post("/vectors/batch/delete", payload.model_dump(), **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    def search_vectors(
        self,
        *,
        query: List[float],
        top_k: int = 10,
        ef_search: int = 50,
        metric: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> SearchResponse:
        """Global vector search (``POST /api/v1/vectors/search``)."""
        payload = SearchRequest(query=query, k=top_k, ef_search=ef_search, metric=metric, filter=filter)
        data = self._api_post("/vectors/search", payload.model_dump(exclude_none=True), **kw)
        raw = data.get("data", data)
        if isinstance(raw, list):
            return SearchResponse(results=[SearchHit(**h) for h in raw if isinstance(h, dict)])
        if isinstance(raw, dict):
            hits = raw.get("results", [])
            return SearchResponse(results=[SearchHit(**h) for h in hits], total=raw.get("total", len(hits)))
        return SearchResponse()

    def count_vectors(self, **kw) -> int:
        """Count all vectors (``GET /api/v1/vectors/count``)."""
        data = self._api_get("/vectors/count", **kw)
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("count", 0)
        return int(result) if result else 0

    def rebuild_global_index(self, **kw) -> VecminResponse:
        """Rebuild global index (``POST /api/v1/vectors/rebuild-index``)."""
        data = self._api_post("/vectors/rebuild-index", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    # ==================================================================
    # Index Management
    # ==================================================================

    def list_indexes(self, **kw) -> List[IndexInfo]:
        """List all indexes (``GET /index``)."""
        data = self._request("GET", "/index", **kw)
        items = data.get("data", data)
        if isinstance(items, list):
            return [IndexInfo(**idx) for idx in items]
        return [IndexInfo(**items)] if isinstance(items, dict) else []

    def get_index_info(self, collection_name: str, **kw) -> IndexInfo:
        """Get index information (``GET /index/{collection_name}``)."""
        data = self._request("GET", f"/index/{collection_name}", **kw)
        return IndexInfo(**data.get("data", data))

    def rebuild_index(self, collection_name: str, **kw) -> VecminResponse:
        """Rebuild an index (``POST /index/{collection_name}/rebuild``)."""
        data = self._request("POST", f"/index/{collection_name}/rebuild", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    def optimize_index(self, collection_name: str, **kw) -> VecminResponse:
        """Optimize an index (``POST /index/{collection_name}/optimize``)."""
        data = self._request("POST", f"/index/{collection_name}/optimize", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    def get_shadow_trajectory(self, collection_name: str, **kw) -> List[Dict[str, Any]]:
        """Get shadow-index Nash trajectory (``GET /index/{collection_name}/shadow/trajectory``)."""
        data = self._request("GET", f"/index/{collection_name}/shadow/trajectory", **kw)
        items = data.get("data", data)
        if isinstance(items, list):
            return items
        return [items] if isinstance(items, dict) else []

    def promote_shadow(self, collection_name: str, **kw) -> VecminResponse:
        """Promote shadow index to primary (``POST /index/{collection_name}/shadow/promote``)."""
        data = self._request("POST", f"/index/{collection_name}/shadow/promote", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    # ==================================================================
    # Stats
    # ==================================================================

    def global_stats(self, **kw) -> GlobalStats:
        """Fetch global database statistics (``GET /stats``)."""
        data = self._request("GET", "/stats", **kw)
        return GlobalStats(**data.get("data", data))

    def index_stats(self, **kw) -> Dict[str, Any]:
        """Fetch index statistics (``GET /stats/index``)."""
        data = self._request("GET", "/stats/index", **kw)
        return data.get("data", data)

    def performance_stats(self, **kw) -> Dict[str, Any]:
        """Fetch performance statistics (``GET /stats/performance``)."""
        data = self._request("GET", "/stats/performance", **kw)
        return data.get("data", data)

    # ==================================================================
    # Cluster Management
    # ==================================================================

    def cluster_login(self, password: str, **kw) -> ClusterLoginResponse:
        """Authenticate and obtain a JWT (``POST /api/v1/cluster/login``)."""
        payload = ClusterLoginRequest(password=password)
        data = self._api_post("/cluster/login", payload.model_dump(), **kw)
        result = data.get("data", data)
        login_resp = ClusterLoginResponse(**result)
        self._auth.update_jwt(login_resp.token, login_resp.expires_in)
        return login_resp

    def cluster_join(self, node_id: str, addr: str, **kw) -> VecminResponse:
        """Join a cluster (``POST /api/v1/cluster/join``)."""
        payload = ClusterJoinRequest(node_id=node_id, addr=addr)
        data = self._api_post("/cluster/join", payload.model_dump(), **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    def cluster_promote(self, node_id: str, **kw) -> VecminResponse:
        """Promote a node (``POST /api/v1/cluster/promote``)."""
        payload = ClusterPromoteRequest(node_id=node_id)
        data = self._api_post("/cluster/promote", payload.model_dump(), **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    def list_nodes(self, **kw) -> List[ClusterNodeInfo]:
        """List cluster nodes."""
        status = self.cluster_status(**kw)
        return status.nodes

    def cluster_status(self, **kw) -> ClusterStatus:
        """Get cluster status (``GET /api/v1/cluster/status``)."""
        data = self._api_get("/cluster/status", **kw)
        return ClusterStatus(**data.get("data", data))

    def add_node(self, node_id: str, addr: str, **kw) -> VecminResponse:
        """Add a node to the cluster (``POST /api/v1/cluster/nodes``)."""
        data = self._api_post("/cluster/nodes", {"node_id": node_id, "addr": addr}, **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    def remove_node(self, node_id: str, **kw) -> VecminResponse:
        """Remove a node from the cluster (``DELETE /api/v1/cluster/nodes``)."""
        data = self._api_delete(f"/cluster/nodes?node_id={node_id}", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    def get_join_info(self, **kw) -> Dict[str, Any]:
        """Get cluster join information (``GET /api/v1/cluster/join_info``)."""
        data = self._api_get("/cluster/join_info", **kw)
        return data.get("data", data)

    # ==================================================================
    # MCP Integration
    # ==================================================================

    def mcp_store_memory(
        self,
        content: str,
        *,
        agent_id: Optional[str] = None,
        sovereignty_token: Optional[str] = None,
        model_id: Optional[str] = None,
        source: str = "sdk",
        **kw,
    ) -> str:
        """Store a memory via the MCP ``store_memory`` tool."""
        effective_agent_id = agent_id or self.agent_id
        effective_token = sovereignty_token or self.sovereignty_token
        effective_model_id = model_id or effective_token
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "store_memory",
                "arguments": {
                    "agent_id": effective_agent_id,
                    "text": content,
                    "source": source,
                    "sovereignty_token": effective_token,
                    "model_id": effective_model_id,
                },
            },
            "id": 1,
        }
        data = self._api_post("/mcp/message", payload, agent_id=effective_agent_id, model_id=effective_model_id, **kw)
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("content", result.get("result", str(result)))
        return str(result)

    def mcp_search_memory(
        self,
        query: str,
        *,
        agent_id: Optional[str] = None,
        sovereignty_token: Optional[str] = None,
        model_id: Optional[str] = None,
        top_k: int = 5,
        **kw,
    ) -> str:
        """Search memories via the MCP ``search_memory`` tool."""
        effective_agent_id = agent_id or self.agent_id
        effective_token = sovereignty_token or self.sovereignty_token
        effective_model_id = model_id or effective_token
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "search_memory",
                "arguments": {
                    "agent_id": effective_agent_id,
                    "query": query,
                    "top_k": top_k,
                    "sovereignty_token": effective_token,
                    "model_id": effective_model_id,
                },
            },
            "id": 2,
        }
        data = self._api_post("/mcp/message", payload, agent_id=effective_agent_id, model_id=effective_model_id, **kw)
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("content", result.get("result", str(result)))
        return str(result)

    def mount_memory(
        self,
        domain: Optional[str] = None,
        *,
        name: Optional[str] = None,
        sovereignty_token: Optional[str] = None,
        mode: str = "evolutionary",
        model_id: Optional[str] = None,
    ) -> "VecminMemorySpace":
        """Mount a cognitive memory space for agent operation.

        Args:
            domain: Alias for name/collection identifier.
            name: Unified cognitive space identifier.
            sovereignty_token: Sovereign domain access token.
            mode: Operating mode (e.g., 'evolutionary').
            model_id: Embedding feature extraction model targeting.

        Returns:
            VecminMemorySpace instance.
        """
        from .memory import AgentMemoryManager
        resolved_name = domain or name or "default"
        resolved_token = sovereignty_token or self.sovereignty_token
        manager = AgentMemoryManager(
            client=self,
            agent_id=self.agent_id,
            sovereignty_token=resolved_token,
            model_id=model_id,
        )
        return manager.mount_memory(domain=resolved_name, mode=mode)

    # ==================================================================
    # Convenience: ensure_collection (backward compat)
    # ==================================================================

    def ensure_collection(
        self,
        name: str,
        *,
        dimension: int = 1536,
        metric_type: str = "Cosine",
        index_type: str = "HNSW",
        index_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Ensure a collection exists, creating it if necessary.

        Args:
            name: Collection name.
            dimension: Vector dimensionality.
            metric_type: Distance metric.
            index_type: Index algorithm.
            index_params: Algorithm parameters.

        Returns:
            ``True`` if the collection exists or was created.
        """
        try:
            self.get_collection(name)
            return True
        except Exception:
            pass
        try:
            self.create_collection(
                name, dimension=dimension, metric_type=metric_type,
                index_type=index_type, index_params=index_params,
            )
            return True
        except Exception:
            return False
