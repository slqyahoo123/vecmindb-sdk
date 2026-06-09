"""VecminDB Async Python SDK Client.

Provides :class:`AsyncVecminClient` – a fully-featured async HTTP client for
the VecminDB REST API.  Supports connection pooling, automatic JWT refresh,
configurable retry with exponential back-off, and structured error handling.

Usage::

    async with AsyncVecminClient("http://localhost:5520", api_key="xxx") as client:
        await client.create_collection("docs", dimension=1536)
        results = await client.search("docs", query=[0.1, 0.2, ...], top_k=10)
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

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
from .retry import RetryConfig, retry_async

logger = logging.getLogger("vecmindb.async_client")


class AsyncVecminClient:
    """Asynchronous client for the VecminDB REST API.

    This client wraps every endpoint exposed by the VecminDB server and
    handles authentication, retries, and response parsing transparently.

    Args:
        base_url: Root URL of the VecminDB server (e.g. ``http://localhost:5520``).
        api_key: Long-lived API key for authentication.
        jwt_token: Pre-obtained JWT bearer token.
        admin_password: Admin password – required for JWT auto-refresh.
        connect_timeout: TCP connection timeout in seconds.
        read_timeout: Socket read timeout in seconds.
        write_timeout: Socket write timeout in seconds.
        max_connections: Maximum number of connections in the pool.
        max_keepalive_connections: Maximum number of idle keep-alive connections.
        retry_config: Retry policy configuration.  Defaults to 3 retries with
            0.5 s back-off factor.

    Example::

        async with AsyncVecminClient("http://localhost:5520", api_key="key") as c:
            collections = await c.list_collections()
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
        # Register the login function for JWT auto-refresh.
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
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            limits=limits,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncVecminClient":  # noqa: D105
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D105
        await self.close()

    async def close(self) -> None:
        """Gracefully close the underlying HTTP connection pool."""
        await self._client.aclose()

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

    async def _build_headers_async(self, **kwargs) -> Dict[str, str]:
        """Build headers, refreshing JWT asynchronously if needed."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        # Refresh JWT asynchronously
        jwt = await self._auth.async_maybe_refresh_jwt()
        if self._auth.api_key:
            headers["x-api-key"] = self._auth.api_key
        if jwt:
            headers["Authorization"] = f"Bearer {jwt}"
        effective_agent_id = kwargs.get("agent_id") or self.agent_id
        effective_model_id = kwargs.get("model_id") or self.sovereignty_token
        headers.update(self._auth.extra_headers(
            agent_id=effective_agent_id,
            model_id=effective_model_id,
            request_id=kwargs.get("request_id"),
        ))
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        **header_kwargs,
    ) -> Any:
        """Execute an HTTP request with retry and error handling.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.).
            path: URL path relative to the server root.
            json_body: Optional JSON-serialisable request body.
            params: Optional query-string parameters.
            **header_kwargs: Passed through to :meth:`_build_headers_async`.

        Returns:
            Parsed JSON response body on success.

        Raises:
            VecminError: On any API-level error.
            ConnectionError: On network failure.
            TimeoutError: On request timeout.
        """
        headers = await self._build_headers_async(**header_kwargs)

        async def _do_request() -> Any:
            try:
                response = await self._client.request(
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

            # Successful response
            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    # Some endpoints (e.g. /metrics) return plain text.
                    return {"code": response.status_code, "message": "Success", "data": response.text}

            # Error response – attempt to parse the standard envelope.
            try:
                body = response.json()
                msg = body.get("message", response.text)
                code = body.get("code", response.status_code)
            except json.JSONDecodeError:
                msg = response.text
                code = response.status_code

            raise exception_from_status(response.status_code, message=msg, code=code)

        return await retry_async(_do_request, self._retry_config)

    async def _api_get(self, path: str, *, params: Optional[Dict[str, Any]] = None, **kw) -> Any:
        return await self._request("GET", f"{self._api_url}{path}", params=params, **kw)

    async def _api_post(self, path: str, body: Optional[Any] = None, **kw) -> Any:
        return await self._request("POST", f"{self._api_url}{path}", json_body=body, **kw)

    async def _api_delete(self, path: str, **kw) -> Any:
        return await self._request("DELETE", f"{self._api_url}{path}", **kw)

    # ------------------------------------------------------------------
    # JWT login (internal)
    # ------------------------------------------------------------------

    async def _login_internal(self, password: str) -> tuple[str, int]:
        """Authenticate against the cluster login endpoint.

        This method is **not** meant for direct use – it is wired into the
        :class:`AuthManager` for automatic JWT refresh.  Users should call
        :meth:`cluster_login` instead.

        Args:
            password: Admin password.

        Returns:
            ``(token, expires_in)`` tuple.
        """
        # Use a direct request without auth headers to avoid circular refresh.
        payload = ClusterLoginRequest(password=password)
        resp = await self._client.post(
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

    async def health_check(self, **kw) -> HealthStatus:
        """Check server health (``GET /api/v1/health``).

        Returns:
            Server health status.
        """
        data = await self._api_get("/health", **kw)
        return HealthStatus(**data.get("data", data))

    async def status_check(self, **kw) -> SubsystemStatus:
        """Retrieve detailed subsystem status (``GET /api/v1/status``).

        Returns:
            Detailed subsystem status.
        """
        data = await self._api_get("/status", **kw)
        return SubsystemStatus(**data.get("data", data))

    async def liveness(self) -> bool:
        """Kubernetes liveness probe (``GET /healthz/live``).

        Returns:
            ``True`` if the server process is alive.
        """
        try:
            resp = await self._client.get(f"{self._base_url}/healthz/live")
            return resp.status_code == 200
        except Exception:
            return False

    async def readiness(self) -> bool:
        """Kubernetes readiness probe (``GET /healthz/ready``).

        Returns:
            ``True`` if the server is ready to accept traffic.
        """
        try:
            resp = await self._client.get(f"{self._base_url}/healthz/ready")
            return resp.status_code == 200
        except Exception:
            return False

    async def metrics(self) -> str:
        """Fetch Prometheus metrics (``GET /metrics``).

        Returns:
            Plain-text Prometheus exposition.
        """
        data = await self._request("GET", "/metrics")
        return data.get("data", str(data))

    # ==================================================================
    # Collection Management
    # ==================================================================

    async def create_collection(
        self,
        name: str,
        *,
        dimension: int = 1536,
        metric_type: str = "Cosine",
        index_type: str = "HNSW",
        index_params: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> CollectionInfo:
        """Create a new collection (``POST /api/v1/collections``).

        Args:
            name: Unique collection identifier.
            dimension: Vector dimensionality.
            metric_type: Distance metric.
            index_type: Index algorithm.
            index_params: Algorithm-specific parameters.

        Returns:
            Created collection metadata.
        """
        payload = CreateCollectionRequest(
            name=name,
            dimension=dimension,
            metric_type=metric_type,
            index_type=index_type,
            index_params=index_params,
        )
        data = await self._api_post("/collections", payload.model_dump(exclude_none=True), **kw)
        return CollectionInfo(**data.get("data", data))

    async def list_collections(self, **kw) -> List[CollectionInfo]:
        """List all collections (``GET /api/v1/collections``).

        Returns:
            List of collection metadata objects.
        """
        data = await self._api_get("/collections", **kw)
        items = data.get("data", data)
        if isinstance(items, list):
            return [CollectionInfo(**c) for c in items]
        return [CollectionInfo(**items)]

    async def get_collection(self, name: str, **kw) -> CollectionInfo:
        """Get collection details (``GET /api/v1/collections/{name}``).

        Args:
            name: Collection identifier.

        Returns:
            Collection metadata.
        """
        data = await self._api_get(f"/collections/{name}", **kw)
        return CollectionInfo(**data.get("data", data))

    async def delete_collection(self, name: str, **kw) -> VecminResponse:
        """Delete a collection (``DELETE /api/v1/collections/{name}``).

        Args:
            name: Collection identifier.

        Returns:
            Deletion confirmation.
        """
        data = await self._api_delete(f"/collections/{name}", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    async def get_collection_stats(self, name: str, **kw) -> CollectionStats:
        """Get collection statistics (``GET /api/v1/collections/{name}/stats``).

        Args:
            name: Collection identifier.

        Returns:
            Statistical summary.
        """
        data = await self._api_get(f"/collections/{name}/stats", **kw)
        return CollectionStats(**data.get("data", data))

    async def rebuild_collection_index(self, name: str, **kw) -> VecminResponse:
        """Rebuild the index for a collection (``POST /api/v1/collections/{name}/rebuild``).

        Args:
            name: Collection identifier.

        Returns:
            Operation acknowledgement.
        """
        data = await self._api_post(f"/collections/{name}/rebuild", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    # ==================================================================
    # Collection-level Vector Operations
    # ==================================================================

    async def insert(
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
        data = await self._api_post(
            f"/collections/{collection}/vectors",
            payload.model_dump(exclude_none=True),
            **kw,
        )
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("id", result.get("data", str(result)))
        return str(result)

    async def batch_insert(
        self,
        collection: str,
        *,
        vectors: List[Dict[str, Any]],
        **kw,
    ) -> List[str]:
        """Batch insert vectors into a collection (``POST /api/v1/collections/{name}/batch``).

        Args:
            collection: Target collection name.
            vectors: List of dicts with keys ``values``, ``id``, ``metadata``.

        Returns:
            List of inserted vector identifiers.
        """
        items = [InsertRequest(**v) for v in vectors]
        payload = BatchInsertRequest(vectors=items)
        data = await self._api_post(
            f"/collections/{collection}/batch",
            payload.model_dump(exclude_none=True),
            **kw,
        )
        result = data.get("data", data)
        if isinstance(result, list):
            return [r.get("id", str(r)) if isinstance(r, dict) else str(r) for r in result]
        return [str(result)]

    async def search(
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
        """Search a collection for nearest neighbours (``POST /api/v1/collections/{name}/search``).

        Args:
            collection: Collection to search.
            query: Query embedding.
            top_k: Number of results.
            ef_search: HNSW search-width parameter.
            metric: Override distance metric.
            filter: Metadata filter expression.

        Returns:
            Search results.
        """
        payload = SearchRequest(query=query, k=top_k, ef_search=ef_search, metric=metric, filter=filter)
        data = await self._api_post(
            f"/collections/{collection}/search",
            payload.model_dump(exclude_none=True),
            **kw,
        )
        raw = data.get("data", data)
        if isinstance(raw, list):
            return SearchResponse(results=[SearchHit(**h) if isinstance(h, dict) else SearchHit(id=str(h)) for h in raw])
        if isinstance(raw, dict):
            hits = raw.get("results", raw.get("hits", []))
            return SearchResponse(
                results=[SearchHit(**h) for h in hits],
                total=raw.get("total", len(hits)),
            )
        return SearchResponse()

    async def get_vector(self, collection: str, id: str, **kw) -> Dict[str, Any]:
        """Retrieve a vector by ID (``GET /api/v1/collections/{name}/vectors/{id}``).

        Args:
            collection: Collection name.
            id: Vector identifier.

        Returns:
            Vector data including values and metadata.
        """
        data = await self._api_get(f"/collections/{collection}/vectors/{id}", **kw)
        return data.get("data", data)

    async def delete_vector(self, collection: str, id: str, **kw) -> VecminResponse:
        """Delete a vector by ID (``DELETE /api/v1/collections/{name}/vectors/{id}``).

        Args:
            collection: Collection name.
            id: Vector identifier.

        Returns:
            Deletion confirmation.
        """
        data = await self._api_delete(f"/collections/{collection}/vectors/{id}", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    # ==================================================================
    # Global Vector Operations
    # ==================================================================

    async def create_vector(
        self,
        *,
        values: List[float],
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        **kw,
    ) -> str:
        """Create a vector in the global namespace (``POST /api/v1/vectors``).

        Args:
            values: Embedding values.
            id: Optional vector identifier.
            metadata: Optional key-value metadata.
            collection: Optional target collection.

        Returns:
            Vector identifier.
        """
        payload = CreateVectorRequest(id=id, values=values, metadata=metadata, collection=collection)
        data = await self._api_post("/vectors", payload.model_dump(exclude_none=True), **kw)
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("id", str(result))
        return str(result)

    async def batch_create_vectors(
        self,
        *,
        vectors: List[Dict[str, Any]],
        **kw,
    ) -> List[str]:
        """Batch create vectors (``POST /api/v1/vectors/batch``).

        Args:
            vectors: List of vector dicts.

        Returns:
            List of created vector identifiers.
        """
        items = [CreateVectorRequest(**v) for v in vectors]
        payload = BatchCreateVectorsRequest(vectors=items)
        data = await self._api_post("/vectors/batch", payload.model_dump(exclude_none=True), **kw)
        result = data.get("data", data)
        if isinstance(result, list):
            return [r.get("id", str(r)) if isinstance(r, dict) else str(r) for r in result]
        return [str(result)]

    async def get_global_vector(self, id: str, **kw) -> Dict[str, Any]:
        """Get a global vector by ID (``GET /api/v1/vectors/{id}``).

        Args:
            id: Vector identifier.

        Returns:
            Vector data.
        """
        data = await self._api_get(f"/vectors/{id}", **kw)
        return data.get("data", data)

    async def delete_global_vector(self, id: str, **kw) -> VecminResponse:
        """Delete a global vector (``DELETE /api/v1/vectors/{id}``).

        Args:
            id: Vector identifier.

        Returns:
            Deletion confirmation.
        """
        data = await self._api_delete(f"/vectors/{id}", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    async def batch_delete_vectors(self, *, ids: List[str], **kw) -> VecminResponse:
        """Batch delete vectors (``POST /api/v1/vectors/batch/delete``).

        Args:
            ids: List of vector identifiers to delete.

        Returns:
            Deletion confirmation.
        """
        payload = BatchDeleteVectorsRequest(ids=ids)
        data = await self._api_post("/vectors/batch/delete", payload.model_dump(), **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    async def search_vectors(
        self,
        *,
        query: List[float],
        top_k: int = 10,
        ef_search: int = 50,
        metric: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> SearchResponse:
        """Global vector search (``POST /api/v1/vectors/search``).

        Args:
            query: Query embedding.
            top_k: Number of results.
            ef_search: HNSW search width.
            metric: Override distance metric.
            filter: Metadata filter.

        Returns:
            Search results.
        """
        payload = SearchRequest(query=query, k=top_k, ef_search=ef_search, metric=metric, filter=filter)
        data = await self._api_post("/vectors/search", payload.model_dump(exclude_none=True), **kw)
        raw = data.get("data", data)
        if isinstance(raw, list):
            return SearchResponse(results=[SearchHit(**h) for h in raw if isinstance(h, dict)])
        if isinstance(raw, dict):
            hits = raw.get("results", [])
            return SearchResponse(results=[SearchHit(**h) for h in hits], total=raw.get("total", len(hits)))
        return SearchResponse()

    async def count_vectors(self, **kw) -> int:
        """Count all vectors (``GET /api/v1/vectors/count``).

        Returns:
            Total number of vectors.
        """
        data = await self._api_get("/vectors/count", **kw)
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("count", 0)
        return int(result) if result else 0

    async def rebuild_global_index(self, **kw) -> VecminResponse:
        """Rebuild global index (``POST /api/v1/vectors/rebuild-index``).

        Returns:
            Operation acknowledgement.
        """
        data = await self._api_post("/vectors/rebuild-index", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    # ==================================================================
    # Index Management
    # ==================================================================

    async def list_indexes(self, **kw) -> List[IndexInfo]:
        """List all indexes (``GET /index``).

        Returns:
            List of index information objects.
        """
        data = await self._request("GET", "/index", **kw)
        items = data.get("data", data)
        if isinstance(items, list):
            return [IndexInfo(**idx) for idx in items]
        return [IndexInfo(**items)] if isinstance(items, dict) else []

    async def get_index_info(self, collection_name: str, **kw) -> IndexInfo:
        """Get index information (``GET /index/{collection_name}``).

        Args:
            collection_name: Collection name.

        Returns:
            Index information.
        """
        data = await self._request("GET", f"/index/{collection_name}", **kw)
        return IndexInfo(**data.get("data", data))

    async def rebuild_index(self, collection_name: str, **kw) -> VecminResponse:
        """Rebuild an index (``POST /index/{collection_name}/rebuild``).

        Args:
            collection_name: Collection name.

        Returns:
            Operation acknowledgement.
        """
        data = await self._request("POST", f"/index/{collection_name}/rebuild", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    async def optimize_index(self, collection_name: str, **kw) -> VecminResponse:
        """Optimize an index (``POST /index/{collection_name}/optimize``).

        Args:
            collection_name: Collection name.

        Returns:
            Operation acknowledgement.
        """
        data = await self._request("POST", f"/index/{collection_name}/optimize", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    async def get_shadow_trajectory(self, collection_name: str, **kw) -> List[Dict[str, Any]]:
        """Get shadow-index Nash trajectory (``GET /index/{collection_name}/shadow/trajectory``).

        Args:
            collection_name: Collection name.

        Returns:
            List of trajectory data points.
        """
        data = await self._request("GET", f"/index/{collection_name}/shadow/trajectory", **kw)
        items = data.get("data", data)
        if isinstance(items, list):
            return items
        return [items] if isinstance(items, dict) else []

    async def promote_shadow(self, collection_name: str, **kw) -> VecminResponse:
        """Promote shadow index to primary (``POST /index/{collection_name}/shadow/promote``).

        Args:
            collection_name: Collection name.

        Returns:
            Promotion acknowledgement.
        """
        data = await self._request("POST", f"/index/{collection_name}/shadow/promote", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    # ==================================================================
    # Stats
    # ==================================================================

    async def global_stats(self, **kw) -> GlobalStats:
        """Fetch global database statistics (``GET /stats``).

        Returns:
            Aggregate statistics.
        """
        data = await self._request("GET", "/stats", **kw)
        return GlobalStats(**data.get("data", data))

    async def index_stats(self, **kw) -> Dict[str, Any]:
        """Fetch index statistics (``GET /stats/index``).

        Returns:
            Index statistics.
        """
        data = await self._request("GET", "/stats/index", **kw)
        return data.get("data", data)

    async def performance_stats(self, **kw) -> Dict[str, Any]:
        """Fetch performance statistics (``GET /stats/performance``).

        Returns:
            Performance statistics.
        """
        data = await self._request("GET", "/stats/performance", **kw)
        return data.get("data", data)

    # ==================================================================
    # Cluster Management
    # ==================================================================

    async def cluster_login(self, password: str, **kw) -> ClusterLoginResponse:
        """Authenticate to the cluster and obtain a JWT (``POST /api/v1/cluster/login``).

        The obtained token is automatically stored in the auth manager for
        subsequent requests and auto-refresh.

        Args:
            password: Admin password.

        Returns:
            Login response containing the JWT token.
        """
        payload = ClusterLoginRequest(password=password)
        data = await self._api_post("/cluster/login", payload.model_dump(), **kw)
        result = data.get("data", data)
        login_resp = ClusterLoginResponse(**result)
        self._auth.update_jwt(login_resp.token, login_resp.expires_in)
        return login_resp

    async def cluster_join(self, node_id: str, addr: str, **kw) -> VecminResponse:
        """Join a cluster (``POST /api/v1/cluster/join``).

        Args:
            node_id: Unique node identifier.
            addr: Network address of the joining node.

        Returns:
            Join acknowledgement.
        """
        payload = ClusterJoinRequest(node_id=node_id, addr=addr)
        data = await self._api_post("/cluster/join", payload.model_dump(), **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    async def cluster_promote(self, node_id: str, **kw) -> VecminResponse:
        """Promote a node (``POST /api/v1/cluster/promote``).

        Args:
            node_id: Node to promote.

        Returns:
            Promotion acknowledgement.
        """
        payload = ClusterPromoteRequest(node_id=node_id)
        data = await self._api_post("/cluster/promote", payload.model_dump(), **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    async def list_nodes(self, **kw) -> List[ClusterNodeInfo]:
        """List cluster nodes.

        The server exposes node management via POST (add) and DELETE (remove).
        This method calls ``GET /api/v1/cluster/status`` and extracts the
        node list.

        Returns:
            List of cluster node descriptors.
        """
        status = await self.cluster_status(**kw)
        return status.nodes

    async def cluster_status(self, **kw) -> ClusterStatus:
        """Get cluster status (``GET /api/v1/cluster/status``).

        Returns:
            Cluster status including node list.
        """
        data = await self._api_get("/cluster/status", **kw)
        return ClusterStatus(**data.get("data", data))

    async def add_node(self, node_id: str, addr: str, **kw) -> VecminResponse:
        """Add a node to the cluster (``POST /api/v1/cluster/nodes``).

        Args:
            node_id: New node identifier.
            addr: Network address.

        Returns:
            Addition acknowledgement.
        """
        data = await self._api_post("/cluster/nodes", {"node_id": node_id, "addr": addr}, **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    async def remove_node(self, node_id: str, **kw) -> VecminResponse:
        """Remove a node from the cluster (``DELETE /api/v1/cluster/nodes``).

        Args:
            node_id: Node identifier to remove.

        Returns:
            Removal acknowledgement.
        """
        data = await self._api_delete(f"/cluster/nodes?node_id={node_id}", **kw)
        return VecminResponse(**data) if isinstance(data, dict) else VecminResponse(data=data)

    async def get_join_info(self, **kw) -> Dict[str, Any]:
        """Get cluster join information (``GET /api/v1/cluster/join_info``).

        Returns:
            Join information.
        """
        data = await self._api_get("/cluster/join_info", **kw)
        return data.get("data", data)

    # ==================================================================
    # MCP Integration
    # ==================================================================

    async def mcp_store_memory(
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
        data = await self._api_post("/mcp/message", payload, agent_id=effective_agent_id, model_id=effective_model_id, **kw)
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("content", result.get("result", str(result)))
        return str(result)

    async def mcp_search_memory(
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
        data = await self._api_post("/mcp/message", payload, agent_id=effective_agent_id, model_id=effective_model_id, **kw)
        result = data.get("data", data)
        if isinstance(result, dict):
            return result.get("content", result.get("result", str(result)))
        return str(result)

    async def mount_memory(
        self,
        domain: Optional[str] = None,
        *,
        name: Optional[str] = None,
        sovereignty_token: Optional[str] = None,
        mode: str = "evolutionary",
        model_id: Optional[str] = None,
    ) -> "AsyncVecminMemorySpace":
        """Mount an asynchronous cognitive memory space for agent operation.

        Args:
            domain: Alias for name/collection identifier.
            name: Unified cognitive space identifier.
            sovereignty_token: Sovereign domain access token.
            mode: Operating mode (e.g., 'evolutionary').
            model_id: Embedding feature extraction model targeting.

        Returns:
            AsyncVecminMemorySpace instance.
        """
        from .memory import AsyncAgentMemoryManager
        resolved_name = domain or name or "default"
        resolved_token = sovereignty_token or self.sovereignty_token
        manager = AsyncAgentMemoryManager(
            client=self,
            agent_id=self.agent_id,
            sovereignty_token=resolved_token,
            model_id=model_id,
        )
        return await manager.mount_memory(domain=resolved_name, mode=mode)

    # ==================================================================
    # Convenience: ensure_collection (backward compat)
    # ==================================================================

    async def ensure_collection(
        self,
        name: str,
        *,
        dimension: int = 1536,
        metric_type: str = "Cosine",
        index_type: str = "HNSW",
        index_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Ensure a collection exists, creating it if necessary.

        This is a convenience method that first tries :meth:`get_collection`
        and falls back to :meth:`create_collection`.

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
            await self.get_collection(name)
            return True
        except Exception:
            pass
        try:
            await self.create_collection(
                name,
                dimension=dimension,
                metric_type=metric_type,
                index_type=index_type,
                index_params=index_params,
            )
            return True
        except Exception:
            return False
