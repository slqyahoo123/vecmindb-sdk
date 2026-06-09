"""VecminDB Agent Memory Manager and Cognitive Spaces.

Provides:
- AgentMemoryManager: Sync memory manager for spawning cognitive spaces.
- AsyncAgentMemoryManager: Async memory manager for spawning cognitive spaces.
- VecminMemorySpace: Sync operations container for memory store, search, evolution, and centroids.
- AsyncVecminMemorySpace: Async operations container for memory store, search, evolution, and centroids.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import VecminClient
    from .async_client import AsyncVecminClient


class AgentMemoryManager:
    """Synchronous Agent Memory Manager for mounting memory spaces."""

    def __init__(
        self,
        client: VecminClient,
        agent_id: str,
        sovereignty_token: str,
        model_id: Optional[str] = None,
    ) -> None:
        self.client = client
        self.agent_id = agent_id
        self.sovereignty_token = sovereignty_token
        self.model_id = model_id or sovereignty_token

    def mount_memory(
        self,
        domain: str = "default",
        mode: str = "evolutionary",
        **kwargs,
    ) -> VecminMemorySpace:
        """Mount a cognitive memory space, ensuring the backing collection exists."""
        self.client.ensure_collection(
            domain,
            dimension=kwargs.get("dimension", 1536),
            metric_type=kwargs.get("metric_type", "Cosine"),
            index_type=kwargs.get("index_type", "HNSW"),
        )
        return VecminMemorySpace(
            client=self.client,
            collection_name=domain,
            agent_id=self.agent_id,
            sovereignty_token=self.sovereignty_token,
            model_id=self.model_id,
            mode=mode,
        )


class AsyncAgentMemoryManager:
    """Asynchronous Agent Memory Manager for mounting memory spaces."""

    def __init__(
        self,
        client: AsyncVecminClient,
        agent_id: str,
        sovereignty_token: str,
        model_id: Optional[str] = None,
    ) -> None:
        self.client = client
        self.agent_id = agent_id
        self.sovereignty_token = sovereignty_token
        self.model_id = model_id or sovereignty_token

    async def mount_memory(
        self,
        domain: str = "default",
        mode: str = "evolutionary",
        **kwargs,
    ) -> AsyncVecminMemorySpace:
        """Mount a cognitive memory space asynchronously, ensuring the backing collection exists."""
        await self.client.ensure_collection(
            domain,
            dimension=kwargs.get("dimension", 1536),
            metric_type=kwargs.get("metric_type", "Cosine"),
            index_type=kwargs.get("index_type", "HNSW"),
        )
        return AsyncVecminMemorySpace(
            client=self.client,
            collection_name=domain,
            agent_id=self.agent_id,
            sovereignty_token=self.sovereignty_token,
            model_id=self.model_id,
            mode=mode,
        )


class VecminMemorySpace:
    """Synchronous Cognitive Memory Space partition."""

    def __init__(
        self,
        client: VecminClient,
        collection_name: str,
        agent_id: str,
        sovereignty_token: str,
        model_id: str,
        mode: str = "evolutionary",
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.agent_id = agent_id
        self.sovereignty_token = sovereignty_token
        self.model_id = model_id
        self.mode = mode

    def store_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "sdk",
        **kwargs,
    ) -> str:
        """Store episodic memory into the sovereign domain partition."""
        return self.client.mcp_store_memory(
            content=text,
            agent_id=self.agent_id,
            sovereignty_token=self.sovereignty_token,
            model_id=self.model_id,
            source=source,
            **kwargs,
        )

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Retrieve matching episodic or evolved memories."""
        res_str = self.client.mcp_search_memory(
            query=query,
            agent_id=self.agent_id,
            sovereignty_token=self.sovereignty_token,
            model_id=self.model_id,
            top_k=top_k,
            **kwargs,
        )
        try:
            parsed = json.loads(res_str)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                for key in ["results", "hits", "data", "content"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return [parsed]
            return [{"text": res_str}]
        except Exception:
            return [{"text": res_str}]

    def evolve(self, **kwargs) -> Dict[str, Any]:
        """Trigger cognitive evolution of memories under this sovereignty space.

        Runs decision loops for pending candidates and optimizes the index layout.
        """
        resp = self.client._api_get(f"/agents/{self.agent_id}/promotion/pending")
        data = resp.get("data", {})
        candidates = data.get("candidates", []) if isinstance(data, dict) else []

        decisions = []
        for c in candidates:
            c_id = c.get("id")
            if not c_id:
                continue
            try:
                dec_resp = self.client._api_post(
                    f"/agents/{self.agent_id}/promotion/decide",
                    body={"candidate_id": c_id}
                )
                dec_data = dec_resp.get("data", {})
                decisions.append({
                    "candidate_id": c_id,
                    "decision": dec_data.get("decision", "deferred")
                })
            except Exception as e:
                decisions.append({
                    "candidate_id": c_id,
                    "decision": f"error: {str(e)}"
                })

        try:
            self.client._api_post(f"/collections/{self.collection_name}/optimize", body={})
            opt_status = "optimized"
        except Exception as e:
            opt_status = f"failed_optimization: {str(e)}"

        return {
            "status": "success",
            "candidates_evaluated": len(candidates),
            "decisions": decisions,
            "index_optimization": opt_status,
        }

    def get_centroids(self, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve LTSM distilled abstract centroids under this cognitive space."""
        resp = self.client._api_get(f"/centroids/{self.collection_name}")
        data = resp.get("data", {})
        if isinstance(data, dict):
            return data.get("centroids", [])
        elif isinstance(data, list):
            return data
        return []


class AsyncVecminMemorySpace:
    """Asynchronous Cognitive Memory Space partition."""

    def __init__(
        self,
        client: AsyncVecminClient,
        collection_name: str,
        agent_id: str,
        sovereignty_token: str,
        model_id: str,
        mode: str = "evolutionary",
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.agent_id = agent_id
        self.sovereignty_token = sovereignty_token
        self.model_id = model_id
        self.mode = mode

    async def store_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "sdk",
        **kwargs,
    ) -> str:
        """Store episodic memory asynchronously into the sovereign domain partition."""
        return await self.client.mcp_store_memory(
            content=text,
            agent_id=self.agent_id,
            sovereignty_token=self.sovereignty_token,
            model_id=self.model_id,
            source=source,
            **kwargs,
        )

    async def search_memory(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Retrieve matching episodic or evolved memories asynchronously."""
        res_str = await self.client.mcp_search_memory(
            query=query,
            agent_id=self.agent_id,
            sovereignty_token=self.sovereignty_token,
            model_id=self.model_id,
            top_k=top_k,
            **kwargs,
        )
        try:
            parsed = json.loads(res_str)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                for key in ["results", "hits", "data", "content"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return [parsed]
            return [{"text": res_str}]
        except Exception:
            return [{"text": res_str}]

    async def evolve(self, **kwargs) -> Dict[str, Any]:
        """Trigger cognitive evolution of memories asynchronously under this sovereignty space.

        Runs decision loops for pending candidates and optimizes the index layout.
        """
        resp = await self.client._api_get(f"/agents/{self.agent_id}/promotion/pending")
        data = resp.get("data", {})
        candidates = data.get("candidates", []) if isinstance(data, dict) else []

        decisions = []
        for c in candidates:
            c_id = c.get("id")
            if not c_id:
                continue
            try:
                dec_resp = await self.client._api_post(
                    f"/agents/{self.agent_id}/promotion/decide",
                    body={"candidate_id": c_id}
                )
                dec_data = dec_resp.get("data", {})
                decisions.append({
                    "candidate_id": c_id,
                    "decision": dec_data.get("decision", "deferred")
                })
            except Exception as e:
                decisions.append({
                    "candidate_id": c_id,
                    "decision": f"error: {str(e)}"
                })

        try:
            await self.client._api_post(f"/collections/{self.collection_name}/optimize", body={})
            opt_status = "optimized"
        except Exception as e:
            opt_status = f"failed_optimization: {str(e)}"

        return {
            "status": "success",
            "candidates_evaluated": len(candidates),
            "decisions": decisions,
            "index_optimization": opt_status,
        }

    async def get_centroids(self, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve LTSM distilled abstract centroids asynchronously under this cognitive space."""
        resp = await self.client._api_get(f"/centroids/{self.collection_name}")
        data = resp.get("data", {})
        if isinstance(data, dict):
            return data.get("centroids", [])
        elif isinstance(data, list):
            return data
        return []
