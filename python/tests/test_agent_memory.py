"""Unit tests for the VecminDB Agent OS Memory abstraction."""

import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from vecmindb import (
    VecminClient,
    AsyncVecminClient,
    connect,
    AgentMemoryManager,
    AsyncAgentMemoryManager,
    VecminMemorySpace,
    AsyncVecminMemorySpace,
)


class TestAgentMemory(unittest.IsolatedAsyncioTestCase):
    """Verify sovereignty agent isolation and cognitive memory space lifecycle."""

    def test_client_init(self) -> None:
        """Verify constructor handles agent_id and sovereignty_token correctly."""
        client = VecminClient(
            "http://localhost:5520",
            agent_id="my_agent",
            sovereignty_token="my_token",
        )
        self.assertEqual(client.agent_id, "my_agent")
        self.assertEqual(client.sovereignty_token, "my_token")

        # default values
        client_default = VecminClient("http://localhost:5520")
        self.assertEqual(client_default.agent_id, "default_agent")
        self.assertEqual(client_default.sovereignty_token, "system")

    def test_connect_shortcut(self) -> None:
        """Verify package-level connect() sets agent parameters correctly."""
        client = connect(
            "http://localhost:5520",
            agent_id="agent_123",
            sovereignty_token="sovereign_abc",
        )
        self.assertIsInstance(client, VecminClient)
        self.assertEqual(client.agent_id, "agent_123")
        self.assertEqual(client.sovereignty_token, "sovereign_abc")

    @patch("vecmindb.client.VecminClient._api_post")
    def test_mcp_store_search_sync(self, mock_post: MagicMock) -> None:
        """Verify MCP parameters propagate sovereignty values correctly."""
        mock_post.return_value = {"data": {"content": "Stored successfully"}}
        client = VecminClient(
            "http://localhost:5520",
            agent_id="a1",
            sovereignty_token="t1",
        )

        res = client.mcp_store_memory("my memory content")
        self.assertEqual(res, "Stored successfully")

        # Verify extra headers and arguments propagation
        mock_post.assert_called_once()
        args = mock_post.call_args[0]
        kwargs = mock_post.call_args[1]
        self.assertEqual(args[0], "/mcp/message")
        self.assertEqual(args[1]["params"]["arguments"]["agent_id"], "a1")
        self.assertEqual(args[1]["params"]["arguments"]["sovereignty_token"], "t1")
        self.assertEqual(args[1]["params"]["arguments"]["model_id"], "t1")
        self.assertEqual(kwargs.get("agent_id"), "a1")
        self.assertEqual(kwargs.get("model_id"), "t1")

    @patch("vecmindb.client.VecminClient._api_post")
    @patch("vecmindb.client.VecminClient.ensure_collection")
    @patch("vecmindb.client.VecminClient._api_get")
    def test_memory_space_lifecycle_sync(
        self,
        mock_get: MagicMock,
        mock_ensure: MagicMock,
        mock_post: MagicMock,
    ) -> None:
        """Verify full memory space mounting, storage, search, evolve, and centroids lifecycle."""
        client = VecminClient(
            "http://localhost:5520",
            agent_id="agent_alice",
            sovereignty_token="sovereign_wonderland",
        )

        # Mount memory space
        space = client.mount_memory(domain="my_domain")
        self.assertIsInstance(space, VecminMemorySpace)
        mock_ensure.assert_called_once_with(
            "my_domain",
            dimension=1536,
            metric_type="Cosine",
            index_type="HNSW",
        )

        # Store memory
        mock_post.return_value = {"data": {"content": "anchored"}}
        store_res = space.store_memory("test memory")
        self.assertEqual(store_res, "anchored")

        # Search memory
        mock_post.return_value = {
            "data": {
                "content": '[{"text": "test memory", "score": 0.9}]',
            },
        }
        search_res = space.search_memory("test query")
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["text"], "test memory")

        # Evolve memory
        mock_get.return_value = {"data": {"candidates": [{"id": "cand_1"}]}}
        # Mock decide & optimize
        mock_post.side_effect = [
            {"data": {"decision": "approved"}},  # decide
            {"data": {"status": "optimized"}},  # optimize
        ]
        evolve_res = space.evolve()
        self.assertEqual(evolve_res["status"], "success")
        self.assertEqual(evolve_res["candidates_evaluated"], 1)
        self.assertEqual(evolve_res["decisions"][0]["decision"], "approved")

        # Get centroids
        mock_get.return_value = {
            "data": {
                "centroids": [
                    {
                        "id": "c_1",
                        "vector": [0.1],
                        "weight": 1.0,
                        "source_count": 5,
                        "created_at": "now",
                    },
                ],
            },
        }
        centroids = space.get_centroids()
        self.assertEqual(len(centroids), 1)
        self.assertEqual(centroids[0]["id"], "c_1")

    async def test_async_lifecycle(self) -> None:
        """Verify full asynchronous memory space lifecycle."""
        client = AsyncVecminClient(
            "http://localhost:5520",
            agent_id="agent_bob",
            sovereignty_token="sovereign_builder",
        )

        # Mocking async client internal calls
        client.ensure_collection = AsyncMock()  # type: ignore[method-assign]
        client._api_post = AsyncMock(return_value={"data": {"content": "anchored"}})  # type: ignore[method-assign]
        client._api_get = AsyncMock()  # type: ignore[method-assign]

        space = await client.mount_memory(domain="async_domain")
        self.assertIsInstance(space, AsyncVecminMemorySpace)
        client.ensure_collection.assert_called_once()

        # Store memory
        store_res = await space.store_memory("async test memory")
        self.assertEqual(store_res, "anchored")

        # Search memory
        client._api_post.return_value = {
            "data": {
                "content": '[{"text": "async search result"}]',
            },
        }
        search_res = await space.search_memory("async query")
        self.assertEqual(search_res[0]["text"], "async search result")

        # Evolve memory
        client._api_get.return_value = {
            "data": {"candidates": [{"id": "cand_async_1"}]},
        }
        client._api_post.side_effect = [
            {"data": {"decision": "deferred"}},
            {"data": {"status": "optimized"}},
        ]
        evolve_res = await space.evolve()
        self.assertEqual(evolve_res["status"], "success")
        self.assertEqual(evolve_res["decisions"][0]["decision"], "deferred")

        # Get centroids
        client._api_get.return_value = {
            "data": {"centroids": [{"id": "c_async_1"}]},
        }
        centroids = await space.get_centroids()
        self.assertEqual(centroids[0]["id"], "c_async_1")

    @patch("vecmindb.client.VecminClient._api_post")
    def test_sovereignty_violation_sync(self, mock_post: MagicMock) -> None:
        """Verify sovereignty violation raises AuthenticationError or PermissionError."""
        from vecmindb.exceptions import AuthenticationError, PermissionError

        # Simulate sovereignty_token rejection with 401 Unauthorized
        mock_post.side_effect = AuthenticationError("Sovereignty token validation failed", code=401)
        client = VecminClient("http://localhost:5520", agent_id="hacker", sovereignty_token="invalid_token")
        
        with self.assertRaises(AuthenticationError):
            client.mcp_store_memory("unauthorized write")

        # Simulate sovereignty_token access restriction with 403 Forbidden
        mock_post.side_effect = PermissionError("Access to sovereign domain denied", code=403)
        with self.assertRaises(PermissionError):
            client.mcp_store_memory("forbidden write")

    async def test_sovereignty_violation_async(self) -> None:
        """Verify sovereignty violation in async client raises AuthenticationError or PermissionError."""
        from vecmindb.exceptions import AuthenticationError, PermissionError

        client = AsyncVecminClient("http://localhost:5520", agent_id="hacker_async", sovereignty_token="invalid_token")
        
        # Simulate 401 Unauthorized
        client._api_post = AsyncMock(side_effect=AuthenticationError("Sovereignty token validation failed", code=401))  # type: ignore[method-assign]
        with self.assertRaises(AuthenticationError):
            await client.mcp_store_memory("unauthorized write")

        # Simulate 403 Forbidden
        client._api_post = AsyncMock(side_effect=PermissionError("Access to sovereign domain denied", code=403))  # type: ignore[method-assign]
        with self.assertRaises(PermissionError):
            await client.mcp_store_memory("forbidden write")


if __name__ == "__main__":
    unittest.main()
