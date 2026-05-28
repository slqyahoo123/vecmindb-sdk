"""VecminDB Python SDK Integration Tests.

These tests exercise the full client API against a running VecminDB instance.
They can also be run in offline/mock mode to verify the SDK structure.
"""

import unittest
import os
from vecmindb import VecminClient, AsyncVecminClient, VecminError
from vecmindb.models import (
    CreateCollectionRequest,
    SearchRequest,
    InsertRequest,
    CollectionInfo,
    SearchResponse,
)
from vecmindb.retry import RetryConfig
from vecmindb.auth import AuthManager


class TestModels(unittest.TestCase):
    """Verify Pydantic model construction and validation."""

    def test_create_collection_request_defaults(self):
        req = CreateCollectionRequest(name="test")
        self.assertEqual(req.name, "test")
        self.assertEqual(req.dimension, 1536)
        self.assertEqual(req.metric_type, "Cosine")
        self.assertEqual(req.index_type, "HNSW")
        self.assertIsNone(req.index_params)

    def test_create_collection_request_custom(self):
        req = CreateCollectionRequest(
            name="docs",
            dimension=768,
            metric_type="L2",
            index_type="IVF",
            index_params={"nlist": 128},
        )
        self.assertEqual(req.dimension, 768)
        self.assertEqual(req.index_params["nlist"], 128)

    def test_search_request(self):
        req = SearchRequest(query=[0.1, 0.2, 0.3], k=5, ef_search=100)
        self.assertEqual(req.k, 5)
        self.assertEqual(len(req.query), 3)

    def test_insert_request(self):
        req = InsertRequest(values=[0.1, 0.2], id="vec-1", metadata={"key": "val"})
        self.assertEqual(req.id, "vec-1")
        self.assertEqual(req.metadata["key"], "val")

    def test_search_response(self):
        resp = SearchResponse(results=[], total=0)
        self.assertEqual(resp.total, 0)

    def test_collection_info(self):
        info = CollectionInfo(name="test", dimension=384, vector_count=100)
        self.assertEqual(info.vector_count, 100)


class TestExceptions(unittest.TestCase):
    """Verify exception hierarchy and mapping."""

    def test_exception_hierarchy(self):
        from vecmindb.exceptions import (
            AuthenticationError,
            PermissionError,
            NotFoundError,
            RateLimitError,
            ServerError,
            ConnectionError,
            TimeoutError,
            exception_from_status,
        )
        self.assertTrue(issubclass(AuthenticationError, VecminError))
        self.assertTrue(issubclass(PermissionError, VecminError))
        self.assertTrue(issubclass(NotFoundError, VecminError))
        self.assertTrue(issubclass(RateLimitError, VecminError))
        self.assertTrue(issubclass(ServerError, VecminError))
        self.assertTrue(issubclass(ConnectionError, VecminError))
        self.assertTrue(issubclass(TimeoutError, VecminError))

    def test_exception_from_status(self):
        from vecmindb.exceptions import (
            AuthenticationError,
            NotFoundError,
            RateLimitError,
            ServerError,
            exception_from_status,
        )
        self.assertIsInstance(exception_from_status(401), AuthenticationError)
        self.assertIsInstance(exception_from_status(404), NotFoundError)
        self.assertIsInstance(exception_from_status(429), RateLimitError)
        self.assertIsInstance(exception_from_status(500), ServerError)
        self.assertIsInstance(exception_from_status(503), ServerError)

    def test_exception_str(self):
        exc = VecminError("test error", code=400)
        self.assertEqual(str(exc), "[400] test error")

        exc2 = VecminError("plain error")
        self.assertEqual(str(exc2), "plain error")


class TestRetryConfig(unittest.TestCase):
    """Verify retry configuration and back-off computation."""

    def test_default_config(self):
        config = RetryConfig()
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.backoff_factor, 0.5)
        self.assertTrue(config.jitter)

    def test_backoff_computation(self):
        config = RetryConfig(backoff_factor=1.0, jitter=False, max_backoff=30.0)
        self.assertAlmostEqual(config.compute_backoff(0), 1.0)
        self.assertAlmostEqual(config.compute_backoff(1), 2.0)
        self.assertAlmostEqual(config.compute_backoff(2), 4.0)
        self.assertAlmostEqual(config.compute_backoff(3), 8.0)

    def test_backoff_capped(self):
        config = RetryConfig(backoff_factor=100.0, jitter=False, max_backoff=30.0)
        self.assertEqual(config.compute_backoff(0), 30.0)


class TestAuthManager(unittest.TestCase):
    """Verify auth manager header generation."""

    def test_api_key_headers(self):
        auth = AuthManager(api_key="test-key-123")
        headers = auth.auth_headers()
        self.assertEqual(headers["x-api-key"], "test-key-123")
        self.assertNotIn("Authorization", headers)

    def test_extra_headers(self):
        auth = AuthManager()
        headers = auth.extra_headers(agent_id="agent-1", request_id="req-1")
        self.assertEqual(headers["x-agent-id"], "agent-1")
        self.assertEqual(headers["x-request-id"], "req-1")
        self.assertNotIn("x-model-id", headers)

    def test_jwt_headers(self):
        import time
        auth = AuthManager(jwt_token="jwt-abc", jwt_expires_at=time.time() + 3600)
        headers = auth.auth_headers()
        self.assertEqual(headers["Authorization"], "Bearer jwt-abc")


class TestClientConstruction(unittest.TestCase):
    """Verify client can be constructed without a running server."""

    def test_sync_client_construction(self):
        client = VecminClient(
            "http://localhost:5520",
            api_key="test-key",
            connect_timeout=3.0,
            read_timeout=10.0,
            retry_config=RetryConfig(max_retries=0),
        )
        self.assertIsNotNone(client)
        client.close()

    def test_async_client_construction(self):
        client = AsyncVecminClient(
            "http://localhost:5520",
            api_key="test-key",
            connect_timeout=3.0,
            read_timeout=10.0,
            retry_config=RetryConfig(max_retries=0),
        )
        self.assertIsNotNone(client)

    def test_context_manager(self):
        with VecminClient("http://localhost:5520", retry_config=RetryConfig(max_retries=0)) as client:
            self.assertIsNotNone(client)


class TestLiveIntegration(unittest.TestCase):
    """Integration tests that require a running VecminDB instance.

    These tests are skipped automatically if the server is not reachable.
    """

    def setUp(self):
        self.api_key = os.getenv("VECMIN_API_KEY", "")
        self.base_url = os.getenv("VECMIN_URL", "http://localhost:5520")
        self.client = VecminClient(
            base_url=self.base_url,
            api_key=self.api_key or None,
            retry_config=RetryConfig(max_retries=1),
        )
        self.test_collection = "sdk_integration_test"

    def tearDown(self):
        self.client.close()

    def is_server_ready(self):
        try:
            return self.client.liveness()
        except Exception:
            return False

    def test_health_check(self):
        if not self.is_server_ready():
            self.skipTest("VecminDB server not running.")
        health = self.client.health_check()
        self.assertIn(health.status, ("healthy", "degraded"))

    def test_end_to_end_flow(self):
        if not self.is_server_ready():
            self.skipTest("VecminDB server not running.")

        # 1. Create collection
        self.client.ensure_collection(self.test_collection, dimension=3)

        # 2. Insert
        doc_id = self.client.insert(
            self.test_collection,
            vector=[0.1, 0.2, 0.3],
            metadata={"agent_id": "sdk_test", "text": "hello world"},
        )
        self.assertIsNotNone(doc_id)

        # 3. Search
        response = self.client.search(
            self.test_collection,
            query=[0.1, 0.2, 0.3],
            top_k=1,
        )
        self.assertIsInstance(response, SearchResponse)

        # 4. Get collection stats
        stats = self.client.get_collection_stats(self.test_collection)
        self.assertEqual(stats.name, self.test_collection)

        # 5. List collections
        collections = self.client.list_collections()
        names = [c.name for c in collections]
        self.assertIn(self.test_collection, names)

        # 6. Delete vector
        self.client.delete_vector(self.test_collection, doc_id)

        # 7. Delete collection
        self.client.delete_collection(self.test_collection)


if __name__ == "__main__":
    unittest.main()
