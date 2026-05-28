import unittest
import os
from vecmindb.client import VecminClient

class TestIntegrations(unittest.TestCase):
    def test_langchain_import(self):
        try:
            from vecmindb.integrations.langchain import VecminDBVectorStore, LANGCHAIN_INSTALLED
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"LangChain integration import failed: {e}")

    def test_llamaindex_import(self):
        try:
            from vecmindb.integrations.llamaindex import VecminDBVectorStore, LLAMAINDEX_INSTALLED
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"LlamaIndex integration import failed: {e}")

if __name__ == "__main__":
    unittest.main()
