# @vecmindb/sdk

Official TypeScript SDK for [VecminDB](https://vecmindb.com) — the high-performance vector database for AI agents.

## Installation

```bash
npm install @vecmindb/sdk
```

### Optional: LangChain.js Integration

```bash
npm install @langchain/core
```

> `@langchain/core` is an optional peer dependency. It is only needed if you use the `VecminDBVectorStore` adapter.

## Quick Start

```typescript
import { VecminClient } from "@vecmindb/sdk";

const client = new VecminClient({
  baseUrl: "http://localhost:8080",
  apiKey: "your-api-key",
});

// Create a collection
await client.createCollection({
  name: "documents",
  dimension: 1536,
  metric_type: "Cosine",
  index_type: "HNSW",
});

// Insert a vector
const id = await client.insert("documents", {
  vector: [0.1, 0.2, /* ... */ 0.5],
  metadata: { title: "My Document", source: "test" },
});

// Search for similar vectors
const results = await client.search("documents", {
  query: [0.1, 0.15, /* ... */ 0.48],
  k: 5,
  ef_search: 100,
});

for (const result of results) {
  console.log(`ID: ${result.id}, Score: ${result.score}`);
  console.log("Metadata:", result.metadata);
}

// Clean up
await client.close();
```

## Authentication

### API Key

The simplest authentication method — pass your API key in the constructor:

```typescript
const client = new VecminClient({
  baseUrl: "http://localhost:8080",
  apiKey: "your-api-key",
});
```

The SDK sends it as the `x-api-key` header on every request.

### JWT

For JWT-based authentication, call `login()` after constructing the client:

```typescript
const client = new VecminClient({ baseUrl: "http://localhost:8080" });

const token = await client.login({
  username: "admin",
  password: "secret",
});

// The JWT is now cached and sent automatically.
// It will be refreshed transparently 5 minutes before expiry.
```

## Collection Management

```typescript
// Create
await client.createCollection({
  name: "my_collection",
  dimension: 768,
  metric_type: "Cosine",    // "Cosine" | "L2" | "InnerProduct"
  index_type: "HNSW",       // "HNSW" | "IVF" | "Flat"
  index_params: { m: 16, ef_construction: 100 },
});

// List
const collections = await client.listCollections();

// Get details
const col = await client.getCollection("my_collection");

// Statistics
const stats = await client.getCollectionStats("my_collection");
console.log(`Vectors: ${stats.vector_count}, Storage: ${stats.storage_bytes} bytes`);

// Delete
await client.deleteCollection("my_collection");
```

## Vector Operations

```typescript
// Single insert
const id = await client.insert("my_collection", {
  id: "doc-001",               // optional — auto-generated if omitted
  vector: [0.1, 0.2, /* ... */],
  metadata: { title: "Hello" },
});

// Batch insert
const ids = await client.batchInsert("my_collection", [
  { vector: [0.1, 0.2], metadata: { title: "A" } },
  { vector: [0.3, 0.4], metadata: { title: "B" } },
]);

// Search
const results = await client.search("my_collection", {
  query: [0.15, 0.25],
  k: 10,
  ef_search: 100,
  filter: { category: "tech" },  // optional metadata filter
});

// Get a single vector
const vec = await client.getVector("my_collection", "doc-001");

// Delete a vector
await client.deleteVector("my_collection", "doc-001");
```

## Index Management

```typescript
// Rebuild a collection's index
await client.rebuildIndex("my_collection");

// List all indexes
const indexes = await client.listIndexes();

// Rebuild a named index
await client.rebuildNamedIndex("my_collection_hnsw");

// Optimize an index
await client.optimizeIndex("my_collection_hnsw");
```

## Cluster Management

```typescript
// Login
const jwt = await client.login({ username: "admin", password: "secret" });

// List cluster nodes
const nodes = await client.listNodes();
for (const node of nodes) {
  console.log(`${node.id} (${node.role}) — ${node.healthy ? "healthy" : "down"}`);
}

// Cluster status
const status = await client.clusterStatus();
console.log(`Status: ${status.status}, Leader: ${status.leader_id}`);

// Create snapshot
await client.createSnapshot();
```

## MCP (Model Context Protocol)

The MCP client provides a persistent SSE connection and JSON-RPC 2.0 interface for AI agent memory operations.

### Convenience Methods (via VecminClient)

```typescript
// Store a memory
await client.mcpStoreMemory("User prefers dark mode", "agent-1", {
  source: "conversation",
});

// Search memories
const memories = await client.mcpSearchMemory("user preferences", "agent-1", 5);
for (const m of memories) {
  console.log(m.content, m.score);
}
```

### Low-level MCP Client (SSE + JSON-RPC)

```typescript
import { VecminMCPClient } from "@vecmindb/sdk";

const mcp = new VecminMCPClient("http://localhost:8080", {
  apiKey: "your-api-key",
  agentId: "agent-1",
});

// Connect to the SSE event stream
await mcp.connect();

// Listen for events
mcp.on("event", ({ event, data }) => {
  console.log(`SSE event: ${event}`, data);
});

// Store a memory via JSON-RPC
await mcp.storeMemory({
  text: "User prefers dark mode",
  agent_id: "agent-1",
  source: "conversation",
});

// Search memories via JSON-RPC
const results = await mcp.searchMemory({
  query: "user preferences",
  agent_id: "agent-1",
  top_k: 5,
});

// Disconnect
await mcp.disconnect();
```

## LangChain.js Integration

Use VecminDB as a vector store in your LangChain.js applications:

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { VecminDBVectorStore } from "@vecmindb/sdk";

const embeddings = new OpenAIEmbeddings();
const store = new VecminDBVectorStore(embeddings, {
  baseUrl: "http://localhost:8080",
  apiKey: "your-api-key",
  collectionName: "my_docs",
  dimension: 1536,
});

// Add documents
await store.addDocuments([
  { pageContent: "VecminDB is a high-performance vector database", metadata: { source: "readme" } },
  { pageContent: "It supports HNSW, IVF, and Flat indexes", metadata: { source: "docs" } },
]);

// Similarity search
const results = await store.similaritySearch("vector database", 5);
for (const doc of results) {
  console.log(doc.pageContent, doc.metadata);
}

// Search with scores
const scored = await store.similaritySearchWithScore("vector database", 5);
for (const [doc, score] of scored) {
  console.log(`Score: ${score} — ${doc.pageContent}`);
}
```

### Factory Methods

```typescript
// From texts
const store = await VecminDBVectorStore.fromTexts(
  ["Hello world", "Goodbye world"],
  [{ label: "greeting" }, { label: "farewell" }],
  embeddings,
  { baseUrl: "http://localhost:8080", apiKey: "your-api-key" },
);

// From documents
const store = await VecminDBVectorStore.fromDocuments(
  [{ pageContent: "Hello", metadata: { label: "greeting" } }],
  embeddings,
  { baseUrl: "http://localhost:8080", apiKey: "your-api-key" },
);
```

## Error Handling

All errors thrown by the SDK are subclasses of `VecminError`:

```typescript
import {
  VecminError,
  AuthenticationError,
  PermissionError,
  NotFoundError,
  RateLimitError,
  ServerError,
} from "@vecmindb/sdk";

try {
  await client.getCollection("nonexistent");
} catch (err) {
  if (err instanceof NotFoundError) {
    console.log("Collection not found:", err.message);
  } else if (err instanceof AuthenticationError) {
    console.log("Check your API key");
  } else if (err instanceof RateLimitError) {
    console.log("Slow down!");
  } else if (err instanceof VecminError) {
    console.log(`Error ${err.code}: ${err.message}`);
  }
}
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `baseUrl` | `string` | — | VecminDB server URL |
| `apiKey` | `string` | — | API key for `x-api-key` header |
| `jwt` | `string` | — | JWT token for `Authorization` header |
| `timeout` | `number` | `30000` | Request timeout (ms) |
| `maxRetries` | `number` | `3` | Max retry attempts on transient failures |
| `backoffFactor` | `number` | `0.5` | Exponential backoff multiplier (seconds) |
| `defaultHeaders` | `Record<string, string>` | `{}` | Custom headers for every request |
| `agentId` | `string` | — | Agent identifier (`x-agent-id` header) |
| `modelId` | `string` | — | Model identifier (`x-model-id` header) |

## Retry Strategy

The SDK automatically retries transient failures (HTTP 429, 500, 502, 503, 504) using exponential backoff with jitter:

- **Attempt 1**: ~500ms delay
- **Attempt 2**: ~1000ms delay
- **Attempt 3**: ~2000ms delay

Each delay includes ±100ms of random jitter to avoid thundering-herd retries.

## License

Apache-2.0
