# MCP interface

The optional MCP layer exposes the verification kernel through a small HTTP job service and a local stdio client. Configure the client with an environment variable rather than committing a host address:

```json
{
  "command": "node",
  "args": ["/path/to/worldledger/mcp/shell/server.js"],
  "env": { "ORGANOID_API": "http://127.0.0.1:8663" }
}
```

`ORGANOID_API` is a compatibility environment variable retained by the client. It must point to a service you control. The public release contains no server address, credentials, tunnel command, or deployment target.

The service supports inspect, run, batch, result, artifact, upload, and golden-compare routes. The semantics of `not_evaluated` and the evidence ledger are documented in the report and architecture notes.
