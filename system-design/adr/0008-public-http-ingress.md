# ADR 0008: Bound public HTTP bodies before application execution

## Status
Proposed; pending independent review.

## Context
Field validation and inference admission start after the HTTP body may already
have been buffered and JSON-decoded. They do not bound an oversized JSON envelope.
The playground uses relative same-origin URLs; public wildcard CORS is unnecessary.

## Decision
Public-demo startup configures `AI_SYSTEMS_LAB_MAX_BODY_BYTES`: default 262144
bytes, valid range 1 through 1048576 inclusive. The 256 KiB default accommodates
current playground/quickstart inputs, maximum ASCII prompt batches, and maximum
ASCII message collections. Unicode and JSON escaping can make model-valid inputs
exceed the aggregate budget. The 1 MiB configuration ceiling prevents accidentally
disabling the protection with a very large number. Developer mode ignores it.

Use a pure ASGI middleware for every HTTP route. Before calling the application,
read actual byte chunks into one bounded accumulator, checking the limit before
each append. Do not retain a list of chunks (including arbitrarily many empty
ones), spawn reader tasks, or decode JSON. Replay an accepted body once, then
forward the original receive channel, preserving SSE disconnect behavior.

This pre-read is intentional: a receive wrapper alone cannot prevent a GET handler
that ignores its body from executing before overflow is discovered. Rejected bodies
must have no downstream side effects. The accumulator is at most the configured
budget; producing the replay bytes entails a bounded copy. A server-owned input
event can already contain a larger chunk; this middleware cannot undo that server
allocation and never copies that oversized chunk into its accumulator.

Reject declared oversize before receive, and actual overflow at the first offending
chunk with sanitized HTTP 413 `request_too_large`. Do not drain the remaining body.
Require one ASCII decimal Content-Length if present (spaces/tabs around the value
and leading zeros are accepted). Reject malformed or duplicate values, CL plus
Transfer-Encoding, and completed length mismatches with sanitized HTTP 400
`invalid_content_length`. Actual overflow takes precedence over an underreported
length mismatch. Avoid converting arbitrarily large header integers. The ASGI
server remains responsible for HTTP wire framing and transfer decoding.

Public mode installs no CORS permission middleware. Same-origin requests work;
cross-origin OPTIONS preflights get the application's 405 without permission
headers. Developer mode retains the original wildcard middleware. No origin
allowlist or credentialed cross-origin support is needed. CORS is not authentication.

## Consequences and limits
Bodies up to the byte limit still undergo existing JSON/model validation; success,
503 overload, 504 inference deadline, SSE, and execution ownership remain unchanged.
Disconnect or cancellation before body completion never starts the application,
and no detached middleware tasks need cleanup. Middleware emits its own errors only
before invoking the application, preventing duplicate responses after response start.

The budget is per HTTP request, not global process memory, active connection count,
upload time, or header bytes. Existing inference deadlines begin after body parsing
and admission; they are not upload deadlines. Rate limiting, authentication, proxy
settings, and deployment remain separate work.
