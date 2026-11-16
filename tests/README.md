# Cross-restart idempotency tests

Standard-library-only verification that Facet Proof exposure requests are
idempotent across real process restarts. No third-party test runner is needed.

## Run

```sh
python3 tests/test_idempotency_restart.py
```

The harness starts the actual HTTP service
(`python3 -m facetproof`) against a fresh on-disk persistent root
(default `/tmp/facetproof-idempotency`, override with `FP_TEST_WORKSPACE`),
restarts the server process between phases, and stops it on exit. The data root
and a JSON ledger are retained for inspection.

## What is covered

1. **Same key, same frozen input.** Re-submitting the identical request returns
   the same exposure id while queued, after a full restart, and after success;
   the attempt counter and lifecycle events are untouched and canvas capacity is
   reserved exactly once.
2. **Same key, different input.** A `key_conflict` (HTTP 409) is returned and the
   original task, its stored request and its events are never mutated. The
   original task still renders a plate attributed to its own frozen seal, while a
   separately keyed request for the other scene renders a byte-distinct plate.
3. **Explicit retry vs network re-send.** A separate worker process
   (`failure_worker.py`) takes a real lease and fails the task using the
   service's public lease-checked `claim()`/`fail()` API. A plain re-send then
   keeps the failed task and its attempt count and writes no `retried` event;
   the explicit `/retry` requeues the same id, clears the lease, bumps the
   allocation generation and the attempt count, and renders a plate on attempt 2.
4. **Persistence and verification.** Content-addressed blobs exist on disk and
   every recorded plate passes the six-pass `/verify` check after another
   restart.

## Why assertions inspect plate bytes, not just status codes

The conflict and retry paths could return 200/409 correctly yet still hand a
caller a plate rendered from the wrong frozen input. The tests therefore read
the actual `object.u32` / `depth.f32` raw channels and all three PNG passes,
compare their SHA-256 hashes between distinct scenes, validate the object-id
channel against each plate's own seal mapping, and use `/pick` to confirm pixel
attribution. The request → exposure → plate correspondence (including hashes) is
written to `idempotency-ledger.json` next to the data root.

## Helper: `failure_worker.py`

```sh
python3 tests/failure_worker.py <data-root> [expected-exposure-id]
```

A minimal worker process that claims one queued exposure and reports it failed.
It deliberately uses only the public `Exposures` API so the resulting `failed`
state is a genuine service-recorded transition (real lease, real attempt, real
event), equivalent to the production worker's render-fault branch.
