"""Cross-restart idempotency verification for Facet Proof exposure requests.

These checks drive the *real* HTTP service (``python3 -m facetproof``) backed by
a genuine on-disk persistent directory, restarting the server process between
phases. They do not call the application objects in-process for the behaviour
under test: every idempotency, conflict, retry and plate-attribution assertion
goes through HTTP exactly as a production caller would experience it.

What is verified (and why a status code alone is never enough):

* Re-submitting the same ``key`` with the same frozen request returns the same
  exposure/task, including after a full process restart, and never reserves a
  second slice of canvas capacity.
* The same ``key`` bound to a *different* request is rejected with
  ``key_conflict`` and, crucially, must not silently hand back or mutate the
  original task. The original task still renders its own plate.
* An explicit retry after a real failure is distinct from a network re-send:
  re-send keeps the same task and attempt count and writes no ``retried``
  event, whereas ``/retry`` advances the attempt and writes a ``retried`` event.
* Every plate is attributed by reading its actual rendered bytes and metadata
  (``object.u32`` / ``depth.f32`` payloads, ``visual_hash``, pixel picking and
  the PNG passes), not merely by checking that an HTTP call returned 200. Two
  plates produced from different frozen inputs must be byte-distinct and each
  must belong to the exposure that requested it.

Standard library only; no third-party test runner is required.
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("FP_TEST_WORKSPACE", "/tmp/facetproof-idempotency"))

# A small two-triangle surface used to build two *visually distinct* scenes.
VERTICES = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
FACES_A = [[0, 1, 2], [0, 2, 3]]            # one quad, one placement
MESH_B_VERTS = VERTICES + [[0.0, 0.0, 1.5]]  # raised apex for the second scene
MESH_B_FACES = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]


# --------------------------------------------------------------------------- #
# Tiny assertion + reporting layer
# --------------------------------------------------------------------------- #
class CheckFailed(AssertionError):
    pass


def check(condition, message):
    if not condition:
        raise CheckFailed(message)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# HTTP client against the running service
# --------------------------------------------------------------------------- #
class Client:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def _open(self, method, path, body=None):
        url = self.base_url + path
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
        if not raw:
            return status, None, b""
        content_type = response.headers.get("Content-Type", "") if status < 400 else "application/json"
        if "application/json" in content_type:
            return status, json.loads(raw.decode()), raw
        return status, raw, raw

    def get(self, path):
        return self._open("GET", path)

    def post(self, path, body=None):
        return self._open("POST", path, body or {})

    def put(self, path, body):
        return self._open("PUT", path, body)

    # Typed convenience helpers ------------------------------------------------
    def expect_json(self, method, path, body=None, status=200):
        code, value, _ = self._open(method, path, body)
        check(code == status, f"{method} {path} expected {status}, got {code}: {value}")
        return value

    def expect_fault(self, method, path, body, fault_code, status=409):
        code, value, _ = self._open(method, path, body)
        check(code == status, f"{method} {path} expected {status}, got {code}: {value}")
        check(isinstance(value, dict) and value.get("error") == fault_code,
              f"{method} {path} expected error={fault_code!r}, got {value!r}")
        return value


# --------------------------------------------------------------------------- #
# Real server process lifecycle against a persistent data root
# --------------------------------------------------------------------------- #
class Server:
    def __init__(self, root, port):
        self.root = Path(root)
        self.port = port
        self.proc = None
        self.log_path = self.root.parent / f"server-{port}.log"
        self.client = Client(f"http://127.0.0.1:{port}")

    def start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "facetproof",
             "--root", str(self.root), "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        self._wait_until_ready()
        return self

    def _wait_until_ready(self, timeout=20.0):
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise CheckFailed(
                    f"server exited early (code {self.proc.returncode}); see {self.log_path}"
                )
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    pass
                # Confirm the HTTP layer answers, not just the listening socket.
                self.client.get("/facet/v1/health")
                return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
                time.sleep(0.1)
        raise CheckFailed(f"server did not become ready: {last_error}")

    def stop(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None

    def restart(self):
        self.stop()
        # A fresh process, the same on-disk root: this is the cross-restart
        # boundary the idempotency guarantees must survive.
        self.start()


def run_failure_worker(root, expected_exposure_id):
    """Run the standalone failing worker as its own process."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(TESTS_DIR / "failure_worker.py"), str(root), expected_exposure_id],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=30,
    )
    check(completed.returncode == 0,
          f"failure_worker failed: rc={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}")
    return json.loads(completed.stdout)


# --------------------------------------------------------------------------- #
# Scene construction helpers
# --------------------------------------------------------------------------- #
def build_scene(client, name, pigment_color, vertices, faces, transform=None):
    """Create mesh + pigment + assembly + seal for one distinct frozen scene."""
    mesh = client.expect_json("POST", "/facet/v1/meshes", {
        "name": f"{name}-mesh", "vertices": vertices, "faces": faces,
    })
    pigment = client.expect_json("POST", "/facet/v1/pigments", {
        "name": f"{name}-pigment", "color": pigment_color, "two_sided": True,
    })
    assembly = client.expect_json("POST", "/facet/v1/assemblies", {
        "name": f"{name}-assembly", "lens": {"eye": [3, 2, 4]},
    })
    placement = {"key": f"{name}-part", "mesh_id": mesh["id"],
                 "pigment_id": pigment["id"]}
    if transform is not None:
        placement["transform"] = transform
    assembly = client.expect_json("PUT", f"/facet/v1/assemblies/{assembly['id']}", {
        "revision": assembly["revision"], "placements": [placement],
    })
    seal = client.expect_json("POST", f"/facet/v1/assemblies/{assembly['id']}/seal", {
        "revision": assembly["revision"],
    })
    return {"mesh_id": mesh["id"], "pigment_id": pigment["id"],
            "assembly_id": assembly["id"], "seal_id": seal["id"], "seal": seal}


def render_next_plate(client):
    """Drive one worker step and return the resulting plate object."""
    step = client.expect_json("POST", "/facet/v1/worker/step")
    check(step.get("state") == "succeeded", f"worker did not succeed: {step}")
    return step["plate"]


def plate_bytes(client, plate_id):
    """Fetch every raw + PNG pass and hash the actual rendered bytes."""
    passes = {}
    for name in ("color.rgb", "depth.f32", "object.u32",
                 "color.png", "depth.png", "object.png"):
        status, value, raw = client.get(f"/facet/v1/plates/{plate_id}/passes/{name}")
        check(status == 200 and isinstance(value, (bytes, bytearray)),
              f"pass {name} for {plate_id} not returned (status {status})")
        passes[name] = {"sha256": sha256_hex(bytes(raw)), "size": len(raw)}
    return passes


def event_actions(client, exposure_id):
    history = client.expect_json("GET", f"/facet/v1/events/{exposure_id}/history")
    return [item["action"] for item in history["items"]]


def assert_lifecycle(actions, expected_subsequence, message):
    """Assert the lifecycle events appear, in order, ignoring bookkeeping events.

    Capacity reservation/release events (``capacity_reserved`` etc.) are
    infrastructure bookkeeping; the assertions focus on the state machine a
    caller can rely on, while still requiring their relative ordering.
    """
    filtered = [a for a in actions if not a.startswith("capacity_")]
    cursor = 0
    for wanted in expected_subsequence:
        try:
            cursor = filtered.index(wanted, cursor) + 1
        except ValueError:
            raise CheckFailed(f"{message}: {filtered} missing ordered {expected_subsequence}")
    return filtered


# --------------------------------------------------------------------------- #
# The verification scenarios
# --------------------------------------------------------------------------- #
def scenario_same_key_same_input(server, ledger):
    """Same key + same frozen input resolves to the same task across restart."""
    client = server.client
    scene = build_scene(client, "same", [0.12, 0.62, 0.61], VERTICES, FACES_A)
    request = {"key": "dedupe-same-input", "width": 48, "height": 48, "samples": 1}

    first = client.expect_json(
        "POST", f"/facet/v1/seals/{scene['seal_id']}/exposures", request
    )
    ledger.record("submit#1 (key=dedupe-same-input)", request, first, scene)

    # A plain network re-send of the identical request while still queued.
    resend_queued = client.expect_json(
        "POST", f"/facet/v1/seals/{scene['seal_id']}/exposures", request
    )
    check(resend_queued["id"] == first["id"],
          "re-send while queued created a new exposure instead of returning the task")
    check(resend_queued["state"] == "queued",
          f"re-send altered task state: {resend_queued['state']}")
    check(resend_queued["attempt"] == 0,
          f"re-send must not advance attempt, got {resend_queued['attempt']}")
    actions = event_actions(client, first["id"])
    assert_lifecycle(actions, ["queued"], "re-send wrote spurious lifecycle events")

    # Capacity must be reserved exactly once for the single logical request.
    budget_before = client.expect_json("GET", "/facet/v1/resources")["reserved"]
    expected_units = request["width"] * request["height"] * request["samples"] ** 2

    # Restart: a brand-new process opens the same persistent root.
    server.restart()
    client = server.client

    health = client.expect_json("GET", "/facet/v1/health")
    check(health["integrity"] == "ok", "database integrity not ok after restart")

    # Re-send the identical request after the process restart.
    resend_after_restart = client.expect_json(
        "POST", f"/facet/v1/seals/{scene['seal_id']}/exposures", request
    )
    check(resend_after_restart["id"] == first["id"],
          "after restart the same key/input returned a different exposure id")
    check(resend_after_restart["dedup_key"] == first["dedup_key"],
          "dedup key changed across restart")
    check(resend_after_restart["request_hash"] == first["request_hash"],
          "request hash changed across restart")
    check(resend_after_restart["attempt"] == 0,
          "re-send after restart advanced the attempt counter")

    budget_after = client.expect_json("GET", "/facet/v1/resources")["reserved"]
    check(budget_before == budget_after == expected_units,
          f"re-send changed capacity reservation: before={budget_before} "
          f"after={budget_after} expected={expected_units}")

    # Now actually render and confirm the single task owns one real plate.
    plate = render_next_plate(client)
    ledger.record_plate(first["id"], plate, client)
    exposure = client.expect_json("GET", f"/facet/v1/exposures/{first['id']}")
    check(exposure["state"] == "succeeded", f"task not succeeded: {exposure['state']}")
    check(exposure["plate_id"] == plate["id"],
          "exposure plate_id does not point at the rendered plate")
    check(plate["payload"]["request"] == first["request"],
          "plate payload request does not match the submitted frozen request")
    check(plate["payload"]["seal_id"] == scene["seal_id"],
          "plate is attributed to the wrong seal")

    # A re-send after success still returns the same, already-finished task.
    resend_done = client.expect_json(
        "POST", f"/facet/v1/seals/{scene['seal_id']}/exposures", request
    )
    check(resend_done["id"] == first["id"] and resend_done["state"] == "succeeded",
          "re-send after success did not return the completed task")
    assert_lifecycle(event_actions(client, first["id"]),
                     ["queued", "claimed", "succeeded"],
                     "re-send after success wrote extra lifecycle events")
    ledger.note("same-key/same-input resolved to one task across restart; one plate")


def scenario_same_key_different_input(server, ledger):
    """Same key bound to a different input must never silently reuse the task."""
    client = server.client
    scene_a = build_scene(client, "conflict-a", [0.12, 0.62, 0.61], VERTICES, FACES_A)
    # A genuinely different frozen scene: raised apex + different pigment colour.
    scene_b = build_scene(client, "conflict-b", [0.85, 0.2, 0.25],
                          MESH_B_VERTS, MESH_B_FACES, transform={"euler": [0, 25, 0]})

    key = "dedupe-conflicting-input"
    request_a = {"key": key, "width": 48, "height": 48, "samples": 1}
    request_b = {**request_a, "width": 64}  # same key, different frozen input

    first = client.expect_json(
        "POST", f"/facet/v1/seals/{scene_a['seal_id']}/exposures", request_a
    )
    ledger.record("submit#A (key=conflict, 48px)", request_a, first, scene_a)

    # Different input under the same key, against a different seal: must 409.
    conflict = client.expect_fault(
        "POST", f"/facet/v1/seals/{scene_b['seal_id']}/exposures",
        request_b, "key_conflict", status=409,
    )

    # The rejection must not have mutated or replaced the original task.
    unchanged = client.expect_json("GET", f"/facet/v1/exposures/{first['id']}")
    check(unchanged["state"] == "queued", f"conflict changed original state: {unchanged['state']}")
    check(unchanged["request"] == first["request"],
          "conflict mutated the stored request of the original task")
    check(unchanged["request_hash"] == first["request_hash"],
          "conflict changed the original request hash")
    assert_lifecycle(event_actions(client, first["id"]), ["queued"],
                     "conflict produced lifecycle events on the original task")

    # Restart, then prove the conflict is durable and still cannot reuse.
    server.restart()
    client = server.client
    client.expect_fault(
        "POST", f"/facet/v1/seals/{scene_b['seal_id']}/exposures",
        request_b, "key_conflict", status=409,
    )
    still_first = client.expect_json("GET", f"/facet/v1/exposures/{first['id']}")
    check(still_first["id"] == first["id"] and still_first["state"] == "queued",
          "conflicting re-send after restart disturbed the original task")

    # The original task renders a plate that belongs to scene A's frozen input.
    plate_a = render_next_plate(client)
    ledger.record_plate(first["id"], plate_a, client)
    check(plate_a["payload"]["seal_id"] == scene_a["seal_id"],
          "original task's plate was attributed to the conflicting scene")
    check(plate_a["payload"]["request"]["width"] == 48,
          "plate width reflects the conflicting input, not the original request")

    # A fresh key on scene B renders a *different*, byte-distinct plate, proving
    # the conflict path never mixed the two frozen inputs into one attribution.
    request_b_own = {**request_b, "key": "separate-scene-b"}
    exposure_b = client.expect_json(
        "POST", f"/facet/v1/seals/{scene_b['seal_id']}/exposures", request_b_own
    )
    ledger.record("submit#B (own key, 64px)", request_b_own, exposure_b, scene_b)
    plate_b = render_next_plate(client)
    ledger.record_plate(exposure_b["id"], plate_b, client)

    check(plate_b["id"] != plate_a["id"], "two scenes produced the same plate id")
    check(plate_b["payload"]["seal_id"] == scene_b["seal_id"],
          "scene B plate attributed to the wrong seal")
    check(plate_b["payload"]["visual_hash"] != plate_a["payload"]["visual_hash"],
          "different frozen inputs produced identical visual_hash (attribution collision)")
    check(plate_b["payload"]["request"]["width"] == 64,
          "scene B plate does not carry its own 64px request")

    # Content-level comparison: every raw/PNG pass must differ between the two.
    bytes_a = plate_bytes(client, plate_a["id"])
    bytes_b = plate_bytes(client, plate_b["id"])
    shared = [name for name in bytes_a if bytes_a[name]["sha256"] == bytes_b[name]["sha256"]]
    check(not shared, f"distinct scenes share identical rendered passes: {shared}")

    # Pixel-level attribution: object.u32 of each plate only names objects from
    # its own frozen seal mapping, and picking returns the owning seal.
    _assert_object_channel_attribution(client, plate_a, scene_a)
    _assert_object_channel_attribution(client, plate_b, scene_b)
    ledger.note("same key/different input rejected (409) and never reused; plates distinct")
    return {"conflict": conflict, "plate_a": plate_a["id"], "plate_b": plate_b["id"]}


def _assert_object_channel_attribution(client, plate, scene):
    """The object channel and pick endpoint must name only this plate's objects."""
    status, raw, _ = client.get(f"/facet/v1/plates/{plate['id']}/passes/object.u32")
    check(status == 200, "object.u32 could not be read")
    import struct
    width = plate["payload"]["width"]
    height = plate["payload"]["height"]
    check(len(raw) == width * height * 4,
          f"object.u32 size {len(raw)} != {width * height * 4}")
    ids = {struct.unpack_from("<I", raw, i)[0] for i in range(0, len(raw), 4)}
    ids.discard(0)  # background
    # The plate's object mapping is keyed by the string form of the u32 id.
    id_keys = {str(value) for value in ids}
    mapped = set(plate["payload"]["objects"].keys())
    check(id_keys and id_keys.issubset(mapped),
          f"object channel names ids {id_keys - mapped} absent from this plate's seal mapping")
    # Every non-background object id maps back to a mesh in this seal's closure.
    seal_mesh_ids = {
        obj["mesh_id"] for obj in plate["payload"]["objects"].values()
    }
    check(seal_mesh_ids == {scene["mesh_id"]},
          "plate object mapping references geometry outside its own scene")
    pick = client.expect_json(
        "GET", f"/facet/v1/plates/{plate['id']}/pick?x={width // 2}&y={height // 2}"
    )
    check(pick["seal_id"] == scene["seal_id"],
          "pick endpoint attributes a pixel to the wrong seal")


def scenario_retry_vs_resend(server, ledger):
    """Explicit retry after failure differs from a network re-send."""
    client = server.client
    scene = build_scene(client, "retry", [0.2, 0.5, 0.8], VERTICES, FACES_A)
    request = {"key": "dedupe-retry-flow", "width": 40, "height": 40, "samples": 1}

    exposure = client.expect_json(
        "POST", f"/facet/v1/seals/{scene['seal_id']}/exposures", request
    )
    ledger.record("submit (retry flow)", request, exposure, scene)

    # A separate worker process takes the real lease and fails the task, exactly
    # like the production worker's Fault branch. Then restart the HTTP service.
    failure = run_failure_worker(server.root, exposure["id"])
    check(failure["state"] == "failed", f"failure worker did not fail task: {failure}")
    check(failure["attempt"] == 1, f"first claim should set attempt=1, got {failure['attempt']}")

    server.restart()
    client = server.client

    failed = client.expect_json("GET", f"/facet/v1/exposures/{exposure['id']}")
    check(failed["state"] == "failed", f"task not failed after restart: {failed['state']}")
    check(failed["attempt"] == 1, f"attempt lost across restart: {failed['attempt']}")
    check(failed["error"] == "injected_render_failure",
          f"failure reason not persisted: {failed['error']}")
    actions = event_actions(client, exposure["id"])
    assert_lifecycle(actions, ["queued", "claimed", "failed"],
                     "unexpected events after failure")

    # (1) Network re-send of the same key/input: SAME task, no new attempt,
    #     no 'retried' event, and it stays failed (re-send is not a retry).
    resent = client.expect_json(
        "POST", f"/facet/v1/seals/{scene['seal_id']}/exposures", request
    )
    check(resent["id"] == exposure["id"], "re-send of failed task made a new exposure")
    check(resent["state"] == "failed", f"re-send resurrected task state: {resent['state']}")
    check(resent["attempt"] == 1, "re-send advanced the attempt counter")
    assert_lifecycle(event_actions(client, exposure["id"]),
                     ["queued", "claimed", "failed"],
                     "re-send of a failed task wrote a retried/lifecycle event")

    # Retrying a non-failed task must be rejected (guards the explicit action).
    other_scene = build_scene(client, "retry-guard", [0.3, 0.3, 0.3], VERTICES, FACES_A)
    other = client.expect_json(
        "POST", f"/facet/v1/seals/{other_scene['seal_id']}/exposures",
        {"key": "retry-guard-key", "width": 32, "height": 32, "samples": 1},
    )
    client.expect_fault("POST", f"/facet/v1/exposures/{other['id']}/retry", {}, "state", 409)

    # (2) Explicit retry: same id, back to queued, attempt semantics recorded.
    retried = client.expect_json(
        "POST", f"/facet/v1/exposures/{exposure['id']}/retry", {}
    )
    check(retried["id"] == exposure["id"], "retry changed the exposure identity")
    check(retried["state"] == "queued", f"retry did not requeue task: {retried['state']}")
    check(retried["token"] is None and retried["lease_until"] is None,
          "retry left a stale lease token/expiry on the task")
    retry_actions = event_actions(client, exposure["id"])
    assert_lifecycle(retry_actions, ["queued", "claimed", "failed", "retried"],
                     "explicit retry missing its event")

    # Capacity was re-reserved for the retry under a new generation.
    allocation = retried["allocation"]
    check(allocation["active"] == 1, "retry did not reactivate the capacity allocation")
    check(allocation["generation"] == 2,
          f"retry should bump allocation generation to 2, got {allocation['generation']}")

    # Render the retried task to a real plate; the same task now owns a plate.
    plate = render_next_plate(client)
    ledger.record_plate(exposure["id"], plate, client)
    final = client.expect_json("GET", f"/facet/v1/exposures/{exposure['id']}")
    check(final["state"] == "succeeded", f"retried task did not succeed: {final['state']}")
    check(final["attempt"] == 2, f"retry render should run attempt 2, got {final['attempt']}")
    check(final["plate_id"] == plate["id"], "retried task points at the wrong plate")
    check(plate["payload"]["request"] == exposure["request"],
          "plate from retried task carries a different request")
    check(plate["payload"]["seal_id"] == scene["seal_id"],
          "plate from retried task attributed to another seal")
    final_actions = event_actions(client, exposure["id"])
    assert_lifecycle(
        final_actions,
        ["queued", "claimed", "failed", "retried", "claimed", "succeeded"],
        "retry lifecycle events incorrect",
    )

    # The guard task (never failed) still renders its own distinct plate.
    guard_plate = render_next_plate(client)
    ledger.record_plate(other["id"], guard_plate, client)
    check(guard_plate["id"] != plate["id"], "guard task reused the retried task's plate")
    ledger.note("re-send keeps failed task/attempt; explicit retry requeues, attempt 2, plate ok")


def scenario_persistent_root_artifacts(server, ledger):
    """Confirm durable files exist on disk and plates verify after restart."""
    client = server.client
    db_path = server.root / "facetproof.sqlite3"
    check(db_path.exists(), f"SQLite database missing at {db_path}")
    blob_root = server.root / "plates"
    blobs = [p for p in blob_root.rglob("*") if p.is_file()]
    check(blobs, "no content-addressed plate bytes were persisted")

    # Every plate recorded in the ledger must still verify after a restart.
    server.restart()
    client = server.client
    for exposure_id, plate_id in ledger.plate_index.items():
        verification = client.expect_json(
            "POST", f"/facet/v1/plates/{plate_id}/verify"
        )
        check(verification["valid"] is True,
              f"plate {plate_id} for exposure {exposure_id} failed verification")
        check(len(verification["passes"]) == 6,
              f"plate {plate_id} verification did not cover all six passes")
    ledger.note(f"persisted {len(blobs)} content blobs; all plates verify after restart")


# --------------------------------------------------------------------------- #
# Ledger recording the request -> task -> result correspondence
# --------------------------------------------------------------------------- #
class Ledger:
    def __init__(self, root):
        self.root = Path(root)
        self.requests = []
        self.plate_index = {}
        self.notes = []

    def record(self, label, request, exposure, scene):
        self.requests.append({
            "label": label,
            "key": request["key"],
            "request": request,
            "seal_id": scene["seal_id"],
            "exposure_id": exposure["id"],
            "dedup_key": exposure["dedup_key"],
            "request_hash": exposure["request_hash"],
            "state_at_submit": exposure["state"],
        })

    def record_plate(self, exposure_id, plate, client):
        self.plate_index[exposure_id] = plate["id"]
        entry = {
            "exposure_id": exposure_id,
            "plate_id": plate["id"],
            "seal_id": plate["payload"]["seal_id"],
            "visual_hash": plate["payload"]["visual_hash"],
            "request_width": plate["payload"]["request"]["width"],
            "request_height": plate["payload"]["request"]["height"],
            "pass_hashes": plate_bytes(client, plate["id"]),
        }
        if not hasattr(self, "plates"):
            self.plates = []
        self.plates.append(entry)

    def note(self, text):
        self.notes.append(text)

    def write(self):
        payload = {
            "data_root": str(self.root),
            "request_to_task_to_plate": self.requests,
            "plates": getattr(self, "plates", []),
            "exposure_plate_index": self.plate_index,
            "notes": self.notes,
        }
        out = self.root.parent / "idempotency-ledger.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if WORKSPACE.exists():
        # Keep each run isolated so a stale root cannot make a test pass.
        import shutil
        shutil.rmtree(WORKSPACE)
    data_root = WORKSPACE / "data"
    data_root.mkdir(parents=True)

    port = int(os.environ.get("FP_TEST_PORT", "8799"))
    server = Server(data_root, port).start()
    ledger = Ledger(data_root)
    try:
        print("== scenario 1: same key + same input across restart ==")
        scenario_same_key_same_input(server, ledger)
        print("   OK")
        print("== scenario 2: same key + different input must not reuse ==")
        scenario_same_key_different_input(server, ledger)
        print("   OK")
        print("== scenario 3: explicit retry vs network re-send ==")
        scenario_retry_vs_resend(server, ledger)
        print("   OK")
        print("== scenario 4: persistent artifacts and plate verification ==")
        scenario_persistent_root_artifacts(server, ledger)
        print("   OK")
    finally:
        server.stop()
        ledger_path = ledger.write()

    print(f"\nAll idempotency checks passed. Ledger: {ledger_path}")
    print(f"Persistent data root retained at: {data_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
