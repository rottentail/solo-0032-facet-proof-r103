"""Minimal worker process that claims one queued exposure and reports it failed.

It mirrors the ``Fault`` branch of ``facetproof.worker.Worker.run_one`` using
only the service's public, lease-checked ``Exposures`` API (``claim`` then
``fail``): a real lease is taken, the attempt counter advances, and the normal
``failed`` event is written. It is therefore a genuine state transition
recorded by the service itself rather than a direct edit of the SQLite file.

It is intentionally a *separate process* so the end-to-end tests can exercise a
real worker failing against the same persistent data root, after which the HTTP
service is restarted and the explicit ``/retry`` path is verified.

Usage::

    python3 failure_worker.py <data-root> [expected-exposure-id]
"""

import json
import sys

from facetproof.application import Application


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: failure_worker.py <data-root> [exposure-id]")
    root = sys.argv[1]
    only = sys.argv[2] if len(sys.argv) > 2 else None

    app = Application(root)
    exposure = app.exposures.claim()
    if exposure is None:
        raise SystemExit("no queued exposure was available to claim")
    if only is not None and exposure["id"] != only:
        raise SystemExit(
            f"claimed {exposure['id']!r} but expected {only!r}"
        )

    # Same lease-checked call the production worker makes when rendering raises
    # a Fault; an expired or lost lease is rejected inside fail().
    result = app.exposures.fail(
        exposure["id"], exposure["token"], "injected_render_failure"
    )
    print(
        json.dumps(
            {
                "exposure_id": exposure["id"],
                "attempt": exposure["attempt"],
                "state": result["state"],
                "error": result["error"],
            }
        )
    )


if __name__ == "__main__":
    main()
