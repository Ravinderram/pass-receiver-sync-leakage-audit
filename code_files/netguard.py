"""Block outbound network access during a local-data run.

Once config.IDSSE_LOCAL_DATA_DIR is set, the pipeline needs no network at all.
This module makes that enforceable rather than aspirational: `arm()` patches
socket.socket.connect so any attempt to open a remote connection raises
NetworkBlocked with the host that was attempted.

It is deliberately blunt. Loopback is allowed (a local database or a notebook
kernel keeps working); everything else is refused. If you deliberately need the
network in a local-data run, call netguard.disarm() around that code.

`banner()` prints the two lines that belong at the top of every real run:

    DATA SOURCE: LOCAL IDSSE (/path/to/IDSSE)
    NETWORK DOWNLOAD: DISABLED
"""

import socket

import config as C

BLOCKED_HOSTS = ("figshare.com", "ndownloader.figshare.com", "doi.org",
                 "datadryad.org", "zenodo.org")
_ARMED = False
_ORIGINAL = socket.socket.connect
ATTEMPTS = []


class NetworkBlocked(RuntimeError):
    """Raised when a local-data run tries to reach the network."""


def _allowed(address):
    try:
        host = address[0]
    except Exception:
        return True                      # unix sockets and the like
    return host in ("127.0.0.1", "::1", "localhost", "0.0.0.0")


def arm(force=False):
    """Install the guard. No-op unless a local dataset is configured."""
    global _ARMED
    if _ARMED or not (force or C.LOCAL_ONLY):
        return False

    def guarded(self, address, *args, **kwargs):
        if not _allowed(address):
            ATTEMPTS.append(address)
            raise NetworkBlocked(
                f"network access blocked: this run is configured to read the "
                f"IDSSE dataset from {C.IDSSE_LOCAL_DIR}, so nothing may be "
                f"downloaded. Attempted connection: {address}. "
                f"If a match is missing, add it to that folder; the pipeline "
                f"will not fetch it.")
        return _ORIGINAL(self, address, *args, **kwargs)

    socket.socket.connect = guarded
    _ARMED = True
    return True


def disarm():
    global _ARMED
    socket.socket.connect = _ORIGINAL
    _ARMED = False


def armed():
    return _ARMED


def status():
    return {"local_only": C.LOCAL_ONLY, "armed": _ARMED,
            "blocked_attempts": [str(a) for a in ATTEMPTS]}


def banner():
    if C.LOCAL_ONLY:
        print(f"DATA SOURCE: LOCAL IDSSE ({C.IDSSE_LOCAL_DIR})")
        print(f"NETWORK DOWNLOAD: {'DISABLED' if _ARMED else 'not yet armed'}")
    else:
        print("DATA SOURCE: not configured "
              "(set IDSSE_LOCAL_DATA_DIR in config.py for local-only mode)")
        print("NETWORK DOWNLOAD: would be attempted by floodlight")
