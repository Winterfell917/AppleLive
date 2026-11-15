"""Show raw UDP packets sent by Sensor Read before starting a capture."""

from __future__ import annotations

import argparse
import json
import socket
import time


def describe_line(raw_line: bytes) -> str:
    try:
        event = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return f"non-JSON payload ({len(raw_line)} bytes)"
    if not isinstance(event, dict):
        return f"JSON {type(event).__name__} ({len(raw_line)} bytes)"
    source = event.get("source", "<missing>")
    sensor = event.get("sensor", "<missing>")
    timestamp = event.get("timestampUnixNs", "<missing>")
    sequence = event.get("sequenceNumber", "<missing>")
    values = event.get("values")
    value_keys = ", ".join(sorted(values)[:6]) if isinstance(values, dict) else "<not an object>"
    return (
        f"source={source!r}, sensor={sensor!r}, sequence={sequence!r}, "
        f"timestampUnixNs={timestamp!r}, values=[{value_keys}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, args.port))
        sock.settimeout(0.5)
        deadline = time.monotonic() + args.seconds
        packet_count = 0
        line_count = 0
        print(f"Listening for raw UDP on udp://{args.host}:{args.port} for {args.seconds:.0f} s")
        while time.monotonic() < deadline:
            try:
                packet, address = sock.recvfrom(65535)
            except socket.timeout:
                continue
            packet_count += 1
            lines = packet.splitlines() or [packet]
            for raw_line in lines:
                if not raw_line:
                    continue
                line_count += 1
                print(f"[{packet_count}/{line_count}] from {address[0]}:{address[1]}: {describe_line(raw_line)}")
        print(f"Finished: {packet_count} UDP packets, {line_count} non-empty event lines.")
        if packet_count == 0:
            print("No packet reached this computer. Check the iPhone sender settings or network path.")


if __name__ == "__main__":
    main()
