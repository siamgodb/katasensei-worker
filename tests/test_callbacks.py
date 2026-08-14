from __future__ import annotations

import hashlib
import hmac

from app.callbacks import sign


class TestSigning:
    def test_signs_the_exact_bytes_that_will_be_sent(self) -> None:
        # Over the body rather than over a summary of it: whatever is not
        # signed can be changed in transit without the signature noticing.
        body = b'{"reports":[],"progress":{"status":"running"}}'

        assert sign("shh", "1755230000", body) == hmac.new(
            b"shh", b"1755230000." + body, hashlib.sha256
        ).hexdigest()

    def test_a_changed_body_changes_the_signature(self) -> None:
        original = b'{"reports":[{"move_number":1}]}'
        tampered = b'{"reports":[{"move_number":2}]}'

        assert sign("shh", "1755230000", original) != sign("shh", "1755230000", tampered)

    def test_a_different_secret_changes_the_signature(self) -> None:
        body = b"{}"

        assert sign("one", "1755230000", body) != sign("two", "1755230000", body)

    def test_the_timestamp_is_inside_the_signature(self) -> None:
        """Not merely sent beside it.

        This endpoint faces the internet now — the worker is on RunPod and has
        no fixed address to sit behind. A signature that covers only the body
        stays valid forever, so one captured final batch could be replayed at
        any point to mark a running review finished. Moving the timestamp has
        to break the signature, or it protects nothing.
        """
        body = b'{"progress":{"final":true}}'

        assert sign("shh", "1755230000", body) != sign("shh", "1755240000", body)

    def test_the_boundary_cannot_be_shifted(self) -> None:
        """A timestamp and a body that concatenate to the same bytes must not
        produce the same signature, or the separator is decoration."""
        assert sign("shh", "17552", b"30000.x") != sign("shh", "1755230000", b"x")
