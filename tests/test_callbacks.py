from __future__ import annotations

import hashlib
import hmac

from app.callbacks import sign


class TestSigning:
    def test_signs_the_exact_bytes_that_will_be_sent(self) -> None:
        # Over the body rather than over a summary of it: whatever is not
        # signed can be changed in transit without the signature noticing.
        body = b'{"reports":[],"progress":{"status":"running"}}'

        assert sign("shh", body) == hmac.new(b"shh", body, hashlib.sha256).hexdigest()

    def test_a_changed_body_changes_the_signature(self) -> None:
        original = b'{"reports":[{"move_number":1}]}'
        tampered = b'{"reports":[{"move_number":2}]}'

        assert sign("shh", original) != sign("shh", tampered)

    def test_a_different_secret_changes_the_signature(self) -> None:
        body = b"{}"

        assert sign("one", body) != sign("two", body)
