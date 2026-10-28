"""Errors shared across Alebrije."""


class LlmError(Exception):
    """The local model server could not be reached or returned garbage.

    `kind` is one of: dns | connect | timeout | http | parse | unsupported.
    """

    def __init__(self, kind: str, detail: str, status: int | None = None):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.status = status
