"""Retrieval-layer errors."""


class UnauthorizedRetrievalError(ValueError):
    """Raised when identity is required but user_id / tenant_id is missing."""