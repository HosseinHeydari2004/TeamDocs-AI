"""Domain exceptions. The API layer maps these to HTTP responses; messages are user-safe."""

from __future__ import annotations


class TeamDocsError(Exception):
    """Base class. ``code`` is a stable machine-readable identifier."""

    code = "internal_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(TeamDocsError):
    code, status_code = "unsupported_file_type", 415


class FileTooLargeError(TeamDocsError):
    code, status_code = "file_too_large", 413


class InvalidFileError(TeamDocsError):
    code, status_code = "invalid_file", 422


class EmptyDocumentError(TeamDocsError):
    code, status_code = "empty_document", 422


class DocumentNotFoundError(TeamDocsError):
    code, status_code = "document_not_found", 404


class IngestionError(TeamDocsError):
    code, status_code = "ingestion_failed", 500


class InvalidQueryError(TeamDocsError):
    code, status_code = "invalid_query", 422


class LLMNotConfiguredError(TeamDocsError):
    code, status_code = "llm_not_configured", 503


class LLMTimeoutError(TeamDocsError):
    code, status_code = "llm_timeout", 504


class LLMRateLimitError(TeamDocsError):
    code, status_code = "llm_rate_limited", 429


class LLMProviderError(TeamDocsError):
    code, status_code = "llm_provider_error", 502


class RetrievalError(TeamDocsError):
    code, status_code = "retrieval_failed", 500


class InvalidUrlError(TeamDocsError):
    code, status_code = "invalid_url", 422


class UrlNotAllowedError(TeamDocsError):
    code, status_code = "url_not_allowed", 422


class UrlFetchError(TeamDocsError):
    code, status_code = "url_fetch_failed", 502


class UrlTimeoutError(TeamDocsError):
    code, status_code = "url_timeout", 504


class AgentTimeoutError(TeamDocsError):
    code, status_code = "agent_timeout", 504


class AgentFailedError(TeamDocsError):
    code, status_code = "agent_error", 500
