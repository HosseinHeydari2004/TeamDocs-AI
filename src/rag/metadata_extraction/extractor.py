from __future__ import annotations
import logging
import time

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .schema import SemanticMetadata
from .prompts import METADATA_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


def build_metadata_context(
        text: str,
        max_chars: int = 12_000,
) -> str:
    """Build a bounded context for document-level metadata extraction."""

    if max_chars < 256:
        raise ValueError("max_chars must be at least 256.")

    text = text.strip()

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    markers = (
        "[BEGINNING OF DOCUMENT]\n",
        "\n\n[MIDDLE OF DOCUMENT]\n",
        "\n\n[END OF DOCUMENT]\n",
    )

    content_budget = max_chars - sum(
        len(marker) for marker in markers
    )

    part_size = content_budget // 3

    beginning = text[:part_size]

    middle_start = (len(text) - part_size) // 2
    middle = text[middle_start:middle_start + part_size]

    ending = text[-part_size:]

    return (
        f"{markers[0]}{beginning}"
        f"{markers[1]}{middle}"
        f"{markers[2]}{ending}"
    )


class MetadataExtractor:
    """Extract and attach semantic metadata to LangChain documents."""

    def __init__(
            self,
            llm: BaseChatModel,
            *,
            max_chars: int = 5000,
            max_retries: int = 2,
            retry_delay: float = 1.0,
            overwrite_existing: bool = False,
    ) -> None:
        if max_chars < 256:
            raise ValueError("max_chars must be at least 256.")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if retry_delay < 0:
            raise ValueError("retry_delay cannot be negative.")

        self.max_chars = max_chars
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.overwrite_existing = overwrite_existing

        self.prompt = ChatPromptTemplate.from_template(
            METADATA_EXTRACTION_PROMPT
        )

        self.structured_llm = llm.with_structured_output(
            SemanticMetadata
        )

        self.chain = self.prompt | self.structured_llm

    def extract_document(
            self,
            document: Document,
    ) -> Document:
        """Extract semantic metadata from a single document."""

        content = document.page_content

        if not content or not content.strip():
            logger.warning(
                "Skipping metadata extraction for empty document_id=%s",
                document.id,
            )
            return document

        context = build_metadata_context(
            content,
            max_chars=self.max_chars,
        )

        semantic_metadata = self._invoke_with_retry(
            context=context,
            document_id=document.id,
        )

        extracted_metadata = semantic_metadata.model_dump(
            exclude_none=True,
        )

        metadata = document.metadata.copy()

        for key, value in extracted_metadata.items():
            if self.overwrite_existing or key not in metadata:
                metadata[key] = value

        return Document(
            id=document.id,
            page_content=document.page_content,
            metadata=metadata,
        )

    def _invoke_with_retry(
            self,
            *,
            context: str,
            document_id: str | None,
    ) -> SemanticMetadata:
        """Invoke the extraction chain with bounded retries."""

        for attempt in range(self.max_retries + 1):
            try:
                result = self.chain.invoke(
                    {"document_content": context}
                )

                if isinstance(result, SemanticMetadata):
                    return result

                return SemanticMetadata.model_validate(result)

            except Exception:
                logger.exception(
                    "Metadata extraction failed for document_id=%s "
                    "(attempt %d/%d)",
                    document_id,
                    attempt + 1,
                    self.max_retries + 1,
                )

                if attempt >= self.max_retries:
                    raise

                delay = self.retry_delay * (2 ** attempt)

                time.sleep(delay)

        raise RuntimeError("Metadata extraction failed unexpectedly.")

    def extract(
            self,
            documents: Sequence[Document],
            *,
            continue_on_error: bool = False,
    ) -> list[Document]:
        """Extract metadata from multiple documents."""

        results: list[Document] = []

        for document in documents:
            try:
                result = self.extract_document(document)
                results.append(result)

            except Exception:
                if not continue_on_error:
                    raise

                logger.error(
                    "Keeping original document after metadata "
                    "extraction failure. document_id=%s",
                    document.id,
                )

                results.append(document)

        return results

    def extract_by_source(
            self,
            documents: Sequence[Document],
            *,
            group_by: str = "source",
            continue_on_error: bool = False,
    ) -> list[Document]:
        """Extract metadata once per source file and copy it to all its chunks."""

        # 1) Group chunk indexes by source
        keys: list[str] = []
        groups: dict[str, list[int]] = {}

        for index, document in enumerate(documents):
            key = document.metadata.get(group_by)

            if key is None:
                # chunk without source -> treated as its own group
                key = f"__no_source__{index}"

            key = str(key)
            keys.append(key)
            groups.setdefault(key, []).append(index)

        # 2) One LLM call per source
        group_metadata: dict[str, dict] = {}

        for key, indexes in groups.items():
            full_text = "\n\n".join(
                documents[i].page_content
                for i in indexes
                if documents[i].page_content
                and documents[i].page_content.strip()
            )

            context = build_metadata_context(
                full_text,
                max_chars=self.max_chars,
            )

            if not context:
                continue

            try:
                semantic_metadata = self._invoke_with_retry(
                    context=context,
                    document_id=key,
                )
            except Exception:
                if not continue_on_error:
                    raise

                logger.error(
                    "Keeping original chunks after metadata "
                    "extraction failure. source=%s",
                    key,
                )
                continue

            group_metadata[key] = semantic_metadata.model_dump(
                exclude_none=True,
            )

        # 3) Copy the source-level metadata to every chunk
        results: list[Document] = []

        for index, document in enumerate(documents):
            extracted_metadata = group_metadata.get(keys[index])

            if not extracted_metadata:
                results.append(document)
                continue

            metadata = document.metadata.copy()

            for meta_key, value in extracted_metadata.items():
                if self.overwrite_existing or meta_key not in metadata:
                    metadata[meta_key] = value

            results.append(
                Document(
                    id=document.id,
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return results
