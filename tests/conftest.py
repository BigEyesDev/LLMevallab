"""Shared fixtures for provider tests."""

SAMPLE_PROMPTS = {
    "extraction": {
        "system": "Extract JSON.",
        "user": "Document:\n{text}",
    },
    "translation": {
        "system": "Translate.",
        "user": "Translate from {source_language}:\n{text}",
    },
    "summarisation": {
        "system": "Summarise as JSON.",
        "user": "Text:\n{text}",
    },
}

SAMPLE_DOCUMENT = {
    "doc_id": "doc-1",
    "source_language": "de",
    "raw_text": "Das Parlament tagte heute in Brüssel.",
    "source": "test",
}

GEMINI_CONFIG = {
    "model_id": "gemini-2.5-flash",
    "temperature": 0.1,
    "max_output_tokens": 2048,
    "max_document_length": 2000,
    "pricing": {"input_per_1m": 0.075, "output_per_1m": 0.30},
}

CLAUDE_CONFIG = {
    "model_id": "claude-sonnet-4-6",
    "temperature": 0.1,
    "max_output_tokens": 2048,
    "max_document_length": 2000,
    "pricing": {"input_per_1m": 3.00, "output_per_1m": 15.00},
}

OPENAI_CONFIG = {
    "model_id": "gpt-4o-mini",
    "base_url": None,
    "temperature": 0.1,
    "max_output_tokens": 2048,
    "max_document_length": 2000,
    "pricing": {"input_per_1m": 0.15, "output_per_1m": 0.60},
}

OPENROUTER_CONFIG = {
    "model_id": "meta-llama/llama-3.3-70b-instruct",
    "base_url": "https://openrouter.ai/api/v1",
    "temperature": 0.1,
    "max_output_tokens": 2048,
    "max_document_length": 2000,
    "pricing": {"input_per_1m": 0.13, "output_per_1m": 0.40},
}
