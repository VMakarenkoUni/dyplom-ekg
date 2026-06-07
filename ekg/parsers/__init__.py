"""Format ingestion. Wraps the legacy parser.py converters."""

from ekg.parsers.detect import detect_format, parse_to_xml, supported_formats

__all__ = ["detect_format", "parse_to_xml", "supported_formats"]
