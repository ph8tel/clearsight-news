from __future__ import annotations

import threading
from typing import Any, Dict


def create_fallback_tokenizer():
	class _FallbackTokenizer:
		model_max_length = 512

		@staticmethod
		def num_special_tokens_to_add(pair: bool = False) -> int:
			return 2

		@staticmethod
		def encode(text: str, add_special_tokens: bool = False):
			return [token for token in text.split() if token]

		@staticmethod
		def decode(tokens, skip_special_tokens: bool = True) -> str:
			return " ".join(tokens)

	return _FallbackTokenizer()


class TokenizerProvider:
	"""Centralized tokenizer management with caching and fallback."""

	def __init__(self) -> None:
		self._tokenizers: Dict[str, Any] = {}

	def get_tokenizer(self, model_name: str) -> Any:
		"""
		Get or load a tokenizer for the given model.
		Caches tokenizers by model name.
		"""
		if model_name not in self._tokenizers:
			self._tokenizers[model_name] = create_fallback_tokenizer()
		return self._tokenizers[model_name]

	def count_tokens(self, text: str, model_name: str) -> int:
		"""Count tokens in text using the specified model's tokenizer."""
		if not text:
			return 0
		tokenizer = self.get_tokenizer(model_name)
		return len(tokenizer.encode(text, add_special_tokens=False))


# Global tokenizer provider instance (lazily initialized under lock)
_provider: TokenizerProvider | None = None
_provider_lock = threading.Lock()


def get_tokenizer_provider() -> TokenizerProvider:
	"""
	Get the global tokenizer provider instance, creating it on first access.

	This function lazily initializes a TokenizerProvider using double-checked
	locking to ensure thread-safety in multi-threaded environments like Flask.
	"""
	global _provider
	with _provider_lock:
		if _provider is None:
			_provider = TokenizerProvider()
	return _provider


def set_tokenizer_provider(provider: TokenizerProvider | None) -> None:
	"""
	Set the global tokenizer provider instance.

	This is useful for testing or dependency injection. Passing None will clear
	the current provider so that the next call to get_tokenizer_provider()
	will create a new TokenizerProvider instance.
	"""
	global _provider
	_provider = provider
