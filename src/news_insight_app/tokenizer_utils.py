from __future__ import annotations


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
