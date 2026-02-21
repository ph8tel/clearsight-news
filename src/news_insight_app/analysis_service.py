import os
from typing import Any, Dict, List

import requests

from .tokenizer_utils import get_tokenizer_provider

QWEN_URL = os.getenv("QWEN_ANALYSIS_URL", "http://192.168.1.108:8000/v1/completions")
MISTRAL_URL = os.getenv("MISTRAL_ANALYSIS_URL", "http://192.168.1.108:8001/v1/completions")
PHI_URL = os.getenv("PHI_ANALYSIS_URL", "http://192.168.1.108:8002/v1/completions")

QWEN_TOKENIZER = "qwen2-7b"
MISTRAL_TOKENIZER = "mistralai/Mistral-7B-Instruct-v0.2"
TOKEN_CLIP_SIZE = 2000


def _truncate_text(text: str, limit: int = 4000) -> str:
	trimmed = text.strip()
	if len(trimmed) <= limit:
		return trimmed
	return trimmed[:limit].rstrip() + " ..."


def _normalize_token_ids(token_ids: List[Any], max_tokens: int) -> List[int]:
	result: List[int] = []
	for token in token_ids:
		if len(result) >= max_tokens:
			break
		if isinstance(token, int):
			result.append(token)
		else:
			result.append(abs(hash(str(token))) % (10 ** 6))
	return result


def _tokenize(text: str, tokenizer_name: str, max_tokens: int) -> List[int]:
	if not text:
		return []
	try:
		provider = get_tokenizer_provider()
		tokenizer = provider.get_tokenizer(tokenizer_name)
		token_ids = tokenizer.encode(text, add_special_tokens=False)
		return _normalize_token_ids(token_ids, max_tokens)
	except Exception:
		fallback_tokens = text.split()
		return _normalize_token_ids(fallback_tokens, max_tokens)


def _build_response(model_name: str, default_text: str) -> Dict[str, Any]:
	return {
		"model": model_name,
		"tokens_used": 0,
		"error": None,
		"text": default_text,
	}


def _call_completion(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
	response = requests.post(endpoint, json=payload, timeout=120)
	response.raise_for_status()
	return response.json()


def analyze_rhetoric(article_text: str) -> Dict[str, Any]:
	result = _build_response(
		"Qwen2-7B",
		"Rhetorical analysis unavailable for this story.",
	)
	result["analysis"] = result["text"]
	trimmed_text = _truncate_text(article_text)
	if not trimmed_text:
		result["error"] = "No content provided."
		return result

	tokens = _tokenize(trimmed_text, QWEN_TOKENIZER, TOKEN_CLIP_SIZE)
	prompt = f"""Analyze this news article for tone and rhetorical devices.

Article:
{trimmed_text}

Provide analysis in this format:
1. Overall Tone: (e.g., neutral, persuasive, alarmist, celebratory)
2. Sentiment: (positive, negative, or neutral with confidence score)
3. Rhetorical Devices Found:
   - List specific devices used (metaphors, appeals to emotion, repetition, loaded language, etc.)
   - Quote examples from the text
4. Bias Indicators: Any signs of bias or framing

Analysis:"""
	try:
		payload = {
			"prompt": prompt,
			"max_tokens": 500,
			"temperature": 0.3,
			"article_tokens": tokens,
			"tokenizer_model": QWEN_TOKENIZER,
		}
		body = _call_completion(QWEN_URL, payload)
		choices = body.get("choices") or []
		analysis_text = choices[0].get("text", "").strip() if choices else ""
		usage = body.get("usage", {})
		result.update({
			"analysis": analysis_text or result["text"],
			"tokens_used": usage.get("total_tokens", 0) or 0,
		})
		result["text"] = result["analysis"]
		return result
	except requests.RequestException as exc:
		result["error"] = f"Qwen request failed: {exc}"
		return result
	except (ValueError, KeyError) as exc:
		result["error"] = f"Qwen response invalid: {exc}"
		return result


def compare_article_texts(primary_text: str, reference_text: str) -> Dict[str, Any]:
	result = _build_response(
		"Mistral-7B",
		"Comparison unavailable for this pair of stories.",
	)
	result["comparison"] = result["text"]
	primary = _truncate_text(primary_text)
	reference = _truncate_text(reference_text)
	if not primary or not reference:
		result["error"] = "One of the articles was empty."
		return result

	primary_tokens = _tokenize(primary, MISTRAL_TOKENIZER, TOKEN_CLIP_SIZE)
	reference_tokens = _tokenize(reference, MISTRAL_TOKENIZER, TOKEN_CLIP_SIZE)
	prompt = f"""Compare these two news articles covering similar topics.

Article 1:
{primary}

Article 2:
{reference}

Provide comparison in this format:
1. Framing Differences: How does each article frame the story?
2. Tone Comparison: Compare the tone and emotional appeal
3. Source Selection: Note any differences in sources cited or perspectives included
4. Key Differences: What facts or angles does one include that the other doesn't?
5. Bias Assessment: Which article appears more balanced?

Comparison:"""
	try:
		payload = {
			"prompt": prompt,
			"max_tokens": 600,
			"temperature": 0.3,
			"primary_tokens": primary_tokens,
			"reference_tokens": reference_tokens,
			"tokenizer_model": MISTRAL_TOKENIZER,
		}
		body = _call_completion(MISTRAL_URL, payload)
		choices = body.get("choices") or []
		comparison_text = choices[0].get("text", "").strip() if choices else ""
		usage = body.get("usage", {})
		result.update({
			"comparison": comparison_text or result["text"],
			"tokens_used": usage.get("total_tokens", 0) or 0,
		})
		result["text"] = result["comparison"]
		return result
	except requests.RequestException as exc:
		result["error"] = f"Mistral request failed: {exc}"
		return result
	except (ValueError, KeyError) as exc:
		result["error"] = f"Mistral response invalid: {exc}"
		return result