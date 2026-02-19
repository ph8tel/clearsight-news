from .sentiment_service import SentimentService
from .tokenizer_utils import create_fallback_tokenizer

_sentiment_service = None
_tokenizer = None

# Mock news data - in real app, this would come from an API
MOCK_NEWS = [
	{
		"id": 1,
		"title": "What to know about how the SAVE America Act could change voting",
		"content": """
        Ahead of the midterm elections, Republicans are again pushing for legislation that requires documentary proof of U.S. citizenship to vote.\n\nThe Trump-backed Safeguard American Voter Eligibility Act, or the SAVE America Act, seeks to address the president's longstanding demands to \"fix\" U.S. elections that he says are \"rigged\" and \"stolen,\" despite no evidence of widespread voter fraud.
		The Save America Act is an expanded version of legislation that the House passed twice in as many years. It failed to clear the Senate in both cases.
Every version of the SAVE Act has had a common throughline: Requiring Americans to provide proof of citizenship when registering to vote in federal elections. For most people, this would likely mean a passport or birth certificate.
While the bill lists other eligible documents that can prove citizenship, they may not meet the measure's requirements, said Sean Morales-Doyle, director of the voting rights and elections program at the Brennan Center for Justice.
One of those documents is an ID that is compliant with the provisions of the REAL ID Act of 2005 and "indicates the applicant is a citizen of the United States."
REAL IDs are available to both citizens and noncitizens, Morales-Doyle said
No one state's REAL ID explicitly marks citizenship status, nor do most state-issued driver's licenses.
The act also requires a government-issued photo ID to vote in person, and a copy of an eligible photo ID both when requesting and submitting an absentee ballot.
There are other provisions in this latest iteration of the SAVE Act, such as requiring mail-in applicants to provide proof of citizenship in person and mandating that states take steps to make sure only U.S. citizens are registered to vote.
The bill would also add criminal penalties for any election official who registers an applicant who fails to provide documentary proof of citizenship. Those penalties apply even if an individual is a U.S. citizen, said Rachel Orey, director of the Bipartisan Policy Center's Elections Project.
This is one of the "most concerning gray areas" in the SAVE America Act because it gives "vague discretion" to an election official who could face a criminal penalty, they said
This "risks creating an environment where election officials are almost overly compliant, taking a very hyper interpretation of the statute, which might mean that this process that is meant to be a fail-safe doesn't actually operate like one in practice because election officials don't have the protection that they would need to make that decision on a case-by-case basis," they said.
A second bill, called the Make Elections Great Again Act, also requires documentation of citizenship to register to vote, along with photo ID provisions. But it also adds an array of other election changes, such as banning universal voting by mail.
Orey said all of the bills under consideration are "unfunded mandates" that need time and resources to implement. A one-year lead is the "optimal" amount of time for states to implement a new policy or procedure, according to recommendations released by The Bipartisan Policy Center following the 2020 election.
Given the range of changes suggested by the SAVE America Act and other bills, Orey said a longer lead time would be warranted.
"We don't recommend that states or the federal government implement election administration policy changes in a federal election year, let alone a policy change that would be as significant as this," they said.
		""",
		"url": "https://www.pbs.org/newshour/politics/how-the-save-america-act-would-make-major-changes-to-voting",
		"source": "NPR Politics",
		"published_at": "2026-02-18T12:30:00Z"
	},
	{
		"id": 2,
		"title": "House passes SAVE Act to require voters to show ID",
		"content": """
The House of Representatives narrowly passed the SAVE America Act on Wednesday, but it faces a tough sell in the Senate.
The House approved the measure on Wednesday by a vote of 218-213, with one Democrat voting in favor of the proposed law that would require voters to provide a birth certificate or passport to prove their citizenship status when registering to vote and produce a valid photo ID to vote.
“It’s just common sense. Americans need an ID to drive, to open a bank account, to buy cold medicine [and] to file for government assistance,” House Speaker Mike Johnson, R-La., told media. “So, why would voting be any different than that?”
Democrats oppose the measure, which Senate Minority Leader Chuck Schumer, D-N.Y., called “Jim Crow 2.0.”
House Minority Leader Hakeem Jeffries, D-N.Y., called the proposed voting law a “desperate effort by Republicans to distract” without saying from what.
“The so-called SAVE Act is not about voter identification,” Jeffries continued. “It is about voter suppression, and they have zero credibility on this issue.” Rep. Henry Cuellar, D-Texas, was the lone Democrat to vote in favor of the measure, which now goes to the Senate for consideration. Rep. Chip Roy, R-Texas, sponsored the bill.
Although Senate Republicans have a simple majority in the upper chamber, they likely lack the 60 votes needed to overcome the Senate’s filibuster rule.
Senate Majority Leader John Thune, R-S.D., on Tuesday said he supports the proposed act but does not have the votes needed to change the filibuster rule to pass it with a simple majority.
The GOP controls 53 Senate seats, while Democrats control 47, including two held by independents who sit with the Senate Democratic Party’s caucus.
Some Republicans have suggested requiring a standing filibuster, which would require those opposing proposed legislation to physically engage in a non-stop filibuster instead of just announcing their intent to do so.""",
		"url": "https://www.breitbart.com/news/house-passes-save-act-to-require-voters-to-show-id/",
		"source": "Breitbart News",
		"published_at": "2026-02-18T12:45:00Z"
	}
]


def generate_summary(text, max_sentences=2):
	"""Generate a concise summary using simple sentence extraction"""
	sentences = [s.strip() for s in text.split('.') if s.strip()]
	if len(sentences) <= max_sentences:
		return text
	return '. '.join(sentences[:max_sentences]) + '. ..'


def _get_sentiment_service():
	global _sentiment_service
	if _sentiment_service is None:
		_sentiment_service = SentimentService()
	return _sentiment_service


def _get_tokenizer():
	global _tokenizer
	if _tokenizer is None:
		from transformers import AutoTokenizer

		model_name = _get_sentiment_service().model_name
		try:
			_tokenizer = AutoTokenizer.from_pretrained(
				model_name,
				use_fast=True,
				local_files_only=True,
			)
		except Exception:
			_tokenizer = create_fallback_tokenizer()
	return _tokenizer

def _get_max_chunk_tokens(tokenizer, max_tokens):
	model_max = getattr(tokenizer, "model_max_length", max_tokens)
	if model_max is None or model_max > 100000:
		model_max = max_tokens
	special_tokens = 0
	if hasattr(tokenizer, "num_special_tokens_to_add"):
		special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
	available = max(1, model_max - special_tokens)
	return min(max_tokens, available)


def _split_long_sentence(sentence_text, tokenizer, max_chunk_tokens):
	chunk_tokens = tokenizer.encode(sentence_text, add_special_tokens=False)
	chunks = []
	for i in range(0, len(chunk_tokens), max_chunk_tokens):
		chunk_ids = chunk_tokens[i:i + max_chunk_tokens]
		chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
		if chunk_text and not chunk_text.endswith("."):
			chunk_text += "."
		if chunk_text:
			chunks.append(chunk_text)
	return chunks


def _chunk_text(text, max_tokens=510):
	"""
	Chunk text into manageable pieces for analysis.
	Uses a tokenizer for accurate token counts and keeps sentence boundaries.
	"""
	if not text:
		return []

	tokenizer = _get_tokenizer()
	max_chunk_tokens = _get_max_chunk_tokens(tokenizer, max_tokens)

	# Split by sentences and group into chunks based on token counts.
	sentences = [s.strip() for s in text.split('.') if s.strip()]
	chunks = []
	current_sentences = []
	current_token_count = 0

	for sentence in sentences:
		sentence_text = f"{sentence}."
		sentence_token_count = len(
			tokenizer.encode(sentence_text, add_special_tokens=False)
		)

		if sentence_token_count > max_chunk_tokens:
			if current_sentences:
				chunks.append(" ".join(current_sentences).strip())
				current_sentences = []
				current_token_count = 0
			chunks.extend(
				_split_long_sentence(sentence_text, tokenizer, max_chunk_tokens)
			)
			continue

		if current_token_count + sentence_token_count <= max_chunk_tokens:
			current_sentences.append(sentence_text)
			current_token_count += sentence_token_count
		else:
			chunks.append(" ".join(current_sentences).strip())
			current_sentences = [sentence_text]
			current_token_count = sentence_token_count

	if current_sentences:
		chunks.append(" ".join(current_sentences).strip())

	return chunks


def analyze_sentiment(text):
	"""Sentiment analysis using the configured model service"""
	service = _get_sentiment_service()
	if not text:
		return service.analyze("")
	
	# For very long texts, we'll analyze sentiment on chunks and combine results
	# This avoids the sequence length limitation
	chunks = _chunk_text(text, max_tokens=510)
	if not chunks:
		return service.analyze("")
	
	if len(chunks) == 1:
		# If only one chunk, analyze directly
		return service.analyze(chunks[0])
	else:
		# For multiple chunks, analyze each and combine results
		# Simple approach: analyze first chunk for overall sentiment
		# In a production system, you'd want more sophisticated aggregation
		return service.analyze(chunks[0])


def extract_keywords(text, num_keywords=5):
	"""Simple keyword extraction"""
	# This is a basic approach - in production, you'd use NLTK or spaCy
	words = text.lower().split()
	# Remove common stop words
	stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'}
	filtered_words = [word.strip('.,!?";()[]{}') for word in words if word not in stop_words and len(word) > 3]

	# Simple frequency count
	word_freq = {}
	for word in filtered_words:
		word_freq[word] = word_freq.get(word, 0) + 1

	# Return top keywords
	sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
	return [word for word, freq in sorted_words[:num_keywords]]


def get_article_insights(text):
	"""Extract various insights from the article"""
	return {
		"word_count": len(text.split()),
		"sentence_count": len([s for s in text.split('.') if s.strip()]),
		"keywords": extract_keywords(text),
		"reading_time_minutes": max(1, len(text.split()) // 200)  # Average 200 words per minute
	}
