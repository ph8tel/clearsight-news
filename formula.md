## sentiment determination formula
This uses six dimensions, each normalized to –1..1:
Dimensions
- Tone polarity
- Emotion polarity
- Emotion intensity
- Rhetorical polarity
- Loaded language density
- Certainty vs speculation
Formula
\mathrm{sentiment}=0.25\cdot \mathrm{tone}+0.25\cdot \mathrm{emotion\_ polarity}+0.15\cdot \mathrm{emotion\_ intensity}+0.15\cdot \mathrm{rhetoric\_ polarity}+0.10\cdot \mathrm{loaded\_ language}+0.10\cdot \mathrm{certainty}
Why this works
- Tone alone is weak → only 25% weight
- Emotions matter more → 40% combined
- Rhetoric matters → 15%
- Loaded language + certainty → subtle but important
This gives you a smooth, expressive sentiment score from –1 to +1.

3. How to compute each dimension
Tone polarity
- positive → +1
- neutral → 0
- negative → –1

Emotion polarity
\mathrm{emotion\_ polarity}=\frac{(\mathrm{joy}+\mathrm{trust})-(\mathrm{anger}+\mathrm{fear}+\mathrm{sadness}+\mathrm{disgust})}{6}

Emotion intensity
\mathrm{emotion\_ intensity}=\frac{\mathrm{sum\  of\  all\  emotion\  scores}}{8}
This captures how “emotional” the article is, regardless of polarity.

Rhetorical polarity
\mathrm{rhetoric\_ polarity}=\frac{(\mathrm{supportive}+\mathrm{analytical})-(\mathrm{alarmist}+\mathrm{dismissive}+\mathrm{sarcastic})}{5}

Loaded language
Model outputs 0–1.
Normalize to –1..1 by:
\mathrm{loaded\_ language}=(x\cdot -1)
Loaded language always pushes negative.

Certainty
\mathrm{certainty}=\mathrm{certainty\_ score}-\mathrm{speculation\_ score}

4. JSON Schema for the 7B model
Here’s the schema your 7B model should output:
{
  "tone": "positive | neutral | negative",
  "emotions": {
    "joy": 0.0,
    "trust": 0.0,
    "fear": 0.0,
    "anger": 0.0,
    "sadness": 0.0,
    "anticipation": 0.0,
    "disgust": 0.0,
    "surprise": 0.0
  },
  "rhetoric": {
    "analytical": 0.0,
    "supportive": 0.0,
    "persuasive": 0.0,
    "alarmist": 0.0,
    "dismissive": 0.0,
    "sarcastic": 0.0
  },
  "loaded_language": 0.0,
  "certainty": {
    "certainty": 0.0,
    "speculation": 0.0
  }
}


All values 0–1.
The backend computes the final sentiment score
