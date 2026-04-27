# Model Card — SoundFit 2.0

## 1. Name

**SoundFit 2.0** — a conversational, agentic music recommender. SoundFit 1.0
was the original rule-based, hand-curated 10-song demo (preserved as the
deterministic ranking core).

## 2. Intended use

A demo and learning artifact. Not a production recommender. Use cases:

- Showing an agentic LLM pattern wired end-to-end: structured intent
  extraction → deterministic tool call → grounded explanation → memory
- Demonstrating local-first model usage (`qwen2.5:3b` via Ollama) with
  property-based evaluation
- A teaching artifact for transparent feature scoring

It is *not* a substitute for Spotify or any production recommender. The
catalog is sampled and the LLM is small.

## 3. How it works (plain language)

1. The user types a natural-language request like *"aux rap songs for the
   car"* or *"songs by The Weeknd"*.
2. A small local LLM (`qwen2.5:3b`) reads the message and fills out a
   structured form: genre, mood, energy level, valence, danceability,
   acousticness, optional artist. If it doesn't know, it leaves blanks.
3. The structured form is fed into the original rule-based scorer
   (preserved untouched from SoundFit 1.0). The scorer ranks every track
   in the catalog by how closely each feature matches.
4. The top results go *back* to the same LLM, which writes one short
   sentence per song explaining why it's a good match — citing real
   feature values from the data, not invented ones.
5. The chat UI displays the ranked songs with feature bars and the
   sentence-long explanation. Optionally pulls album art from Spotify.

## 4. Data

**Source.** ~5,000 tracks sampled from the public HuggingFace dataset
[`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
(~114k total). The audio features are Spotify-derived: `energy`, `valence`,
`danceability`, `acousticness`, `tempo_bpm`, `instrumentalness`,
`popularity`. We don't use Spotify's API for features (their
`/audio-features` endpoint was deprecated for new apps in Nov 2024) — the
HuggingFace dump preserves the same numbers.

**Sampling.** Popularity-weighted (quadratic) stratified sampling: each of
the 12 genre families gets at least 100 tracks, and within each genre,
high-popularity tracks are ~50× more likely to be picked than low-popularity
ones. This deliberately biases the working set toward household-name
artists (Kendrick Lamar, Bad Bunny, Adele, Billie Eilish, Taylor Swift,
etc.) so user queries about specific famous artists usually find matches.

**Genre normalization.** Spotify's raw genre field is granular and messy
("japanese-trap", "deep-house", "synth-pop"). We map each into a 12-family
taxonomy: `pop, rock, metal, hip-hop, electronic, jazz, classical, country,
r&b, latin, folk, reggae`. Anything unmapped becomes "other" and is
filtered out of the working catalog. The original raw genre is kept in a
`genre_raw` column for display.

**Mood derivation.** Spotify has no mood field, so we derive one
deterministically from audio features (see [`derive_mood`](src/data.py)).
The mapping in plain language:

- High instrumentalness + low energy → `focused`
- Low valence + high energy → `intense`
- Low valence + low energy → `moody`
- Low valence + high acousticness → `relaxed`
- High valence + high energy → `happy`
- High valence + low/mid energy → `chill`
- Falls back to `chill` for ambiguous cases

This is a transparent rule, not a black box; you can read it in 10 lines
of Python and predict the mood for any input.

**Offline fallback.** If HuggingFace is unreachable, the app uses
[data/songs_sample.csv](data/songs_sample.csv) — 88 hand-curated tracks
covering all 12 genres and 6 moods. Tests run on this sample.

## 5. Strengths

- **Real agentic pattern, not a chat wrapper.** Plans (intent parsing),
  acts (calls the deterministic scorer as a tool), grounds output (citing
  real feature values), remembers (multi-turn refinement), degrades
  gracefully (Ollama unavailable → rule-based mode).
- **Hard integrity guarantees, not just prompts:**
  - The LLM cannot return a song ID outside the candidate set —
    enforced by post-validation, not by hoping the prompt is followed.
  - The LLM cannot leak an artist from prior chat history — `artist` is
    dropped if its name doesn't appear in the *current* user message.
  - OOV genres (e.g. "polka") are nulled, not coerced into a similar
    catalog genre.
- **Slang-aware.** The system prompt teaches qwen2.5:3b common music
  slang ("aux songs", "vibing", "hype", "study music", "heartbreak") with
  feature mappings. Lets a 3B model handle "give me aux rap" correctly.
- **Measurable.** 30 YAML eval cases across 3 types (behavioral, golden,
  refinement) with property-based assertions. CSV + summary JSON
  outputs feed the Streamlit Eval tab. Last run: 30/30 with 0
  hallucinated IDs.
- **Local-first.** Runs entirely offline on `qwen2.5:3b` via Ollama. No
  API keys required for the core experience. `OLLAMA_HOST` env var lets
  you point at a remote machine without code changes.

## 6. Limitations and bias

- **Small model.** qwen2.5:3b is ~2GB and runs on CPU. It handles the
  intended queries well, but subtle or unusual phrasing may slip past
  the slang dictionary. The hard validators catch most failure modes
  but they don't make the LLM smarter — only safer.
- **Catalog is 5,000 of ~114k.** Even with popularity weighting, some
  artists won't be present. The agent surfaces a `note` when an artist
  isn't found and shows similar-style picks instead — but it's not a
  global music search.
- **Mood is derived, not measured.** Our `derive_mood` rule is a useful
  approximation, not a ground-truth label. A jazz ballad might be
  classified as `moody` when a human would say `wistful`. The taxonomy
  has 6 moods to keep things simple.
- **Genre families are coarse.** We collapse Spotify's hundreds of
  micro-genres into 12 families. A user asking for "math rock" gets
  routed to `rock` — the LLM may correctly extract `genre=rock` and
  energy/mood that match math rock's profile, but the catalog can't
  distinguish it from, say, classic rock.
- **Popularity bias.** Sampling weights toward popular tracks, which
  means independent or non-Western artists are systematically
  underrepresented unless their popularity score happens to be high.
- **English-language slang only.** The system prompt's slang dictionary
  is English-centric; queries in other languages, or English slang from
  other regions, won't get the same feature-mapping support.
- **Single-user, no taste profile.** The agent doesn't learn from the
  user across sessions. Each conversation is fresh.

## 7. Evaluation

**Methodology.** Property-based assertions against ~30 YAML cases. We
intentionally do *not* use exact-match comparisons or LLM-as-judge:

- Exact-match would flake on LLM nondeterminism even at temperature=0
  (qwen2.5:3b's tokenizer/scheduling don't perfectly determinize on CPU).
- LLM-as-judge with a 3B model judging its own output is circular and
  noisy. A larger judge model would mean importing exactly the dependency
  this project tries to avoid.

Instead each case asserts properties of the result:

- "Top result's `energy` must be ≥ 0.7"
- "Top result's `genre` must be in `[pop, rock, hip-hop, electronic]`"
- "Average `energy` of top-5 in turn 2 must exceed turn 1 by at least
  +0.10" (refinement consistency)
- "No song ID returned may be outside the catalog" (hard fail)

**Headline metrics from the most recent run** (see
`eval/results/summary.json`):

| Metric | Value |
|---|---|
| Total cases | 30 |
| Passed | 30 (100%) |
| Hallucinated IDs | 0 |
| Behavioral pass rate | 100% (12/12) |
| Golden pass rate | 100% (12/12) |
| Refinement pass rate | 100% (6/6) |
| Runtime | ~5:30 on CPU (~46 LLM calls) |

**Determinism.** All Ollama calls use `temperature=0, seed=42, top_p=1.0`.
This brings nondeterminism close to zero in practice, though not exactly
zero on CPU.

## 8. Future work

- **Larger model option.** Add a `SOUNDFIT_MODEL` env var path that picks
  up `qwen2.5:7b` or hosted Claude/GPT when available. Currently fixed at
  `qwen2.5:3b` to keep the offline promise simple.
- **Embedding-based candidate retrieval.** Replace the brute-force
  scoring loop with a sentence-embedding ANN index for queries that
  describe vibes the rule-based scorer can't capture (e.g. lyrical
  themes). Keep the current scorer as the re-ranking pass.
- **A real evaluation set with human labels.** Right now the YAML cases
  are author-written; they encode my guesses about user intent.
  Crowd-sourced or in-the-loop labels would be more honest.
- **Multi-language slang.** The system prompt is English-only.
- **Personalization.** A persistent user profile that learns from
  thumbs-up / skip signals over time.

## 9. Reflection

Building this taught me the gap between *prompting an LLM to do the right
thing* and *guaranteeing it does*. The earliest version of the agent
hallucinated artists in explanations, leaked context across turns, and
occasionally returned IDs that didn't exist. Every one of those failure
modes is now a hard validator in the code, not a line in the system
prompt — because the prompt is hope and the validator is policy.

The other thing I didn't appreciate going in: the deterministic core
matters more, not less, when you wrap it in an LLM. The original 3-tier
weighted scorer in `recommender.py` is the part that makes the system
explainable, debuggable, and evaluable. The agent layer adds reach (a
user can type natural language) and polish (sentence-long explanations)
but the *truth* of the recommendation comes from the math, and the eval
harness measures the agent against that truth. If the LLM disagreed with
the scorer the right answer would be to fix the scorer's features, not
to let the LLM override.

## 10. Original SoundFit 1.0 model card (preserved for context)

The original card from the rule-based-only version is preserved below for
reference. It captures the starting point and the educational reflection
on weighted feature scoring before the agentic layer was added.

> The recommender is designed to give users new song recommendations
> based on what they appreciate the most in music.
>
> The program will check on what the user likes the most in terms of the
> mood and genre of his songs, with other aspects. It looks into those
> values and gives suggestions of new songs that they might like.
>
> The model uses a dataset of 10 songs, with the title, genre, and moods
> being represented as strings and other parts in the set represented as
> percentages.
>
> The system can identify a strong correlation on genre and mood. If
> they are both a match, then it is 60% more likely to be recommended.
>
> Genre is an exact binary match worth 3 points — the single largest
> factor worth 30% of the overall scoring. A song in the user's preferred
> genre starts with a 3-point head start over every song outside it.
>
> For user profiles 3 and 4, I have experimented with the different
> aspects of the preferences. I have discovered that lofi genre gets
> associated highly with a chill mood, while pop is more associated with
> a happy mood.
>
> One thing that I learned about these recommenders is how almost all
> of them use mathematical calculations on scoring and being almost
> inaccurate with what the user intends to find. Sure, keeping track of
> the user's favorite features in music is great, but what if the user
> wanted something completely different? What if he is making a playlist
> for different vibes and moods? I found it rather interesting how
> almost all systems rely on pure data consumption and can be inaccurate
> with what a person may be looking for.

That observation about *intent vs. data* is exactly the gap SoundFit 2.0
tries to close: the LLM bridges natural-language intent into the
structured features the scorer already knows how to handle, while the
scorer remains the source of truth.
