import json
import re
import subprocess
from typing import List, Dict, Any

import mysql.connector
import requests
import spacy
import wikipediaapi
from mysql.connector import errorcode

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt')
nltk.download('punkt_tab')

nlp = spacy.load("en_core_web_sm")

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="MovieDescBot/1.0 (tanishk.singh@example.com)",
)

OMDB_API_KEY = "369e1ac0"
import spacy

# Load SpaCy's English NER model
nlp = spacy.load("en_core_web_sm")

def entity_level_verification(description: str, sources: list[str]) -> list[str]:
    """
    Verify if entities in `description` are present in any of the `sources`.
    Returns a list of ungrounded entities.
    """
    source_text = " ".join(sources).lower()
    doc = nlp(description)

    ungrounded_entities = []

    for ent in doc.ents:
        if ent.label_ in {"PERSON", "GPE", "LOC", "ORG", "DATE", "EVENT"}:
            if ent.text.lower() not in source_text:
                ungrounded_entities.append(f"{ent.text} ({ent.label_})")

    return ungrounded_entities


def preprocess_text(text: str) -> str:
    """Preprocess text by normalizing whitespace, lemmatizing, removing
    stopwords and punctuation, and filtering POS to nouns, adjectives,
    verbs, and proper nouns.

    Args:
        text: The input text string.

    Returns:
        The preprocessed text string.
    """
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    filtered_tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.pos_ in {"NOUN", "ADJ", "VERB", "PROPN"}
    ]
    return " ".join(filtered_tokens)


def clean_and_mask_text(text: str, starcast: List[str]) -> str:
    """Mask PERSON, GPE, ORG named entities, proper nouns missed by NER,
    and starcast names in the text.

    Args:
        text: The input text string.
        starcast: List of starcast names to mask.

    Returns:
        The masked text string.
    """
    text = re.sub(r"\s+", " ", text).strip()

    # Preprocessing is done but not used here except for potential extensions
    _ = preprocess_text(text)

    doc = nlp(text)
    masked_text = text

    ents = sorted(doc.ents, key=lambda e: len(e.text), reverse=True)

    for ent in ents:
        if ent.label_ in {"PERSON", "GPE", "ORG"}:
            pattern = re.compile(rf"\b{re.escape(ent.text)}\b", flags=re.IGNORECASE)
            masked_text = pattern.sub("[REDACTED]", masked_text)

    entity_texts = {ent.text.lower() for ent in ents}

    for token in doc:
        if token.pos_ == "PROPN" and token.text.lower() not in entity_texts:
            pattern = re.compile(rf"\b{re.escape(token.text)}\b", flags=re.IGNORECASE)
            masked_text = pattern.sub("[REDACTED]", masked_text)

    for actor in sorted(starcast, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(actor)}\b", flags=re.IGNORECASE)
        masked_text = pattern.sub("[REDACTED]", masked_text)

    return masked_text


def get_wikipedia_context(movie: Dict[str, Any], sentences: int = 2) -> str:
    """Retrieve Wikipedia summary for movie context using prioritized search
    terms.

    Args:
        movie: Dictionary containing movie data.
        sentences: Number of sentences to return from summary.

    Returns:
        Wikipedia summary text or fallback string if no page found.
    """
    title = movie.get("title", "")
    search_terms = [title]
    search_terms.extend([f"{genre} film" for genre in movie.get("genres", ["Drama"])])
    search_terms.extend(
        [
            "Indian cinema",
            "Bollywood",
            "family drama in India",
            "Indian military",
            "British Raj",
            "Indian rural life",
            "crime in India",
            "surrogacy in India",
            "Hindi film industry",
            "Tollywood",
            "Kollywood",
            "Indian film directors",
            "Indian film producers",
            "Indian movie plots",
            "Indian screenplay structure",
        ]
    )

    print(f"\nSearching Wikipedia for terms (priority order): {search_terms}")
    for term in search_terms:
        page = wiki.page(term)
        if page.exists():
            summary = ". ".join(page.summary.split(". ")[:sentences]) + "."
            print(f"Wikipedia context used from page: {term}")
            print(f"Wikipedia Summary:\n{summary}\n")
            return summary
    print("No Wikipedia context found.\n")
    return "No context available."


def get_omdb_data(title: str) -> Dict[str, Any]:
    """Fetch movie data from OMDb API.

    Args:
        title: Movie title.

    Returns:
        Dictionary of OMDb response data or empty dict on failure.
    """
    try:
        response = requests.get(
            f"http://www.omdbapi.com/?t={title}&plot=full&apikey={OMDB_API_KEY}"
        )
        data = response.json()
        if data.get("Response") == "True":
            print(f"OMDb Data for '{title}':")
            print(json.dumps(data, indent=2))
            return data
    except requests.RequestException as e:
        print(f"OMDb API error for {title}: {e}")
    return {}


def construct_prompt(movie: Dict[str, Any]) -> str:
    """Construct the prompt text for the LLaMA model.

    Args:
        movie: Dictionary containing movie data.

    Returns:
        Formatted prompt string.
    """
    print(f"\nConstructing prompt for: {movie['title']}")
    omdb_data = get_omdb_data(movie["title"])

    merged_description = " ".join(
        movie["description"].get("cinema", []) + movie["description"].get("tvplus", [])
    )
    plot_outline = movie.get("plot outline", "")
    omdb_plot = omdb_data.get("Plot", "")

    full_text = f"{merged_description} {plot_outline} {omdb_plot}".strip()

    print(f"\nMerged Plot Description (Before Masking):\n{full_text}\n")

    masked_text = clean_and_mask_text(full_text, movie.get("starcast", []))
    print(f"Masked Plot Description:\n{masked_text}\n")

    genre_context = get_wikipedia_context(movie)

    prompt = (
        f"Write a short, spoiler-free and vivid description of a {movie['languages'][0]} "
        f"{movie['genres'][0]} film.\nAvoid mentioning the title or any actor names.\n\n"
        f"Plot (anonymized):\n{masked_text}\n\n"
        f"Background Context (from Wikipedia):\n{genre_context}\n\n"
        f"Other Info (from OMDb):\n"
        f"- Year: {omdb_data.get('Year', 'N/A')}\n"
        f"- IMDb Rating: {omdb_data.get('imdbRating', 'N/A')}\n"
        f"- Director: [REDACTED]\n"
        f"- Actors: [REDACTED]\n\n"
        "Create a description that sparks curiosity and reflects the tone of the movie."
    )

    print(f"Final Prompt Sent to LLaMA:\n{prompt}\n")
    return prompt


def run_llama(prompt: str) -> str:
    """Run the LLaMA model with given prompt via subprocess.

    Args:
        prompt: Prompt string to send.

    Returns:
        Output string from LLaMA model.
    """
    print("⚙️ Running LLaMA...")
    process = subprocess.Popen(
        ["ollama", "run", "llama3.2"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, error = process.communicate(input=prompt.encode())
    result = output.decode().strip()
    print(f"LLaMA Output:\n{result}\n")
    return result


def ensure_table(cursor: mysql.connector.cursor_cext.CMySQLCursor) -> None:
    """Create the movie_descriptions table if it does not exist.

    Args:
        cursor: MySQL cursor object.
    """
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS movie_descriptions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """
    )
from bert_score import score as bert_score_fn  # at the top

def evaluate_with_metrics(generated: str, reference: str, title: str) -> Dict[str, Any]:
    """Evaluate generated text using BLEU, ROUGE, and BERTScore."""
    smoothing_fn = SmoothingFunction().method1
    from nltk.tokenize import word_tokenize

    reference_tokens = [word_tokenize(reference.lower())]
    generated_tokens = word_tokenize(generated.lower())

    bleu = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing_fn)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated)

    # BERTScore
    P, R, F1 = bert_score_fn([generated], [reference], lang='en', rescale_with_baseline=True)

    return {
        "title": title,
        "BLEU": round(bleu, 4),
        "ROUGE-1": round(rouge_scores["rouge1"].fmeasure, 4),
        "ROUGE-L": round(rouge_scores["rougeL"].fmeasure, 4),
        "BERTScore-F1": round(F1[0].item(), 4),
        "BERTScore-Precision": round(P[0].item(), 4),
        "BERTScore-Recall": round(R[0].item(), 4),
    }

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def detect_hallucinations_keyword(description: str, sources: List[str]) -> List[str]:
    hallucinated = []
    for sentence in description.split('.'):
        sentence = sentence.strip()
        if not sentence:
            continue
        if not any(sentence.lower() in src.lower() or sentence[:15].lower() in src.lower() for src in sources):
            hallucinated.append(sentence)
    return hallucinated

def detect_hallucinations_semantic(description: str, sources: List[str], threshold: float = 0.7) -> List[str]:
    hallucinated = []
    desc_sentences = [s.strip() for s in description.split('.') if s.strip()]
    for sent in desc_sentences:
        similarities = [util.cos_sim(model.encode(sent), model.encode(src)).item() for src in sources]
        if max(similarities) < threshold:
            hallucinated.append(sent)
    return hallucinated
def main() -> None:
    """Main function to process movies and store descriptions in DB."""
    with open("sample_movies_copy.json", "r") as f:
        movies = json.load(f)

    try:
        cnx = mysql.connector.connect(
            host="localhost", user="root", password="tanishk07", database="movie"
        )
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            cnx = mysql.connector.connect(
                host="localhost", user="root", password="tanishk07"
            )
            cursor = cnx.cursor()
            cursor.execute(
                "CREATE DATABASE movie CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
            cnx.database = "movie"
        else:
            print(f"MySQL connection error: {err}")
            return

    cursor = cnx.cursor()
    ensure_table(cursor)

    for movie in movies:
        title = movie["title"]
        print(f"\n🟩 Processing movie: {title}")

        # Construct prompt and generate description
        prompt = construct_prompt(movie)
        description = run_llama(prompt)

        # Evaluation metrics
        reference = " ".join(movie["description"].get("cinema", []))
        metrics = evaluate_with_metrics(description, reference, title)
        print(f"📊 Evaluation Metrics for '{title}':\n{metrics}\n")

        # Get input sources for hallucination detection
        omdb_data = get_omdb_data(title)
        merged_description = " ".join(
            movie["description"].get("cinema", []) + movie["description"].get("tvplus", [])
        )
        plot_outline = movie.get("plot outline", "")
        omdb_plot = omdb_data.get("Plot", "")
        genre_context = get_wikipedia_context(movie)
        masked_text = clean_and_mask_text(f"{merged_description} {plot_outline} {omdb_plot}", movie.get("starcast", []))

        sources = [masked_text, genre_context, omdb_plot]

        # Hallucination detection: Method 1 - Keyword Overlap
        hallucinated_keywords = detect_hallucinations_keyword(description, sources)
        if hallucinated_keywords:
            print("⚠️ Keyword Overlap Hallucinations:")
            for s in hallucinated_keywords:
                print(f"- {s}")
        else:
            print("✅ No hallucinations (Keyword method).")

        # Hallucination detection: Method 2 - Semantic Similarity
        hallucinated_semantics = detect_hallucinations_semantic(description, sources)
        if hallucinated_semantics:
            print("⚠️ Semantic Hallucinations:")
            for s in hallucinated_semantics:
                print(f"- {s}")
        else:
            print("✅ No hallucinations (Semantic method).")

        # ✅ Hallucination detection: Method 3 - Entity-level verification
        ungrounded_entities = entity_level_verification(description, sources)
        if ungrounded_entities:
            print("⚠️ Entity-level Hallucinations:")
            for e in ungrounded_entities:
                print(f"- {e}")
        else:
            print("✅ No hallucinations (Entity method).")

        # Store final description in database
        print(f"\n💾 Storing description for '{title}'…")
        insert_sql = """
            INSERT INTO movie_descriptions (title, description)
            VALUES (%s, %s)
        """
        cursor.execute(insert_sql, (title, description))
        cnx.commit()
        print("✅ Stored in DB.\n")

    cursor.close()
    cnx.close()

if __name__ == "__main__":
    main()
