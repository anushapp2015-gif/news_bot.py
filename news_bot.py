import feedparser, requests, os, json
from supabase import create_client
import google.generativeai as genai

# --- Secrets from GitHub Actions ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
GEMINI_KEY   = os.environ["GEMINI_KEY"]
PEXELS_KEY   = os.environ["PEXELS_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Your exact category names as seen in table ---
TOPICS = {
    "India":    "https://news.google.com/rss/search?q=india+news&hl=en-IN&gl=IN&ceid=IN:en",
    "TS/AP":    "https://news.google.com/rss/search?q=telangana+OR+andhra+pradesh&hl=en-IN&gl=IN&ceid=IN:en",
    "Sports":   "https://news.google.com/rss/search?q=india+sports&hl=en-IN&gl=IN&ceid=IN:en",
    "Cinema":   "https://news.google.com/rss/search?q=bollywood+OR+tollywood&hl=en-IN&gl=IN&ceid=IN:en",
    "Business": "https://news.google.com/rss/search?q=india+business+economy&hl=en-IN&gl=IN&ceid=IN:en",
}

# --- Safe keywords filter for kids ---
BLOCKED = [
    "murder","killed","rape","assault","stab","dead body","suicide",
    "attack","bomb","terrorist","molest","abduct","kidnap","robbery",
    "shoot","gun","massacre","riot","blast","explosion","arson",
    "death toll","fatality","accident","crash","fire"
]

def is_safe(title: str) -> bool:
    t = title.lower()
    return not any(word in t for word in BLOCKED)

def get_source_name(entry) -> str:
    # Google News RSS puts source in 'source' tag or title after " - "
    if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
        return entry.source.title[:50]  # truncate to fit column
    title = entry.get("title", "")
    if " - " in title:
        return title.split(" - ")[-1].strip()[:50]
    return "News Bot"

def clean_title(raw_title: str) -> str:
    # Remove "Source Name - " suffix that Google News adds
    if " - " in raw_title:
        return raw_title.rsplit(" - ", 1)[0].strip()
    return raw_title.strip()

def fetch_top_articles(feed_url: str, limit: int = 8) -> list:
    feed = feedparser.parse(feed_url)
    safe = [e for e in feed.entries if is_safe(e.get("title", ""))]
    return safe[:limit]

def ai_process(title: str, raw_summary: str) -> dict:
    prompt = f"""
You are writing news for Indian school kids aged 10-14.

Original headline: {title}
Original text: {raw_summary}

Return ONLY valid JSON (no markdown backticks, no explanation):
{{
  "headline": "Simple rewritten headline in 10-12 words, present tense, easy English",
  "content": "Write exactly 3 sentences about this news in very simple English for kids. Be factual and educational.",
  "vocab1_word": "One interesting word from this topic (not too basic)",
  "vocab1_meaning": "One simple sentence explaining the word for a 12-year-old",
  "vocab2_word": "A second useful word from this topic",
  "vocab2_meaning": "One simple sentence explaining the word for a 12-year-old"
}}
"""
    resp = model.generate_content(prompt)
    text = resp.text.strip()
    # Clean any markdown fences if Gemini adds them
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def format_vocabulary(data: dict) -> str:
    # Stored in your single 'vocabulary' text column
    return (
        f"📘 {data['vocab1_word']}: {data['vocab1_meaning']}  |  "
        f"📗 {data['vocab2_word']}: {data['vocab2_meaning']}"
    )

def fetch_image(keyword: str) -> str:
    try:
        url = f"https://api.pexels.com/v1/search?query={keyword}&per_page=5&orientation=landscape"
        headers = {"Authorization": PEXELS_KEY}
        r = requests.get(url, headers=headers, timeout=10).json()
        photos = r.get("photos", [])
        if photos:
            # Pick medium size — good balance for mobile apps
            return photos[0]["src"]["large"]
    except Exception as e:
        print(f"  Image fetch error: {e}")
    return ""

def run():
    print("=== News Bot Starting ===")

    for category, feed_url in TOPICS.items():
        print(f"\n── [{category}]")
        articles = fetch_top_articles(feed_url, limit=8)

        if not articles:
            print("  No safe articles found, skipping.")
            continue

        saved = 0
        for article in articles:
            if saved >= 2:
                break
            try:
                raw_title   = clean_title(article.get("title", ""))
                raw_summary = article.get("summary", raw_title)[:800]
                share_url   = article.get("link", "")
                source_name = get_source_name(article)

                print(f"  → {raw_title[:65]}...")

                # AI: rewrite + vocabulary
                data = ai_process(raw_title, raw_summary)

                # Vocabulary formatted for your column
                vocabulary = format_vocabulary(data)

                # Image from Pexels
                image_url = fetch_image(category.lower().replace("/", " "))

                # Insert — matching YOUR exact column names
                supabase.table("news").insert({
                    "category":   category,
                    "headline":   data["headline"],
                    "content":    data["content"],
                    "vocabulary": vocabulary,
                    "image_url":  image_url,
                    "source":     source_name,
                    "share_url":  share_url,
                    "edited_by":  "news_bot",
                }).execute()

                saved += 1
                print(f"  ✓ [{saved}/2] {data['headline']}")

            except json.JSONDecodeError as e:
                print(f"  ✗ Gemini returned invalid JSON: {e}")
                continue
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        print(f"  Done — {saved} articles saved")

    # Clean up articles older than 7 days
    try:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        supabase.table("news").delete().lt("created_at", cutoff).execute()
        print("\n✓ Old articles cleaned up (>7 days)")
    except Exception as e:
        print(f"\n  Cleanup skipped: {e}")

    print("\n=== Done ===")

if __name__ == "__main__":
    run()
