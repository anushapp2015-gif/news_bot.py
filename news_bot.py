import os, time, uuid, tempfile, requests
from datetime import datetime, timezone
from supabase import create_client
import google.generativeai as genai
import feedparser
from urllib.parse import quote

# --- Secrets ---
SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_KEY"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "news-images")
GEMINI_KEY      = os.environ["GEMINI_KEY"]
TABLE_NAME      = "news"

# --- Init ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Google News RSS per category ---
CATEGORY_FEEDS = {
    "TS/AP": [
        "https://news.google.com/rss/search?q=telangana&hl=en-IN&gl=IN&ceid=IN:en",
        "https://news.google.com/rss/search?q=andhra+pradesh&hl=en-IN&gl=IN&ceid=IN:en",
    ],
    "India":         ["https://news.google.com/rss/search?q=india+news&hl=en-IN&gl=IN&ceid=IN:en"],
    "Sports":        ["https://news.google.com/rss/search?q=india+sports&hl=en-IN&gl=IN&ceid=IN:en"],
    "Entertainment": ["https://news.google.com/rss/search?q=bollywood+tollywood+cinema&hl=en-IN&gl=IN&ceid=IN:en"],
    "Business":      ["https://news.google.com/rss/search?q=india+business+economy&hl=en-IN&gl=IN&ceid=IN:en"],
}

# --- Violence filter ---
FORBIDDEN_WORDS = [
    "rape", "murder", "kill", "suicide", "violence", "dead", "dies",
    "shot", "stab", "assault", "blast", "bomb", "terror", "kidnap",
    "robbery", "massacre", "riot", "arson", "hostage", "execution"
]

def is_violent(text):
    if not text:
        return False
    lower = text.lower()
    return any(word in lower for word in FORBIDDEN_WORDS)

def clean_title(raw_title: str) -> str:
    # Google News appends " - Source Name" — remove it
    if " - " in raw_title:
        return raw_title.rsplit(" - ", 1)[0].strip()
    return raw_title.strip()

def get_source_name(entry) -> str:
    if hasattr(entry, "source") and hasattr(entry.source, "title"):
        return entry.source.title[:80]
    title = entry.get("title", "")
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return "Google News"

def fetch_articles(feed_url: str, limit: int = 8) -> list:
    try:
        feed = feedparser.parse(feed_url)
        safe = [e for e in feed.entries if not is_violent(e.get("title", ""))]
        return safe[:limit]
    except Exception as e:
        print(f"  ⚠️ RSS fetch failed: {e}")
        return []

# --- Gemini: summary + vocab + image prompt ---
def ai_process(title: str, raw_summary: str) -> dict:
    prompt = f"""You are writing news for Indian school kids aged 10-14.

Original headline: {title}
Original text: {raw_summary}

Return ONLY valid JSON (no markdown, no backticks):
{{
  "headline": "Simple rewritten headline in 10-12 words, present tense",
  "content": "3 simple sentences explaining this news for kids. Factual and educational.",
  "vocab": "word1 – meaning1,,word2 – meaning2",
  "image_prompt": "A colorful, friendly, cartoon-style illustration for this news. Safe for kids. No text in image. Max 15 words."
}}"""

    for attempt in range(3):
        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip().replace("```json","").replace("```","").strip()
            import json
            return json.loads(text)
        except Exception as e:
            print(f"  ⚠️ Gemini error (attempt {attempt+1}/3): {e}")
            time.sleep(3)

    # Fallback
    return {
        "headline": title,
        "content": raw_summary[:200],
        "vocab": "",
        "image_prompt": f"colorful cartoon illustration about {title[:40]}"
    }

# --- Generate AI image via Pollinations.ai (FREE, no key needed) ---
def generate_ai_image(prompt: str, category: str) -> str:
    try:
        # Make prompt safe and kid-friendly
        safe_prompt = f"colorful cartoon illustration, kid friendly, {prompt}, bright colors, no text, flat design"
        encoded     = quote(safe_prompt)
        # Pollinations free AI image API
        image_url   = f"https://image.pollinations.ai/prompt/{encoded}?width=800&height=450&nologo=true"

        # Download the generated image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        # Upload to YOUR Supabase storage bucket
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        file_name = f"news_images/{uuid.uuid4()}.jpg"
        supabase.storage.from_(SUPABASE_BUCKET).upload(file_name, tmp_path)

        public_url = (
            f"{SUPABASE_URL}/storage/v1/object/public/"
            f"{SUPABASE_BUCKET}/{file_name}"
        )
        print(f"  🎨 AI image generated & uploaded")
        return public_url

    except Exception as e:
        print(f"  ⚠️ AI image failed: {e}")
        # Category fallback if generation fails
        FALLBACKS = {
            "India":         "https://image.pollinations.ai/prompt/colorful+map+of+india+cartoon+style+kids+friendly?width=800&height=450&nologo=true",
            "TS/AP":         "https://image.pollinations.ai/prompt/telangana+andhra+pradesh+colorful+cartoon+landscape?width=800&height=450&nologo=true",
            "Sports":        "https://image.pollinations.ai/prompt/kids+playing+cricket+football+colorful+cartoon?width=800&height=450&nologo=true",
            "Entertainment": "https://image.pollinations.ai/prompt/colorful+movie+stage+lights+cartoon+style+kids?width=800&height=450&nologo=true",
            "Business":      "https://image.pollinations.ai/prompt/colorful+cartoon+coins+charts+piggy+bank+kids?width=800&height=450&nologo=true",
        }
        return FALLBACKS.get(category, "https://placehold.co/800x450?text=News")

# --- Main ---
def run():
    print("=== News Bot Starting ===\n")
    all_articles = []

    for category, feed_urls in CATEGORY_FEEDS.items():
        print(f"── [{category}]")
        collected = 0

        for feed_url in feed_urls:
            if collected >= 2:
                break

            articles = fetch_articles(feed_url, limit=8)

            for entry in articles:
                if collected >= 2:
                    break

                raw_title   = clean_title(entry.get("title", ""))
                raw_summary = entry.get("summary", raw_title)[:800]
                share_url   = entry.get("link", "")
                source_name = get_source_name(entry)

                if is_violent(raw_summary):
                    print(f"  🚫 Filtered: {raw_title[:60]}")
                    continue

                print(f"  → {raw_title[:70]}")

                # Gemini: rewrite + vocab + image prompt
                data = ai_process(raw_title, raw_summary)

                # Pollinations: generate AI image
                image_url = generate_ai_image(data["image_prompt"], category)

                all_articles.append({
                    "category":   category,
                    "headline":   data["headline"],
                    "content":    data["content"],
                    "vocabulary": data["vocab"],
                    "image_url":  image_url,
                    "share_url":  share_url,
                    "source":     source_name,
                    "edited_by":  "news_bot",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

                collected += 1
                print(f"  ✓ [{collected}/2] {data['headline'][:65]}")
                time.sleep(3)  # avoid rate limits

        print(f"  Done — {collected} articles\n")

    if not all_articles:
        print("❌ No articles found.")
        return

    # Delete old → insert fresh
    print("🗑️  Clearing old news...")
    try:
        supabase.table(TABLE_NAME).delete().neq("id", 0).execute()
        print("✓ Cleared\n")
    except Exception as e:
        print(f"  ⚠️ Clear failed: {e}")

    print("📥 Inserting new articles...")
    for article in all_articles:
        try:
            supabase.table(TABLE_NAME).insert(article).execute()
            print(f"  ✅ [{article['category']}] {article['headline'][:65]}")
        except Exception as e:
            print(f"  ❌ Insert failed: {e}")

    print(f"\n=== Done — {len(all_articles)} articles saved ===")

if __name__ == "__main__":
    run()
