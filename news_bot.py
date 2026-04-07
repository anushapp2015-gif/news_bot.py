import os, time, uuid, tempfile, requests, json, re
from datetime import datetime, timezone
from urllib.parse import quote, urljoin
from supabase import create_client
import google.generativeai as genai
from bs4 import BeautifulSoup

# --- Secrets ---
SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_KEY"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "NewsImages")
GEMINI_KEY      = os.environ["GEMINI_KEY"]
TABLE_NAME      = "news"

# --- Init ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

BASE_URL = "https://telanganatoday.com"

# --- Telangana Today category pages ---
CATEGORY_URLS = {
    "TS/AP": [
        "https://telanganatoday.com/telangana",
        "https://telanganatoday.com/andhra-pradesh",
    ],
    "India":         ["https://telanganatoday.com/india"],
    "Sports":        ["https://telanganatoday.com/sport"],
    "Entertainment": ["https://telanganatoday.com/entertainment"],
    "Business":      ["https://telanganatoday.com/business"],
}

# --- Violence/negative filter ---
FORBIDDEN_WORDS = [
    "rape", "murder", "kill", "killed", "suicide", "violence",
    "dead", "dies", "died", "shot", "stab", "assault", "blast",
    "bomb", "terror", "kidnap", "robbery", "massacre", "riot",
    "arson", "hostage", "execution", "maoist", "naxal", "militant",
    "drug", "arrest", "scam", "fraud", "trafficking", "porn",
    "sex offense", "nude", "war", "attack", "clash", "protest",
    "strike", "accident", "crash"
]

def is_violent(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(word in lower for word in FORBIDDEN_WORDS)

# --- Normalize title for duplicate detection ---
def normalize(title: str) -> str:
    # Lowercase, remove punctuation, extra spaces
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def titles_are_similar(t1: str, t2: str, threshold: float = 0.6) -> bool:
    """Check if two titles share enough words to be considered duplicates."""
    words1 = set(normalize(t1).split())
    words2 = set(normalize(t2).split())
    # Remove common stop words
    stop = {"the","a","an","is","in","of","to","and","for","on","at",
            "by","with","from","as","it","its","that","this","was",
            "are","be","has","had","have","will","been","not","but"}
    words1 -= stop
    words2 -= stop
    if not words1 or not words2:
        return False
    overlap = words1 & words2
    similarity = len(overlap) / max(len(words1), len(words2))
    return similarity >= threshold

# --- Scrape article links from category page ---
def get_article_links(url: str, limit: int = 10) -> list:
    try:
        res = requests.get(url,
                          headers={"User-Agent": "Mozilla/5.0"},
                          timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        links = []
        for a in soup.select("h3 a, h2 a")[:limit]:
            href = a.get("href", "")
            if not href:
                continue
            if not href.startswith("http"):
                href = urljoin(BASE_URL, href)
            if href not in links:
                links.append(href)
        return links
    except Exception as e:
        print(f"  ⚠️ Failed to get links from {url}: {e}")
        return []

# --- Scrape full article content ---
def scrape_article(url: str):
    try:
        res = requests.get(url,
                          headers={"User-Agent": "Mozilla/5.0"},
                          timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # Headline
        h1 = soup.find("h1")
        headline = h1.get_text(strip=True) if h1 else "Untitled"

        # Full article paragraphs
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40  # skip tiny/nav paragraphs
        ]
        content = " ".join(paragraphs)
        content = " ".join(content.split()[:300])  # first 300 words

        # Image
        image_url = extract_image(soup, url)

        return headline, content, image_url
    except Exception as e:
        print(f"  ❌ Scrape failed {url}: {e}")
        return None, None, None

def extract_image(soup, base_url: str) -> str:
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return urljoin(base_url, og["content"])
    for selector in ["article img", ".entry-content img",
                     ".post-content img", ".single-post img"]:
        img = soup.select_one(selector)
        if img and img.get("src"):
            return urljoin(base_url, img["src"])
    return ""

# --- Gemini: rewrite + vocabulary + image prompt ---
def ai_process(headline: str, content: str) -> dict:
    prompt = f"""You are an educational news writer for Indian school kids aged 10-14.

Original headline: {headline}
Original article: {content}

Return ONLY a valid JSON object. No markdown, no backticks, no explanation.

{{
  "headline": "Rewritten simple headline, 8-12 words, present tense, easy English",
  "content": "Write a proper 3-4 sentence news paragraph in simple English. Explain what happened, who is involved, where it happened, and why it matters for India. Use words a 12-year-old understands easily.",
  "vocab_word_1": "One interesting or new word from this news topic",
  "vocab_meaning_1": "One simple sentence explanation suitable for a 12-year-old",
  "vocab_word_2": "Another useful word from this news topic",
  "vocab_meaning_2": "One simple sentence explanation suitable for a 12-year-old",
  "image_prompt": "Simple colorful cartoon scene for this news, no people, no text, kid-friendly, bright, 12 words max"
}}"""

    for attempt in range(3):
        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            # Clean markdown if Gemini adds it
            text = re.sub(r'```json', '', text)
            text = re.sub(r'```', '', text)
            text = text.strip()
            # Try direct parse
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Extract JSON block if wrapped in extra text
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    return json.loads(match.group())
                raise
        except Exception as e:
            print(f"  ⚠️ Gemini attempt {attempt+1}/3 failed: {e}")
            time.sleep(3)

    # Fallback values
    return {
        "headline": headline[:80],
        "content": content[:300],
        "vocab_word_1": "",
        "vocab_meaning_1": "",
        "vocab_word_2": "",
        "vocab_meaning_2": "",
        "image_prompt": f"colorful cartoon about {headline[:30]}"
    }

def format_vocabulary(data: dict) -> str:
    parts = []
    w1 = data.get("vocab_word_1", "").strip()
    m1 = data.get("vocab_meaning_1", "").strip()
    w2 = data.get("vocab_word_2", "").strip()
    m2 = data.get("vocab_meaning_2", "").strip()
    if w1 and m1:
        parts.append(f"📘 {w1}: {m1}")
    if w2 and m2:
        parts.append(f"📗 {w2}: {m2}")
    return "  |  ".join(parts)

# --- Generate AI image + upload to Supabase ---
def generate_and_upload_image(image_prompt: str, category: str) -> str:
    try:
        safe_prompt = (
            f"colorful cartoon, kid friendly, Indian context, "
            f"{image_prompt}, bright colors, no text, flat design"
        )
        encoded   = quote(safe_prompt)
        image_url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width=800&height=450&nologo=true&seed={int(time.time())}"
        )
        print(f"  🎨 Generating AI image...")
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()

        if len(response.content) < 1000:
            raise ValueError("Image response too small")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        file_name = f"news_images/{uuid.uuid4()}.jpg"
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            file_name, tmp_path,
            {"content-type": "image/jpeg"}
        )

        public_url = (
            f"{SUPABASE_URL}/storage/v1/object/public/"
            f"{SUPABASE_BUCKET}/{file_name}"
        )
        print(f"  ✅ Image uploaded")
        return public_url

    except Exception as e:
        print(f"  ⚠️ Image failed: {e}")
        FALLBACKS = {
            "India":         "https://image.pollinations.ai/prompt/colorful+india+map+cartoon+kids+educational?width=800&height=450&nologo=true",
            "TS/AP":         "https://image.pollinations.ai/prompt/telangana+hyderabad+charminar+colorful+cartoon?width=800&height=450&nologo=true",
            "Sports":        "https://image.pollinations.ai/prompt/kids+cricket+bat+ball+stadium+colorful+cartoon?width=800&height=450&nologo=true",
            "Entertainment": "https://image.pollinations.ai/prompt/movie+clapboard+popcorn+colorful+cartoon+kids?width=800&height=450&nologo=true",
            "Business":      "https://image.pollinations.ai/prompt/coins+piggy+bank+chart+colorful+cartoon+kids?width=800&height=450&nologo=true",
        }
        return FALLBACKS.get(category, "https://placehold.co/800x450?text=News")

# --- Main ---
def run():
    print("=== News Bot Starting ===\n")
    all_articles  = []
    seen_urls     = set()   # track article URLs already used
    seen_titles   = []      # track headlines for similarity check

    for category, cat_urls in CATEGORY_URLS.items():
        print(f"── [{category}]")
        collected = 0

        for cat_url in cat_urls:
            if collected >= 2:
                break

            links = get_article_links(cat_url, limit=12)
            print(f"  Found {len(links)} links from {cat_url}")

            for link in links:
                if collected >= 2:
                    break

                # --- Duplicate URL check ---
                if link in seen_urls:
                    print(f"  ⏭️  Skipped (duplicate URL): {link[:60]}")
                    continue

                print(f"  → Scraping: {link}")
                headline, content, _ = scrape_article(link)

                if not headline or not content:
                    print("  ⚠️  Empty article, skipping")
                    continue

                # --- Violence filter ---
                if is_violent(headline) or is_violent(content):
                    print(f"  🚫 Filtered (violent): {headline[:60]}")
                    continue

                # --- Duplicate TITLE check across all categories ---
                is_dup = any(
                    titles_are_similar(headline, seen)
                    for seen in seen_titles
                )
                if is_dup:
                    print(f"  ⏭️  Skipped (similar title exists): {headline[:60]}")
                    continue

                # Mark as seen
                seen_urls.add(link)
                seen_titles.append(headline)

                # AI rewrite
                data       = ai_process(headline, content)
                vocabulary = format_vocabulary(data)
                image_url  = generate_and_upload_image(
                    data.get("image_prompt", f"news {category}"),
                    category
                )

                all_articles.append({
                    "category":   category,
                    "headline":   data["headline"],
                    "content":    data["content"],
                    "vocabulary": vocabulary,
                    "image_url":  image_url,
                    "share_url":  link,
                    "source":     "Telangana Today",
                    "edited_by":  "news_bot",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

                collected += 1
                print(f"  ✓ [{collected}/2] {data['headline'][:65]}")
                print(f"       📚 {vocabulary[:80]}")
                time.sleep(3)

        print(f"  Done — {collected} articles saved\n")

    if not all_articles:
        print("❌ No articles collected.")
        return

    # Clear old rows
    print("🗑️  Clearing old news...")
    try:
        supabase.table(TABLE_NAME).delete().neq("id", 0).execute()
        print("✓ Cleared\n")
    except Exception as e:
        print(f"  ⚠️ Clear failed: {e}")

    # Insert fresh
    print("📥 Inserting articles...")
    success = 0
    for article in all_articles:
        try:
            supabase.table(TABLE_NAME).insert(article).execute()
            print(f"  ✅ [{article['category']}] {article['headline'][:60]}")
            success += 1
        except Exception as e:
            print(f"  ❌ Insert failed: {e}")

    print(f"\n=== Done — {success}/{len(all_articles)} articles saved ===")

if __name__ == "__main__":
    run()
