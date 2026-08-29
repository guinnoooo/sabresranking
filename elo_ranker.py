"""
Pairwise Player Ranker (Supabase-backed)
-----------------------------------------
Public pairwise "who's better" ranking site. Votes are stored permanently
in Supabase (Postgres) instead of a local CSV, plus some basic anti-abuse
protections:

  - voter identity: hashed IP address, falling back to a per-session id
    when no IP is available (e.g. local development)
  - matchup cooldown: the same voter won't be shown the same pair again
    for a while, so they can't just repeatedly click one matchup
  - diminishing influence: nobody is ever blocked from voting, but past
    a threshold of votes in a day, each further vote from that identity
    counts for progressively less - a burst from one source moves
    ratings much less than the same number of votes spread across many
    different voters

Images: give players.csv an "image" column with a file path or URL, or
drop a file named after the player (e.g. "Tom Brady.jpg") into an
"images" folder next to this script - either is picked up automatically.
Every photo is center-cropped to the same fixed portrait ratio, and
shown both in the matchup view and in the rankings list below. Anyone
with no photo found falls back to images/placeholder.jpg - add a file
there to control what that looks like.

One-time setup:
  1. Create a free Supabase project, then run schema.sql in its SQL
     editor (Database > SQL Editor) to create the tables.
  2. Copy secrets.toml.example to .streamlit/secrets.toml and fill in
     your Supabase URL/key. Never commit the real secrets.toml.
  3. pip install -r requirements.txt
  4. streamlit run elo_ranker.py
"""

import hashlib
import os
import uuid
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from supabase import create_client

PLAYERS_FILE = "players.csv"
IMAGES_DIR = "images"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
IMAGE_ASPECT_RATIO = 7 / 10  # width / height - every photo gets cropped to this shape
PLACEHOLDER_IMAGE = os.path.join(IMAGES_DIR, "placeholder.jpg")  # used when a player has no photo

K_FACTOR = 32
INITIAL_RATING = 1500

# --- anti-abuse tuning (no hard caps - voting is never blocked) ---
DAMPING_WINDOW_HOURS = 1         # window used to count an identity's "recent" votes
FREE_VOTES_BEFORE_DAMPING = 100  # votes in that window before influence starts shrinking
DAMPING_PER_EXTRA_VOTE = 0.15    # how fast K shrinks per vote past the free threshold
MIN_K = 4                        # K never drops below this floor - votes always count for something
PAIR_COOLDOWN_HOURS = 1          # don't re-show the same matchup to the same voter within this window

st.set_page_config(page_title="Sabre Ranker", layout="centered")


# ---------- Supabase ----------

@st.cache_resource
def get_client():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


# ---------- voter identity ----------

def get_voter_id():
    """
    Hashed IP when Streamlit can see one (true on real deployments),
    otherwise a per-session random id as a weaker fallback (e.g. local
    dev, where st.context.ip_address is always None).
    Note: IP alone is spoofable (VPNs, shared networks) - this is a
    deterrent layer, not a security guarantee.
    """
    salt = st.secrets.get("supabase", {}).get("voter_salt", "local-dev-salt")
    ip = st.context.ip_address
    if ip:
        return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()
    if "session_voter_id" not in st.session_state:
        st.session_state.session_voter_id = f"session:{uuid.uuid4().hex}"
    return st.session_state.session_voter_id


# ---------- players / images (still local - these rarely change) ----------

def load_players():
    path = PLAYERS_FILE if os.path.exists(PLAYERS_FILE) else "players_sample.csv"
    if not os.path.exists(path):
        st.error("Couldn't find players.csv (or players_sample.csv). Add one with a 'name' column.")
        st.stop()
    df = pd.read_csv(path)
    if "name" not in df.columns:
        st.error("Your CSV needs a column called 'name'.")
        st.stop()
    return df


def find_local_image(name):
    if not os.path.isdir(IMAGES_DIR):
        return None
    candidates = [name, name.replace(" ", "_"), name.replace(" ", "-")]
    for candidate in candidates:
        for ext in IMAGE_EXTENSIONS:
            path = os.path.join(IMAGES_DIR, candidate + ext)
            if os.path.exists(path):
                return path
    return None


def get_player_image(name, players_df):
    if "image" in players_df.columns:
        row = players_df.loc[players_df["name"] == name]
        if not row.empty:
            val = row.iloc[0]["image"]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    local = find_local_image(name)
    if local:
        return local
    return PLACEHOLDER_IMAGE


@st.cache_data(show_spinner=False)
def load_cropped_image(path_or_url, max_width):
    """
    Open a local file or URL and center-crop it to IMAGE_ASPECT_RATIO,
    so every player's photo displays at the same shape regardless of
    what size/orientation the original was. Cached so the same image
    isn't re-downloaded and re-cropped on every rerun.
    """
    if path_or_url.startswith(("http://", "https://")):
        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
    else:
        img = Image.open(path_or_url)
    img = img.convert("RGB")

    w, h = img.size
    if w / h > IMAGE_ASPECT_RATIO:
        # wider than target - crop the sides
        new_w = round(h * IMAGE_ASPECT_RATIO)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # taller than target - crop top/bottom
        new_h = round(w / IMAGE_ASPECT_RATIO)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    if img.width > max_width:
        new_height = round(max_width / IMAGE_ASPECT_RATIO)
        img = img.resize((max_width, new_height))

    return img


def safe_player_image(name, players_df, max_width):
    path = get_player_image(name, players_df)
    if not path:
        return None
    try:
        return load_cropped_image(path, max_width)
    except Exception:
        return None


# ---------- ratings (Supabase) ----------

def load_ratings(supabase, players_df):
    resp = supabase.table("ratings").select("*").execute()
    ratings_df = pd.DataFrame(resp.data) if resp.data else pd.DataFrame(columns=["name", "rating", "comparisons"])

    existing = set(ratings_df["name"]) if not ratings_df.empty else set()
    missing = players_df.loc[~players_df["name"].isin(existing), ["name"]].copy()
    if not missing.empty:
        missing["rating"] = float(INITIAL_RATING)
        missing["comparisons"] = 0
        supabase.table("ratings").insert(missing.to_dict("records")).execute()
        ratings_df = pd.concat([ratings_df, missing], ignore_index=True)

    return ratings_df


def get_rating_row(supabase, name):
    resp = supabase.table("ratings").select("*").eq("name", name).execute()
    if not resp.data:
        st.error(f"Couldn't find a rating row for {name} - try refreshing the page.")
        st.stop()
    return resp.data[0]


# ---------- Elo math ----------

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, score_a, k):
    """score_a is 1 if A won, 0 if A lost."""
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1 - exp_a
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - exp_b)
    return new_a, new_b


# ---------- anti-abuse checks ----------

def recent_vote_count(supabase, voter_id, minutes):
    cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(minutes=minutes)).isoformat()
    resp = (
        supabase.table("votes")
        .select("id", count="exact")
        .eq("voter_id", voter_id)
        .gte("created_at", cutoff)
        .execute()
    )
    return resp.count or 0


def effective_k(votes_in_window):
    """
    Nobody gets blocked - but the first FREE_VOTES_BEFORE_DAMPING votes
    from an identity in the window count fully, and each vote past that
    counts for progressively less. A burst from one source ends up
    moving ratings far less than the same number of votes spread across
    many different people.
    """
    over_threshold = max(0, votes_in_window - FREE_VOTES_BEFORE_DAMPING)
    k = K_FACTOR / (1 + over_threshold * DAMPING_PER_EXTRA_VOTE)
    return max(MIN_K, k)


def recent_pairs_for_voter(supabase, voter_id, hours):
    cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)).isoformat()
    resp = (
        supabase.table("votes")
        .select("winner,loser")
        .eq("voter_id", voter_id)
        .gte("created_at", cutoff)
        .execute()
    )
    return {frozenset((row["winner"], row["loser"])) for row in (resp.data or [])}


# ---------- pair selection ----------

def pick_pair(ratings_df, excluded_pairs=None, last_pair=None):
    """
    Favor players with fewer comparisons so far, so early rounds spread
    votes across everyone. Also skips any pair this voter has already
    been shown recently (excluded_pairs), and never immediately repeats
    the pair they just saw (last_pair).
    """
    excluded_pairs = excluded_pairs or set()
    if len(ratings_df) < 2:
        st.error("Need at least 2 players in players.csv.")
        st.stop()

    sorted_df = ratings_df.sort_values("comparisons")
    pool_size = max(6, len(sorted_df) // 3)
    pool = sorted_df.head(pool_size)

    for _ in range(30):
        sample_from = pool if len(pool) >= 2 else ratings_df
        pair = sample_from.sample(2)
        a_name, b_name = pair.iloc[0]["name"], pair.iloc[1]["name"]
        key = frozenset((a_name, b_name))
        if a_name != b_name and key != last_pair and key not in excluded_pairs:
            return a_name, b_name

    # everything's on cooldown for this voter - fall back to any valid pair
    pair = ratings_df.sample(2)
    return pair.iloc[0]["name"], pair.iloc[1]["name"]


# ---------- app ----------

supabase = get_client()
players_df = load_players()
voter_id = get_voter_id()

if "ratings_df" not in st.session_state:
    st.session_state.ratings_df = load_ratings(supabase, players_df)

if "current_pair" not in st.session_state:
    excluded = recent_pairs_for_voter(supabase, voter_id, PAIR_COOLDOWN_HOURS)
    st.session_state.current_pair = pick_pair(st.session_state.ratings_df, excluded_pairs=excluded)
    st.session_state.last_pair = None

ratings_df = st.session_state.ratings_df
name_a, name_b = st.session_state.current_pair

st.title("Who's the greatest Sabre?")
total_votes = int(ratings_df["comparisons"].sum() / 2)
st.caption(f"{total_votes} comparisons made so far · {len(ratings_df)} players in the pool")


def register_vote(winner, loser):
    recent = recent_vote_count(supabase, voter_id, DAMPING_WINDOW_HOURS * 60)

    winner_row = get_rating_row(supabase, winner)
    loser_row = get_rating_row(supabase, loser)
    k = effective_k(recent)
    new_winner_rating, new_loser_rating = update_elo(winner_row["rating"], loser_row["rating"], 1, k)

    supabase.table("ratings").update({
        "rating": new_winner_rating,
        "comparisons": winner_row["comparisons"] + 1,
    }).eq("name", winner).execute()
    supabase.table("ratings").update({
        "rating": new_loser_rating,
        "comparisons": loser_row["comparisons"] + 1,
    }).eq("name", loser).execute()
    supabase.table("votes").insert({
        "winner": winner,
        "loser": loser,
        "voter_id": voter_id,
    }).execute()

    st.session_state.ratings_df = load_ratings(supabase, players_df)
    st.session_state.last_pair = frozenset((name_a, name_b))
    excluded = recent_pairs_for_voter(supabase, voter_id, PAIR_COOLDOWN_HOURS)
    st.session_state.current_pair = pick_pair(
        st.session_state.ratings_df, excluded_pairs=excluded, last_pair=st.session_state.last_pair
    )


def show_image(name):
    img = safe_player_image(name, players_df, max_width=500)
    if img is not None:
        st.image(img, use_container_width=True)


col1, col2 = st.columns(2)
with col1:
    show_image(name_a)
    if st.button(name_a, use_container_width=True):
        register_vote(name_a, name_b)
        st.rerun()
with col2:
    show_image(name_b)
    if st.button(name_b, use_container_width=True):
        register_vote(name_b, name_a)
        st.rerun()

if st.button("Skip this pair"):
    excluded = recent_pairs_for_voter(supabase, voter_id, PAIR_COOLDOWN_HOURS)
    st.session_state.last_pair = frozenset((name_a, name_b))
    st.session_state.current_pair = pick_pair(
        ratings_df, excluded_pairs=excluded, last_pair=st.session_state.last_pair
    )
    st.rerun()

st.divider()


TOP_N_WITH_IMAGES = 10
GRID_COLUMNS = 2


def render_top_grid(top_df, players_df):
    """Top players shown as big photo cards, name underneath - sexymp.uk style."""
    rows = list(enumerate(top_df.itertuples(), start=1))
    for start in range(0, len(rows), GRID_COLUMNS):
        chunk = rows[start:start + GRID_COLUMNS]
        cols = st.columns(len(chunk))
        for col, (rank, row) in zip(cols, chunk):
            with col:
                img = safe_player_image(row.name, players_df, max_width=400)
                if img is not None:
                    st.image(img, use_container_width=True)
                st.markdown(f"**{rank}. {row.name}**")
                st.caption(f"{int(round(row.rating))} · {int(row.comparisons)} comparisons")


def render_rest_list(rest_df):
    """Everyone past the top N - plain text rows, no images."""
    header = st.columns([1, 5, 2, 2])
    header[1].markdown("**Player**")
    header[2].markdown("**Rating**")
    header[3].markdown("**Comparisons**")

    for i, row in enumerate(rest_df.itertuples(), start=TOP_N_WITH_IMAGES + 1):
        cols = st.columns([1, 5, 2, 2])
        cols[0].write(f"{i}.")
        cols[1].write(row.name)
        cols[2].write(int(round(row.rating)))
        cols[3].write(int(row.comparisons))


def render_rankings(df, players_df):
    sorted_df = df.sort_values("rating", ascending=False).reset_index(drop=True)
    top_df = sorted_df.head(TOP_N_WITH_IMAGES)
    rest_df = sorted_df.iloc[TOP_N_WITH_IMAGES:]

    render_top_grid(top_df, players_df)
    if not rest_df.empty:
        st.divider()
        render_rest_list(rest_df)


with st.expander("Current rankings", expanded=False):
    render_rankings(ratings_df, players_df)
