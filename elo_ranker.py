

import hashlib
import os
import time
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
FREE_VOTES_BEFORE_DAMPING = 50000  # votes in that window before influence starts shrinking
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


def execute_with_retry(builder, attempts=3, delay_seconds=1.5):
    """
    Run a Supabase query, retrying a couple of times on transient network
    errors (dropped connections, a paused-project wake-up, brief blips)
    before giving up. Used in place of a bare .execute() everywhere.
    """
    last_error = None
    for attempt in range(attempts):
        try:
            return builder.execute()
        except Exception as e:
            last_error = e
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
    raise last_error


# ---------- voter identity ----------

def get_voter_id():
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
    resp = execute_with_retry(supabase.table("ratings").select("*"))
    ratings_df = pd.DataFrame(resp.data) if resp.data else pd.DataFrame(columns=["name", "rating", "comparisons"])

    existing = set(ratings_df["name"]) if not ratings_df.empty else set()
    missing = players_df.loc[~players_df["name"].isin(existing), ["name"]].copy()
    if not missing.empty:
        missing["rating"] = float(INITIAL_RATING)
        missing["comparisons"] = 0
        execute_with_retry(supabase.table("ratings").insert(missing.to_dict("records")))
        ratings_df = pd.concat([ratings_df, missing], ignore_index=True)

    # Supabase/PostgREST returns whole-number ratings (e.g. a fresh 1500)
    # as JSON integers rather than decimals, which makes pandas infer an
    # int column. Later assigning a real (decimal) Elo score into that
    # column then fails. Force the dtypes explicitly so that never happens.
    ratings_df["rating"] = ratings_df["rating"].astype(float)
    ratings_df["comparisons"] = ratings_df["comparisons"].astype(int)

    return ratings_df


def get_rating_rows(supabase, names):
    resp = execute_with_retry(supabase.table("ratings").select("*").in_("name", names))
    rows = {row["name"]: row for row in (resp.data or [])}
    missing = [n for n in names if n not in rows]
    if missing:
        st.error(f"Couldn't find rating rows for: {', '.join(missing)} - try refreshing the page.")
        st.stop()
    return rows


# ---------- Elo math ----------

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 250))


def update_elo(rating_a, rating_b, score_a, k):
    """score_a is 1 if A won, 0 if A lost."""
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1 - exp_a
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - exp_b)
    return new_a, new_b


# ---------- anti-abuse checks ----------

def fetch_recent_votes(supabase, voter_id, hours):
    """One query covering both the damping count and the pair-cooldown
    check, since both just need this voter's recent votes."""
    cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)).isoformat()
    resp = execute_with_retry(
        supabase.table("votes")
        .select("winner,loser,created_at")
        .eq("voter_id", voter_id)
        .gte("created_at", cutoff)
    )
    return resp.data or []


def pairs_from_votes(votes):
    return {frozenset((v["winner"], v["loser"])) for v in votes}


def effective_k(votes_in_window):
    over_threshold = max(0, votes_in_window - FREE_VOTES_BEFORE_DAMPING)
    k = K_FACTOR / (1 + over_threshold * DAMPING_PER_EXTRA_VOTE)
    return max(MIN_K, k)


# ---------- pair selection ----------

def pick_pair(ratings_df, excluded_pairs=None, last_pair=None):
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
    recent_votes = fetch_recent_votes(supabase, voter_id, PAIR_COOLDOWN_HOURS)
    excluded = pairs_from_votes(recent_votes)
    st.session_state.current_pair = pick_pair(st.session_state.ratings_df, excluded_pairs=excluded)
    st.session_state.last_pair = None

ratings_df = st.session_state.ratings_df
name_a, name_b = st.session_state.current_pair

st.title("Who's the greatest Sabre?")
total_votes = int(ratings_df["comparisons"].sum() / 2)
st.caption(f"{total_votes} comparisons made so far · {len(ratings_df)} players in the pool")


def register_vote(winner, loser):
    window_hours = max(DAMPING_WINDOW_HOURS, PAIR_COOLDOWN_HOURS)
    recent_votes = fetch_recent_votes(supabase, voter_id, window_hours)

    damping_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=DAMPING_WINDOW_HOURS)
    recent_count = sum(1 for v in recent_votes if pd.Timestamp(v["created_at"]) >= damping_cutoff)

    cooldown_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=PAIR_COOLDOWN_HOURS)
    excluded_pairs = {
        frozenset((v["winner"], v["loser"]))
        for v in recent_votes
        if pd.Timestamp(v["created_at"]) >= cooldown_cutoff
    }

    rows = get_rating_rows(supabase, [winner, loser])
    k = effective_k(recent_count)
    new_winner_rating, new_loser_rating = update_elo(rows[winner]["rating"], rows[loser]["rating"], 1, k)
    new_winner_comparisons = rows[winner]["comparisons"] + 1
    new_loser_comparisons = rows[loser]["comparisons"] + 1

    execute_with_retry(supabase.table("ratings").upsert([
        {"name": winner, "rating": new_winner_rating, "comparisons": new_winner_comparisons},
        {"name": loser, "rating": new_loser_rating, "comparisons": new_loser_comparisons},
    ]))
    execute_with_retry(supabase.table("votes").insert({
        "winner": winner,
        "loser": loser,
        "voter_id": voter_id,
    }))

    # update the in-memory copy directly instead of re-fetching the whole
    # table - saves a full-table round trip on every single vote
    df = st.session_state.ratings_df
    df.loc[df["name"] == winner, "rating"] = new_winner_rating
    df.loc[df["name"] == winner, "comparisons"] = new_winner_comparisons
    df.loc[df["name"] == loser, "rating"] = new_loser_rating
    df.loc[df["name"] == loser, "comparisons"] = new_loser_comparisons

    excluded_pairs.add(frozenset((winner, loser)))
    st.session_state.last_pair = frozenset((winner, loser))
    st.session_state.current_pair = pick_pair(df, excluded_pairs=excluded_pairs, last_pair=st.session_state.last_pair)


def show_image(name):
    img = safe_player_image(name, players_df, max_width=500)
    if img is not None:
        st.image(img, use_container_width=True)



st.html("""
<style>
.st-key-vote_row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-areas:
        "img1 img2"
        "btn1 btn2";
    column-gap: 1rem;
    row-gap: 0.5rem;
}
.st-key-vote_row [data-testid="stElementContainer"]:nth-of-type(1),
.st-key-vote_row [data-testid="element-container"]:nth-of-type(1) { grid-area: img1; }
.st-key-vote_row [data-testid="stElementContainer"]:nth-of-type(2),
.st-key-vote_row [data-testid="element-container"]:nth-of-type(2) { grid-area: btn1; align-self: end; }
.st-key-vote_row [data-testid="stElementContainer"]:nth-of-type(3),
.st-key-vote_row [data-testid="element-container"]:nth-of-type(3) { grid-area: btn2; align-self: end; }
.st-key-vote_row [data-testid="stElementContainer"]:nth-of-type(4),
.st-key-vote_row [data-testid="element-container"]:nth-of-type(4) { grid-area: img2; }

@media (max-width: 640px) {
    .st-key-vote_row {
        grid-template-columns: 1fr;
        grid-template-areas:
            "img1"
            "btn1"
            "btn2"
            "img2";
    }
}
</style>
""")

with st.container(key="vote_row"):
    show_image(name_a)
    if st.button(name_a, use_container_width=True):
        with st.spinner("Saving your vote..."):
            try:
                register_vote(name_a, name_b)
            except Exception:
                st.warning("Couldn't reach the database just now - please try again in a moment.")
                st.stop()
        st.rerun()
    if st.button(name_b, use_container_width=True):
        with st.spinner("Saving your vote..."):
            try:
                register_vote(name_b, name_a)
            except Exception:
                st.warning("Couldn't reach the database just now - please try again in a moment.")
                st.stop()
        st.rerun()
    show_image(name_b)

if st.button("Skip this pair"):
    with st.spinner("Loading next matchup..."):
        recent_votes = fetch_recent_votes(supabase, voter_id, PAIR_COOLDOWN_HOURS)
        excluded = pairs_from_votes(recent_votes)
        st.session_state.last_pair = frozenset((name_a, name_b))
        st.session_state.current_pair = pick_pair(
            ratings_df, excluded_pairs=excluded, last_pair=st.session_state.last_pair
        )
    st.rerun()

st.divider()


TOP_N_WITH_IMAGES = 10
GRID_COLUMNS = 5


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
