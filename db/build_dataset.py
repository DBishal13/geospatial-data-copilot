"""Builds the demo dataset: fetches real power-infrastructure geometry for
San Francisco from OpenStreetMap (Overpass API), then synthesizes the
asset-management fields OSM doesn't track (install date, inspection date,
condition/damage score, inspection notes) on top of those real locations.

Run: python db/build_dataset.py
"""
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from agent import config  # noqa: E402

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Downtown / Mission / SoMa San Francisco — small enough for a fast Overpass
# query, dense enough (~1,500 power nodes) for a meaningful demo.
BBOX = {"south": 37.75, "west": -122.45, "north": 37.80, "east": -122.39}

ASSET_TYPE_MAP = {
    "pole": "utility_pole",
    "tower": "transmission_tower",
    "transformer": "transformer",
}

TODAY = date.today()
RANDOM_SEED = 42

# Inspection note templates, grouped by severity band, so semantic search
# over "corrosion", "vegetation", etc. has real variety to distinguish.
NOTES_LOW = [
    "Routine inspection, no issues found. Structure sound and upright.",
    "Asset in good condition. Hardware secure, no visible wear.",
    "No defects observed during scheduled inspection.",
    "Passed visual inspection. Paint/coating intact.",
]
NOTES_MED = [
    "Minor surface corrosion observed on base, not yet structural.",
    "Slight lean detected (under 5 degrees), monitor at next cycle.",
    "Vegetation encroachment near base, recommend clearing.",
    "Faded warning signage, hardware otherwise sound.",
    "Minor woodpecker damage to wood surface, no structural concern.",
    "Small crack in insulator housing, scheduled for follow-up.",
]
NOTES_HIGH = [
    "Significant corrosion on lower structure, recommend priority repair.",
    "Pronounced lean (over 15 degrees), structural risk, urgent follow-up.",
    "Heavy rust and corrosion at base, metal thinning visible.",
    "Fire scorching on lower section from nearby vegetation fire.",
    "Cracked pole base with visible splitting, high failure risk.",
    "Severe corrosion around mounting hardware, transformer housing compromised.",
    "Animal nest causing insulation damage, corrosion visible underneath.",
    "Vandalism/graffiti plus corrosion damage to access panel.",
]

# Appended to each note to reduce exact-duplicate text across ~1,500 assets
# (many assets would otherwise share byte-identical notes, which degrades
# semantic-search quality — Chroma's approximate nearest-neighbor index
# performs poorly when huge numbers of vectors are exact duplicates).
DETAIL_SUFFIXES = [
    "Photo documented for records.",
    "Access was limited due to nearby parked vehicles.",
    "Nearby vegetation recently trimmed by crew.",
    "No further action needed at this time.",
    "Recommend re-inspection at the standard interval.",
    "Cross-street landmark noted in field log.",
    "Weather conditions were clear during inspection.",
    "Inspector flagged for supervisor review.",
    "Consistent with prior inspection history for this asset.",
    "Adjacent assets in the same block were also checked.",
    "Field crew used a bucket truck for close inspection.",
    "Noted for inclusion in next maintenance cycle.",
]


def fetch_osm_assets() -> list[dict]:
    query = f"""
    [out:json][timeout:60];
    (
      node["power"~"^(pole|tower|transformer)$"]
        ({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
    );
    out body;
    """
    headers = {"User-Agent": "geospatial-data-copilot/0.1 (portfolio demo project)"}
    resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=90)
    resp.raise_for_status()
    elements = resp.json()["elements"]

    assets = []
    for el in elements:
        power_tag = el.get("tags", {}).get("power")
        asset_type = ASSET_TYPE_MAP.get(power_tag)
        if not asset_type:
            continue
        assets.append({"osm_id": el["id"], "lat": el["lat"], "lon": el["lon"], "asset_type": asset_type})
    return assets


def synthesize_fields(assets: list[dict], rng: random.Random) -> list[dict]:
    for i, a in enumerate(assets, start=1):
        a["id"] = i
        install = date(1995, 1, 1) + timedelta(days=rng.randint(0, (date(2018, 1, 1) - date(1995, 1, 1)).days))
        a["install_date"] = install

        earliest_inspection = max(install, TODAY - timedelta(days=365 * 3))
        span_days = max((TODAY - earliest_inspection).days, 1)
        last_inspection = earliest_inspection + timedelta(days=rng.randint(0, span_days))
        a["last_inspection_date"] = last_inspection

        # Older time-since-inspection skews (weakly) toward higher damage score.
        days_since = (TODAY - last_inspection).days
        base = min(60, days_since / 20)
        score = int(max(0, min(100, rng.gauss(base, 20))))
        a["condition_score"] = score

        # Most assets degrade gradually year over year; a minority improve
        # (post-maintenance). Drives the "condition score decline" query.
        delta = rng.gauss(6, 10)
        a["condition_score_last_year"] = int(max(0, min(100, score - delta)))

        if score < 30:
            base_note = rng.choice(NOTES_LOW)
        elif score < 65:
            base_note = rng.choice(NOTES_MED)
        else:
            base_note = rng.choice(NOTES_HIGH)
        a["inspection_note"] = f"{base_note} {rng.choice(DETAIL_SUFFIXES)}"
    return assets


def load_into_duckdb(assets: list[dict]) -> None:
    config.DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DUCKDB_PATH))
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute((ROOT_DIR / "db" / "schema.sql").read_text())
    con.execute("DELETE FROM assets")

    con.executemany(
        """
        INSERT INTO assets
            (id, osm_id, asset_type, install_date, last_inspection_date,
             condition_score, condition_score_last_year, inspection_note, lon, lat, geom)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ST_Point(?, ?))
        """,
        [
            (
                a["id"], a["osm_id"], a["asset_type"], a["install_date"], a["last_inspection_date"],
                a["condition_score"], a["condition_score_last_year"], a["inspection_note"],
                a["lon"], a["lat"], a["lon"], a["lat"],
            )
            for a in assets
        ],
    )
    count = con.execute("SELECT count(*) FROM assets").fetchone()[0]
    con.close()
    print(f"Loaded {count} assets into {config.DUCKDB_PATH}")


def main():
    print(f"Fetching OSM power infrastructure for bbox {BBOX} ...")
    raw_assets = fetch_osm_assets()
    if not raw_assets:
        raise SystemExit("No assets returned from Overpass — check network access or bbox.")
    print(f"Fetched {len(raw_assets)} real assets from OpenStreetMap.")

    rng = random.Random(RANDOM_SEED)
    assets = synthesize_fields(raw_assets, rng)
    load_into_duckdb(assets)


if __name__ == "__main__":
    main()
