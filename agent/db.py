"""Safe, read-only SQL execution against the DuckDB asset database.

The database is a single local file; run db/build_dataset.py (a writer)
only while the API/CLI (readers) are stopped.
"""
import re

import duckdb

from agent import config

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|copy|pragma|create|call|export|import)\b",
    re.IGNORECASE,
)

SCHEMA_DESCRIPTION = """\
Table: assets
  id                    BIGINT      -- primary key
  osm_id                BIGINT      -- source OpenStreetMap node id
  asset_type            VARCHAR     -- one of: utility_pole, transmission_tower, transformer
  install_date          DATE
  last_inspection_date  DATE
  condition_score       INTEGER     -- 0 = excellent condition, 100 = critical/severe damage (current)
  condition_score_last_year INTEGER -- same scale, as of ~1 year ago (use for decline/trend questions)
  inspection_note       VARCHAR     -- free-text inspection note
  lon                   DOUBLE      -- longitude (WGS84)
  lat                   DOUBLE      -- latitude (WGS84)
  geom                  GEOMETRY    -- ST_Point(lon, lat); use only inside ST_* functions, never SELECT it directly

Spatial helper functions available (DuckDB `spatial` extension):
  ST_Point(lon, lat)                     -- build a point
  ST_Distance_Sphere(geom_a, geom_b)      -- great-circle distance in METERS between two points
  ST_X(geom) / ST_Y(geom)                 -- extract lon/lat back out of a geometry

Known landmark coordinates (lat, lon) you may use for "near <place>" questions:
  San Francisco City Hall: 37.7793, -122.4193
  Dolores Park:            37.7596, -122.4269
  Salesforce Tower:        37.7897, -122.3972
"""


def is_safe_select(sql: str) -> bool:
    stripped = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(with\b.*?)?select\b", stripped):
        return False
    return not _FORBIDDEN.search(stripped)


def run_select(sql: str, row_limit: int = 200) -> dict:
    """Executes a SELECT and returns {"rows": [...], "total_count": int,
    "truncated": bool}. total_count is the TRUE number of matching rows even
    when `rows` is capped at row_limit — callers must use total_count (not
    len(rows)) when reporting counts, since a capped preview silently
    undercounts otherwise."""
    if not is_safe_select(sql):
        raise ValueError(f"Refusing to execute non-SELECT or unsafe SQL: {sql!r}")

    stripped = sql.strip().rstrip(";").strip()
    has_limit = "limit" in stripped.lower()
    capped_sql = stripped if has_limit else f"{stripped} LIMIT {row_limit}"

    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        con.execute("LOAD spatial;")
        cursor = con.execute(capped_sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()

        if has_limit:
            total_count = len(rows)
        else:
            total_count = con.execute(f"SELECT count(*) FROM ({stripped}) AS _sub").fetchone()[0]
    finally:
        con.close()

    dict_rows = [dict(zip(columns, row)) for row in rows]
    return {"rows": dict_rows, "total_count": total_count, "truncated": total_count > len(dict_rows)}
