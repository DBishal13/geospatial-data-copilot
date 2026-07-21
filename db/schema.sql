-- DuckDB schema for the Geospatial Data Copilot.
-- Note: this project uses DuckDB + its `spatial` extension rather than
-- PostgreSQL + PostGIS (see README "Tech stack note" for why) — the SQL
-- surface (ST_Point, ST_Distance_Sphere, ST_X/ST_Y, GEOMETRY columns) is
-- deliberately chosen to mirror PostGIS closely.

INSTALL spatial;
LOAD spatial;

CREATE TABLE IF NOT EXISTS assets (
    id BIGINT PRIMARY KEY,
    osm_id BIGINT,
    asset_type VARCHAR NOT NULL,       -- utility_pole | transmission_tower | transformer
    install_date DATE,
    last_inspection_date DATE,
    condition_score INTEGER,           -- 0 = excellent condition, 100 = critical/severe damage (current)
    condition_score_last_year INTEGER, -- same scale, as of ~1 year ago (for trend/decline queries)
    inspection_note VARCHAR,
    lon DOUBLE NOT NULL,
    lat DOUBLE NOT NULL,
    geom GEOMETRY NOT NULL             -- ST_Point(lon, lat), SRID 4326 (WGS84)
);
