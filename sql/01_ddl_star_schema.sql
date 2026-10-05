CREATE SCHEMA IF NOT EXISTS dw;

CREATE TABLE IF NOT EXISTS dw.dim_date (
  date_key INTEGER PRIMARY KEY, full_date DATE NOT NULL, day INTEGER, month INTEGER,
  month_name VARCHAR(20), quarter INTEGER, year INTEGER, day_of_week INTEGER,
  day_name VARCHAR(20), is_weekend BOOLEAN
);
CREATE TABLE IF NOT EXISTS dw.dim_channel (
  channel_key INTEGER PRIMARY KEY, channel_name VARCHAR(100) NOT NULL
);
CREATE TABLE IF NOT EXISTS dw.dim_category (
  category_key INTEGER PRIMARY KEY, category VARCHAR(150), subcategory VARCHAR(150), case_type VARCHAR(100)
);
CREATE TABLE IF NOT EXISTS dw.dim_team (
  team_key INTEGER PRIMARY KEY, team_name VARCHAR(150)
);
CREATE TABLE IF NOT EXISTS dw.dim_case_status (
  status_key INTEGER PRIMARY KEY, status VARCHAR(50), sla_status VARCHAR(50)
);
CREATE TABLE IF NOT EXISTS dw.fact_case (
  case_number      BIGINT PRIMARY KEY,
  created_date_key INTEGER NOT NULL REFERENCES dw.dim_date(date_key),
  closed_date_key  INTEGER REFERENCES dw.dim_date(date_key),
  channel_key      INTEGER NOT NULL REFERENCES dw.dim_channel(channel_key),
  category_key     INTEGER NOT NULL REFERENCES dw.dim_category(category_key),
  team_key         INTEGER NOT NULL REFERENCES dw.dim_team(team_key),
  status_key       INTEGER NOT NULL REFERENCES dw.dim_case_status(status_key),
  created_ts       TIMESTAMP NOT NULL,
  closed_ts        TIMESTAMP,
  created_hour     INTEGER,
  resolution_hours NUMERIC(12,2),
  is_within_sla    INTEGER,
  is_closed        INTEGER,
  case_count       INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_fact_created  ON dw.fact_case(created_date_key);
CREATE INDEX IF NOT EXISTS ix_fact_category ON dw.fact_case(category_key);
CREATE INDEX IF NOT EXISTS ix_fact_channel  ON dw.fact_case(channel_key);
CREATE INDEX IF NOT EXISTS ix_fact_team     ON dw.fact_case(team_key);

CREATE OR REPLACE VIEW dw.v_case AS
SELECT f.*, d.full_date AS created_date, d.year, d.month, d.day_name,
       c.channel_name, k.category, k.subcategory, k.case_type, t.team_name, s.status, s.sla_status
FROM dw.fact_case f
JOIN dw.dim_date d        ON d.date_key = f.created_date_key
JOIN dw.dim_channel c     USING (channel_key)
JOIN dw.dim_category k    USING (category_key)
JOIN dw.dim_team t        USING (team_key)
JOIN dw.dim_case_status s USING (status_key);
