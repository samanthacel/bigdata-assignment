-- 1. KPI utama
SELECT COUNT(*) AS total_case,
       ROUND(100.0 * AVG(is_within_sla), 2) AS sla_compliance_pct,
       ROUND(AVG(resolution_hours), 2) AS avg_resolution_h,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_hours) AS median_h,
       PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY resolution_hours) AS p90_h
FROM dw.fact_case;

-- 2. Roll-up / drill-down: kategori -> subkategori
SELECT k.category, k.subcategory, COUNT(*) AS cases,
       ROUND(100.0 * AVG(f.is_within_sla), 2) AS sla_pct, ROUND(AVG(f.resolution_hours), 2) AS avg_h
FROM dw.fact_case f JOIN dw.dim_category k USING (category_key)
GROUP BY ROLLUP (k.category, k.subcategory)
ORDER BY k.category NULLS LAST, cases DESC;

-- 3. Tren bulanan (slice waktu)
SELECT d.year, d.month, COUNT(*) AS cases, ROUND(100.0 * AVG(f.is_within_sla), 2) AS sla_pct
FROM dw.fact_case f JOIN dw.dim_date d ON d.date_key = f.created_date_key
GROUP BY d.year, d.month ORDER BY d.year, d.month;

-- 4. Dice: kanal x tipe
SELECT c.channel_name, k.case_type, COUNT(*) AS cases
FROM dw.fact_case f JOIN dw.dim_channel c USING (channel_key) JOIN dw.dim_category k USING (category_key)
GROUP BY GROUPING SETS ((c.channel_name, k.case_type), (c.channel_name), ())
ORDER BY 1 NULLS LAST, 3 DESC;

-- 5. Beban kerja per hari & jam (perencanaan staf)
SELECT d.day_name, f.created_hour, COUNT(*) AS cases
FROM dw.fact_case f JOIN dw.dim_date d ON d.date_key = f.created_date_key
GROUP BY d.day_name, f.created_hour ORDER BY cases DESC LIMIT 20;

-- 6. Top-3 subkategori per kategori (window function)
SELECT * FROM (
  SELECT k.category, k.subcategory, COUNT(*) AS n,
         RANK() OVER (PARTITION BY k.category ORDER BY COUNT(*) DESC) AS rk
  FROM dw.fact_case f JOIN dw.dim_category k USING (category_key)
  GROUP BY k.category, k.subcategory) x
WHERE rk <= 3 ORDER BY category, rk;

-- 7. Tim dengan pelanggaran SLA terbanyak
SELECT t.team_name, COUNT(*) FILTER (WHERE f.is_within_sla = 0) AS breached,
       ROUND(100.0 * AVG(f.is_within_sla), 2) AS sla_pct
FROM dw.fact_case f JOIN dw.dim_team t USING (team_key)
GROUP BY t.team_name ORDER BY breached DESC;

-- 8. Backlog case terbuka
SELECT k.category, COUNT(*) AS open_cases
FROM dw.fact_case f JOIN dw.dim_category k USING (category_key)
WHERE f.is_closed = 0 GROUP BY k.category ORDER BY open_cases DESC;
