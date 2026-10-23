-- Monthly price trend. (SQLite syntax; on PostgreSQL use to_char(s.sale_date, 'YYYY-MM').)
SELECT strftime('%Y-%m', s.sale_date)          AS month,
       COUNT(*)                                AS sales,
       AVG(s.sale_price)                       AS avg_price,
       AVG(s.sale_price * 1.0 / p.living_area) AS avg_price_per_sqft
FROM sales s
JOIN properties p ON p.property_id = s.property_id
GROUP BY month
ORDER BY month;
