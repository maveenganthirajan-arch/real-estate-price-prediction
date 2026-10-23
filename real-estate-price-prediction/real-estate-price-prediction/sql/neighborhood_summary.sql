-- Market summary per neighbourhood.
SELECT n.name                                  AS neighborhood,
       n.zip_code,
       COUNT(*)                                AS sales,
       AVG(s.sale_price)                       AS avg_price,
       AVG(s.sale_price * 1.0 / p.living_area) AS avg_price_per_sqft,
       MIN(s.sale_price)                       AS min_price,
       MAX(s.sale_price)                       AS max_price
FROM sales s
JOIN properties p    ON p.property_id = s.property_id
JOIN neighborhoods n ON n.neighborhood_id = p.neighborhood_id
GROUP BY n.name, n.zip_code
ORDER BY avg_price_per_sqft DESC;
