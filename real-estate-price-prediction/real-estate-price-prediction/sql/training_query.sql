-- Flat modelling table, ordered by time (used for the time-based split).
SELECT s.sale_id,
       p.property_id,
       n.name AS neighborhood,
       n.city,
       n.zip_code,
       p.latitude,
       p.longitude,
       p.bedrooms,
       p.bathrooms,
       p.living_area,
       p.lot_size,
       p.property_type,
       p.year_built,
       s.sale_date,
       s.sale_price
FROM sales s
JOIN properties p    ON p.property_id = s.property_id
JOIN neighborhoods n ON n.neighborhood_id = p.neighborhood_id
ORDER BY s.sale_date, s.sale_id;
