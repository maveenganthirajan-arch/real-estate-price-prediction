-- Portable schema (runs on SQLite as used by the pipeline, and on PostgreSQL).
-- PostgreSQL:  psql -d realestate -f sql/schema.sql
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS properties;
DROP TABLE IF EXISTS neighborhoods;

CREATE TABLE neighborhoods (
    neighborhood_id INTEGER PRIMARY KEY,
    name            VARCHAR(100) NOT NULL UNIQUE,
    city            VARCHAR(100),
    zip_code        VARCHAR(10),
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION
);

CREATE TABLE properties (
    property_id     VARCHAR(20) PRIMARY KEY,
    neighborhood_id INTEGER NOT NULL REFERENCES neighborhoods (neighborhood_id),
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    bedrooms        INTEGER,
    bathrooms       NUMERIC(3, 1),
    living_area     INTEGER NOT NULL,
    lot_size        INTEGER,
    property_type   VARCHAR(30) NOT NULL,
    year_built      INTEGER
);

CREATE TABLE sales (
    sale_id     INTEGER PRIMARY KEY,
    property_id VARCHAR(20) NOT NULL REFERENCES properties (property_id),
    sale_date   DATE NOT NULL,
    sale_price  NUMERIC(12, 0) NOT NULL CHECK (sale_price > 0)
);

CREATE INDEX idx_sales_date         ON sales (sale_date);
CREATE INDEX idx_properties_hood    ON properties (neighborhood_id);
CREATE INDEX idx_properties_type    ON properties (property_type);
