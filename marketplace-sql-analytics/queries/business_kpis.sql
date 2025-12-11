-- ===========================================
-- BUSINESS KPI QUERIES
-- ===========================================

-- 1. GMV (Gross Merchandise Value)
SELECT SUM(total_amount) AS GMV
FROM 'data/orders.csv'
WHERE order_status = 'delivered';

-- 2. Average Order Value (AOV)
SELECT AVG(total_amount) AS AOV
FROM 'data/orders.csv'
WHERE order_status = 'delivered';

-- 3. Order success rate
SELECT
    COUNT(*) FILTER (WHERE order_status = 'delivered') * 100.0 / COUNT(*) AS success_rate
FROM 'data/orders.csv';

-- 4. Top items by quantity sold
SELECT 
    oi.item_id,
    SUM(quantity) AS total_quantity
FROM 'data/order_items.csv' oi
GROUP BY oi.item_id
ORDER BY total_quantity DESC
LIMIT 10;

-- 5. Rating → delivery time correlation
SELECT 
    rating,
    AVG(o.delivery_timestamp - o.order_timestamp) AS avg_delivery_time
FROM 'data/ratings.csv' r
JOIN 'data/orders.csv' o ON o.order_id = r.order_id
GROUP BY rating
ORDER BY rating DESC;
