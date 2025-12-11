-- ===========================================
-- BUSINESS KPIS (WITH EXPORTS)
-- ===========================================

-- 1. GMV → outputs/gmv.csv
COPY (
    SELECT SUM(total_amount) AS GMV
    FROM 'data/orders.csv'
    WHERE order_status = 'delivered'
)
TO 'outputs/gmv.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 2. Average Order Value → outputs/aov.csv
COPY (
    SELECT AVG(total_amount) AS AOV
    FROM 'data/orders.csv'
    WHERE order_status = 'delivered'
)
TO 'outputs/aov.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 3. Success rate → outputs/success_rate.csv
COPY (
    SELECT
        COUNT(*) FILTER (WHERE order_status = 'delivered') * 100.0 / COUNT(*) AS success_rate
    FROM 'data/orders.csv'
)
TO 'outputs/success_rate.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 4. Best-selling items → outputs/top_items.csv
COPY (
    SELECT 
        oi.item_id,
        SUM(quantity) AS total_quantity
    FROM 'data/order_items.csv' oi
    GROUP BY oi.item_id
    ORDER BY total_quantity DESC
    LIMIT 10
)
TO 'outputs/top_items.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 5. Rating vs Delivery Time → outputs/rating_vs_delivery.csv
COPY (
    SELECT 
        rating,
        AVG(o.delivery_timestamp - o.order_timestamp) AS avg_delivery_time
    FROM 'data/ratings.csv' r
    JOIN 'data/orders.csv' o 
        ON o.order_id = r.order_id
    GROUP BY rating
    ORDER BY rating DESC
)
TO 'outputs/rating_vs_delivery.csv'
(FORMAT CSV, HEADER, DELIMITER ',');
