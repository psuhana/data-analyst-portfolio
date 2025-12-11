-- ===========================================
-- RESTAURANT PERFORMANCE QUERIES (WITH EXPORTS)
-- ===========================================

-- 1. Top restaurants by order volume → outputs/top_restaurants.csv
COPY (
    SELECT 
        restaurant_id,
        COUNT(*) AS total_orders
    FROM 'data/orders.csv'
    GROUP BY restaurant_id
    ORDER BY total_orders DESC
) 
TO 'outputs/top_restaurants.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 2. Average order value (AOV) by restaurant → outputs/restaurant_aov.csv
COPY (
    SELECT 
        restaurant_id,
        AVG(total_amount) AS avg_order_value
    FROM 'data/orders.csv'
    GROUP BY restaurant_id
    ORDER BY avg_order_value DESC
)
TO 'outputs/restaurant_aov.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 3. Cuisine ranking → outputs/cuisine_performance.csv
COPY (
    SELECT 
        r.cuisine_type,
        COUNT(*) AS total_orders
    FROM 'data/orders.csv' o
    JOIN 'data/restaurants.csv' r 
        ON r.restaurant_id = o.restaurant_id
    GROUP BY r.cuisine_type
    ORDER BY total_orders DESC
)
TO 'outputs/cuisine_performance.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 4. Most expensive menu items → outputs/expensive_items.csv
COPY (
    SELECT *
    FROM 'data/menu_items.csv'
    ORDER BY price DESC
    LIMIT 15
)
TO 'outputs/expensive_items.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 5. Restaurant rating summary → outputs/restaurant_ratings.csv
COPY (
    SELECT 
        restaurant_id,
        AVG(rating) AS avg_rating,
        COUNT(*) AS rating_count
    FROM 'data/ratings.csv'
    GROUP BY restaurant_id
    ORDER BY avg_rating DESC
)
TO 'outputs/restaurant_ratings.csv'
(FORMAT CSV, HEADER, DELIMITER ',');
