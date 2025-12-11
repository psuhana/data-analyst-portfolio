-- ===========================================
-- RESTAURANT PERFORMANCE QUERIES
-- ===========================================

-- 1. Top restaurants by order volume
SELECT 
    restaurant_id,
    COUNT(*) AS total_orders
FROM 'data/orders.csv'
GROUP BY restaurant_id
ORDER BY total_orders DESC
LIMIT 10;

-- 2. Average order value (AOV) by restaurant
SELECT 
    restaurant_id,
    AVG(total_amount) AS avg_order_value
FROM 'data/orders.csv'
GROUP BY restaurant_id
ORDER BY avg_order_value DESC;

-- 3. Cuisine performance ranking
SELECT 
    r.cuisine_type,
    COUNT(*) AS total_orders
FROM 'data/orders.csv' o
JOIN 'data/restaurants.csv' r ON r.restaurant_id = o.restaurant_id
GROUP BY r.cuisine_type
ORDER BY total_orders DESC;

-- 4. Most expensive menu items (top 15)
SELECT *
FROM 'data/menu_items.csv'
ORDER BY price DESC
LIMIT 15;

-- 5. Restaurant rating distribution
SELECT 
    restaurant_id,
    AVG(rating) AS avg_rating,
    COUNT(*) AS rating_count
FROM 'data/ratings.csv'
GROUP BY restaurant_id
ORDER BY avg_rating DESC
LIMIT 10;
