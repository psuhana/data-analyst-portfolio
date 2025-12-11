-- 1. Total customers
SELECT COUNT(*) AS total_customers
FROM 'data/customers.csv';

-- 2. Monthly Active Users (MAU)
SELECT
    strftime(order_timestamp, '%Y-%m') AS month,
    COUNT(DISTINCT customer_id) AS monthly_active_users
FROM 'data/orders.csv'
GROUP BY month
ORDER BY month;

-- 3. Repeat customers
SELECT 
    customer_id,
    COUNT(*) AS total_orders
FROM 'data/orders.csv'
GROUP BY customer_id
HAVING total_orders > 1
ORDER BY total_orders DESC;

-- 4. Top 10 highest spending customers
SELECT 
    customer_id,
    SUM(total_amount) AS total_spent
FROM 'data/orders.csv'
GROUP BY customer_id
ORDER BY total_spent DESC
LIMIT 10;

-- 5. Customers with > 5 cancelled orders (churn risk signal)
SELECT 
    customer_id,
    COUNT(*) AS cancelled_orders
FROM 'data/orders.csv'
WHERE order_status = 'cancelled'
GROUP BY customer_id
HAVING cancelled_orders >= 5;
