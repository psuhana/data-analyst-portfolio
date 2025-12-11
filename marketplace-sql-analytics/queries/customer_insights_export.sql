-- ===========================================
-- CUSTOMER INSIGHTS QUERIES (WITH EXPORTS)
-- ===========================================

-- 1. Total customers → outputs/total_customers.csv
COPY (
    SELECT COUNT(*) AS total_customers
    FROM 'data/customers.csv'
)
TO 'outputs/total_customers.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 2. Monthly Active Users (MAU) → outputs/mau.csv
COPY (
    SELECT
        strftime(order_timestamp, '%Y-%m') AS month,
        COUNT(DISTINCT customer_id) AS monthly_active_users
    FROM 'data/orders.csv'
    GROUP BY month
    ORDER BY month
)
TO 'outputs/mau.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 3. Repeat customers → outputs/repeat_customers.csv
COPY (
    SELECT 
        customer_id,
        COUNT(*) AS total_orders
    FROM 'data/orders.csv'
    GROUP BY customer_id
    HAVING total_orders > 1
    ORDER BY total_orders DESC
)
TO 'outputs/repeat_customers.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 4. Top 10 highest spending customers → outputs/top_spenders.csv
COPY (
    SELECT 
        customer_id,
        SUM(total_amount) AS total_spent
    FROM 'data/orders.csv'
    GROUP BY customer_id
    ORDER BY total_spent DESC
    LIMIT 10
)
TO 'outputs/top_spenders.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

--------------------------------------------------------

-- 5. Customers with > 5 cancelled orders → outputs/churn_risk_customers.csv
COPY (
    SELECT 
        customer_id,
        COUNT(*) AS cancelled_orders
    FROM 'data/orders.csv'
    WHERE order_status = 'cancelled'
    GROUP BY customer_id
    HAVING cancelled_orders >= 5
)
TO 'outputs/churn_risk_customers.csv'
(FORMAT CSV, HEADER, DELIMITER ',');
