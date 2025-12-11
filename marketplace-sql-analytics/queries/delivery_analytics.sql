-- 1. Average delivery time per order
SELECT 
    order_id,
    (delivery_timestamp - order_timestamp) AS delivery_time
FROM 'data/orders.csv'
LIMIT 20;

-- 2. Rider efficiency score
SELECT 
    delivery_rider_id AS rider_id,
    AVG(delivery_timestamp - order_timestamp) AS avg_delivery_time,
    COUNT(order_id) AS total_deliveries
FROM 'data/orders.csv'
WHERE order_status = 'delivered'
GROUP BY rider_id
ORDER BY avg_delivery_time ASC;

-- 3. SLA breach % (45 min threshold)
SELECT
    COUNT(*) FILTER (WHERE delivery_timestamp > order_timestamp + INTERVAL '45 minutes') * 100.0 
        / COUNT(*) AS sla_breach_pct
FROM 'data/orders.csv'
WHERE order_status = 'delivered';

-- 4. Hourly delivery load patterns
SELECT
    strftime(order_timestamp, '%H') AS hour,
    COUNT(*) AS total_orders
FROM 'data/orders.csv'
GROUP BY hour
ORDER BY hour;

-- 5. Rider cancellation attribution
SELECT
    delivery_rider_id AS rider_id,
    COUNT(*) AS cancellations
FROM 'data/orders.csv'
WHERE order_status = 'cancelled'
GROUP BY rider_id
ORDER BY cancellations DESC
LIMIT 10;
