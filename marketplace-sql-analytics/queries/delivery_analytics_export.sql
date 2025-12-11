COPY (
    SELECT 
        order_id,
        (delivery_timestamp - order_timestamp) AS delivery_time
    FROM 'data/orders.csv'
)
TO 'outputs/delivery_times.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

COPY (
    SELECT 
        delivery_rider_id AS rider_id,
        AVG(delivery_timestamp - order_timestamp) AS avg_delivery_time,
        COUNT(order_id) AS total_deliveries
    FROM 'data/orders.csv'
    WHERE order_status = 'delivered'
    GROUP BY rider_id
    ORDER BY avg_delivery_time ASC
)
TO 'outputs/rider_efficiency.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

COPY (
    SELECT
        COUNT(*) FILTER (
            WHERE delivery_timestamp > order_timestamp + INTERVAL '45 minutes'
        ) * 100.0 / COUNT(*) AS sla_breach_percentage
    FROM 'data/orders.csv'
    WHERE order_status = 'delivered'
)
TO 'outputs/sla_breach_rate.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

COPY (
    SELECT
        strftime(order_timestamp, '%H') AS hour,
        COUNT(*) AS total_orders
    FROM 'data/orders.csv'
    GROUP BY hour
    ORDER BY hour
)
TO 'outputs/hourly_orders.csv'
(FORMAT CSV, HEADER, DELIMITER ',');

COPY (
    SELECT
        delivery_rider_id AS rider_id,
        COUNT(*) AS cancellations
    FROM 'data/orders.csv'
    WHERE order_status = 'cancelled'
    GROUP BY rider_id
    ORDER BY cancellations DESC
)
TO 'outputs/rider_cancellations.csv'
(FORMAT CSV, HEADER, DELIMITER ',');
