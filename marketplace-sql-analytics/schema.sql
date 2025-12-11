-- ============================================
-- Marketplace SQL Analytics Project - Database Schema
-- City: Chennai
-- ============================================

-- -------------------------------
-- 1) Customers Table
-- -------------------------------
CREATE TABLE customers (
    customer_id        INT PRIMARY KEY,
    name               VARCHAR(100),
    email              VARCHAR(150),
    phone              VARCHAR(50),
    signup_date        DATE,
    city               VARCHAR(50)
);

-- -------------------------------
-- 2) Restaurants Table
-- -------------------------------
CREATE TABLE restaurants (
    restaurant_id             INT PRIMARY KEY,
    name                      VARCHAR(150),
    cuisine_type              VARCHAR(100),
    city                      VARCHAR(50),
    avg_preparation_time_min  INT,
    rating                    DECIMAL(3,2)
);

-- -------------------------------
-- 3) Menu Items Table
-- -------------------------------
CREATE TABLE menu_items (
    item_id        INT PRIMARY KEY,
    restaurant_id  INT,
    item_name      VARCHAR(150),
    price          DECIMAL(10,2),
    category       VARCHAR(100),
    
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);

-- -------------------------------
-- 4) Delivery Riders Table
-- -------------------------------
CREATE TABLE delivery_riders (
    rider_id       INT PRIMARY KEY,
    name           VARCHAR(100),
    city           VARCHAR(50),
    joining_date   DATE,
    active_status  VARCHAR(20)
);

-- -------------------------------
-- 5) Orders Table
-- -------------------------------
CREATE TABLE orders (
    order_id           INT PRIMARY KEY,
    customer_id        INT,
    restaurant_id      INT,
    order_timestamp    TIMESTAMP,
    delivery_timestamp TIMESTAMP,
    total_amount       DECIMAL(10,2),
    delivery_rider_id  INT,
    payment_method     VARCHAR(20),
    order_status       VARCHAR(20),

    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id),
    FOREIGN KEY (delivery_rider_id) REFERENCES delivery_riders(rider_id)
);

-- -------------------------------
-- 6) Order Items Table
-- -------------------------------
CREATE TABLE order_items (
    order_item_id  INT PRIMARY KEY,
    order_id       INT,
    item_id        INT,
    quantity       INT,
    item_price     DECIMAL(10,2),
    
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (item_id) REFERENCES menu_items(item_id)
);

-- -------------------------------
-- 7) Ratings Table
-- -------------------------------
CREATE TABLE ratings (
    rating_id        INT PRIMARY KEY,
    order_id         INT,
    customer_id      INT,
    restaurant_id    INT,
    rating           INT,
    review_text      TEXT,
    review_timestamp TIMESTAMP,

    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);
