# Marketplace SQL Analytics — Chennai Food Delivery Case Study

This project is an end-to-end SQL analytics case study built using a synthetic but realistic food delivery marketplace dataset inspired by platforms like Zomato and Swiggy.

The goal of the project is to demonstrate strong SQL querying, business KPI design, data modeling, and analytics workflows using a lightweight analytical database engine (DuckDB).

The project is fully reproducible, requires zero database setup, and produces BI-ready outputs for dashboarding.

---

## Project Overview

The analysis covers a multi-sided food delivery marketplace involving:
- Customers
- Restaurants
- Menu items
- Orders
- Delivery riders
- Ratings and reviews

Key business questions addressed include customer behavior, restaurant performance, delivery efficiency, and high-level marketplace KPIs.

---

## Project Structure

    marketplace-sql-analytics/
    │
    ├── data/
    │   ├── customers.csv
    │   ├── restaurants.csv
    │   ├── menu_items.csv
    │   ├── orders.csv
    │   ├── order_items.csv
    │   ├── delivery_riders.csv
    │   └── ratings.csv
    │
    ├── queries/
    │   ├── customer_insights.sql
    │   ├── restaurant_performance.sql
    │   ├── delivery_analytics.sql
    │   ├── business_kpis.sql
    │   ├── customer_insights_export.sql
    │   ├── restaurant_performance_export.sql
    │   ├── delivery_analytics_export.sql
    │   └── business_kpis_export.sql
    │
    ├── outputs/
    │   └── (auto-generated CSVs for BI dashboards)
    │
    ├── schema.sql
    └── README.md

---

## Tools and Technologies

- SQL (standard analytical SQL)
- DuckDB (CLI-based analytical database)
- CSV-based data modeling
- Power BI / Excel-ready outputs

DuckDB is used to run SQL queries directly on CSV files without requiring a database server or data ingestion.

---

## Dataset Overview

The dataset represents a single-city marketplace (Chennai) and contains approximately:
- 2,000 customers
- 50 restaurants
- 200 delivery riders
- ~12,000 orders

### Tables Included
- customers — user demographic and signup data
- restaurants — cuisine type, ratings, preparation time
- menu_items — item-level pricing and categories
- orders — timestamps, payment method, and order status
- order_items — item-level order breakdown
- delivery_riders — rider information and activity
- ratings — customer reviews and ratings

All relationships are documented in `schema.sql`.

---

## Analytics Covered

### Customer Insights
- Monthly Active Users (MAU)
- Repeat customer behavior
- High-value customers
- Churn risk indicators

### Restaurant Performance
- Order volume by restaurant
- Average Order Value (AOV)
- Cuisine-level demand trends
- Menu price analysis
- Rating distributions

### Delivery Analytics
- Rider efficiency
- Delivery SLA breaches
- Hourly order patterns
- Cancellation attribution

### Business KPIs
- Gross Merchandise Value (GMV)
- Average Order Value (AOV)
- Order success rate
- Top-selling items
- Rating vs delivery time relationship

---

## How to Run the Queries

Start DuckDB from the project directory:

    duckdb

Run any analysis file:

    .read queries/customer_insights.sql

---

## Exporting Results for BI

Export scripts generate CSV outputs into the `outputs/` folder.

Example:

    .read queries/business_kpis_export.sql

These outputs can be directly loaded into Power BI, Tableau, or Excel for visualization.

---

## Data Model

The project follows a normalized relational schema:

    customers (1) ──── (∞) orders
    restaurants (1) ──── (∞) orders
    orders (1) ──── (∞) order_items
    restaurants (1) ──── (∞) menu_items
    orders (1) ──── (1) ratings
    delivery_riders (1) ──── (∞) orders

---

## Author

This project is part of a data analytics portfolio demonstrating SQL analytics, business KPI development, and marketplace data modeling using real-world inspired datasets.
