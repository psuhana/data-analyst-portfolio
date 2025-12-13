# Marketplace SQL Analytics: Chennai Food Delivery Case Study

This project is an end-to-end SQL analytics case study built using a synthetic but realistic **food delivery marketplace dataset** (inspired by Zomato, Swiggy, Blinkit).

It demonstrates strong SQL analytics, KPI design, data modeling, and a clean project architecture — all runnable using DuckDB with zero setup or installation.

The project includes:
- SQL analysis scripts  
- BI-ready exports  
- a documented relational schema  
- ready-to-run DuckDB workflows  
---

## Tools & Technologies

| Component | Choice |
|----------|--------|
| SQL Engine | DuckDB (CLI) |
| Query Language | Standard SQL |
| Visualization | Power BI-ready exports |
| Data Model | Normalized relational schema |
| Domain | Food Delivery Marketplace |

---

## Dataset Overview

This project includes ~12,000 orders and realistic marketplace entities:

### **Tables**
- **customers** – demographic and signup info  
- **restaurants** – cuisine type, location, prep times  
- **menu_items** – food items and pricing  
- **orders** – timestamps, payment method, status  
- **order_items** – per-item order breakdown  
- **delivery_riders** – rider info and activity  
- **ratings** – customer feedback + timestamps  

All tables are documented in `schema.sql`.
---

## Analytics Covered

### Customer Insights  
- MAU (Monthly Active Users)  
- Repeat customers  
- Top spenders  
- Churn risk signals  

### Restaurant Performance  
- GMV  
- AOV  
- Cuisine trends  
- Rating insights  
- Best menu items  

### Delivery Operations  
- Rider efficiency  
- SLA breach rate  
- Hourly demand patterns  
- Cancellation attribution  

### Business KPIs  
- GMV  
- Success Rate  
- Top-selling items  
- Delivery time vs Rating  

---

## Data Schema

The system follows a normalized relational model:

- customers (1) ──── (∞) orders
- restaurants (1) ──── (∞) orders
- orders (1) ──── (∞) order_items
- restaurants (1) ──── (∞) menu_items
- orders (1) ──── (1) ratings
- delivery_riders (1) ──── (∞) orders

## Author

This project is part of a data analytics portfolio demonstrating SQL, BI, and marketplace analytics expertise.
Designed to be simple to run, easy to extend, and realistic in terms of real-world business KPIs.

If you'd like to collaborate or discuss the project, feel free to reach out!
