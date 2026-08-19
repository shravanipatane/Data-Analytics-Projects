# 🚚 Week 2 — Data Collection, Cleaning & Preprocessing

## 📌 Project: Two-Wheeler Last-Mile Logistics Analytics

This project is a continuation of the Week 1 strategic planning project for **TwoWheel Express**, a two-wheeler last-mile delivery operation in the Thane–Mumbai corridor.

The objective of Week 2 is to collect, inspect, clean, transform, and prepare logistics data for further analysis and predictive modeling.

---

## 🎯 Objectives

The main objectives of this week are:

- Understand the structure of the logistics dataset.
- Inspect data types and data quality.
- Identify missing values.
- Detect and remove duplicate records.
- Identify and handle outliers.
- Standardize categorical variables.
- Transform numerical variables where required.
- Prepare an analysis-ready dataset.
- Document the complete preprocessing workflow.

---

## 🚚 Business Context

TwoWheel Express operates last-mile delivery services across several locations in the Thane–Mumbai corridor.

The dataset contains information about delivery operations, including:

- Delivery zones
- Vehicle types
- Traffic conditions
- Weather conditions
- Delivery time slots
- Delivery distance
- Order weight
- Rider information
- Delivery time
- Promised delivery time
- Delivery performance
- Operating cost

The cleaned dataset will be used for the next stages of the logistics analytics project.

---

## 📊 Dataset

The project uses a logistics dataset representing delivery orders handled by a two-wheeler delivery fleet.

### Key Variables

| Variable | Description |
|---|---|
| `order_id` | Unique identifier for each delivery |
| `zone` | Delivery operating zone |
| `vehicle_type` | Type of two-wheeler used |
| `traffic_level` | Traffic condition during delivery |
| `weather` | Weather condition |
| `time_slot` | Delivery time period |
| `distance_km` | Delivery distance in kilometres |
| `order_weight_kg` | Package weight |
| `rider_id` | Delivery rider identifier |
| `delivery_time_min` | Actual delivery time |
| `promised_time_min` | Expected delivery time |
| `on_time` | Indicates whether delivery was on time |
| `cost_per_km` | Operating cost per kilometre |
| `total_cost` | Total delivery operating cost |

---

## 🔍 Data Inspection

The raw dataset was inspected to understand its structure, dimensions, data types, descriptive statistics, missing values, and duplicate records.

Examples of Python commands used:

```python
df.head()
df.info()
df.shape
df.describe()
df.isnull().sum()
df.duplicated().sum()
