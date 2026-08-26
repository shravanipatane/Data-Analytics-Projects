# Week 3 — Advanced Data Analysis & Visualization in Logistics 🏍️📦

**Domain:** Two-Wheeler (Bike / Scooter / E-Bike) Last-Mile Delivery Logistics
**Tools:** Python · Pandas · NumPy · Matplotlib · Seaborn

An end-to-end exploratory data analysis (EDA) and visualization project on a
simulated two-wheeler last-mile delivery operation — analyzing delivery
performance, cost drivers, and operational bottlenecks across a fleet of
scooters, motorbikes, and e-bikes.

---

## 📌 Project Objective

Analyze a two-wheeler logistics dataset to answer:

- What drives delivery time — distance, traffic, or weather?
- Which vehicle type is the most cost-efficient vs. the fastest?
- Which delivery zones are operational bottlenecks?
- How does delivery performance impact customer satisfaction?

---

## 📁 Repository Structure

```
Week-3-Advanced-EDA-Visualization/
│
├── README.md
├── data/
│   └── two_wheeler_logistics.csv        # 2,000-row simulated dataset
├── scripts/
│   ├── 01_generate_data.py              # Dataset simulation
│   ├── 02_eda_analysis.py               # Descriptive stats & correlations
│   └── 03_visualizations.py             # All chart generation
├── visualizations/
│   ├── 01_delivery_time_dist.png
│   ├── 02_delaybox_traffic.png
│   ├── 03_distance_vs_time.png
│   ├── 04_avg_cost_vehicle.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_delay_rate_zone.png
│   ├── 07_daily_trend.png
│   └── 08_vehicle_share_pie.png
└── report/
    ├── Week3_Logistics_Analysis_Report.docx   # Full written report
    └── eda_summary.txt                        # Raw EDA console output
```

---

## 🗂️ Dataset

A realistic 2,000-record dataset was simulated (`scripts/01_generate_data.py`)
with deliberately engineered relationships (e.g. traffic and rain slow down
deliveries and raise delay risk) so the analysis reflects genuine logistics
behaviour.

| Column | Description |
|---|---|
| `delivery_id`, `date`, `rider_id` | Record identifiers |
| `vehicle_type` | Scooter / Motorbike / E-Bike |
| `zone` | One of 5 delivery zones |
| `weather`, `traffic_condition`, `time_of_day` | Environmental context |
| `distance_km`, `shipment_weight_kg` | Trip characteristics |
| `delivery_time_min`, `sla_minutes`, `delayed` | Performance vs SLA |
| `fuel_cost_inr`, `delivery_cost_inr` | Cost metrics (INR) |
| `customer_rating` | 1–5 satisfaction score |

---

## 🔍 Key Insights

- **Distance is king:** distance correlates strongly with both delivery time (r = 0.81) and fuel cost (r = 0.82).
- **Speed vs. satisfaction:** delivery time is the strongest driver of customer rating (r = -0.62) — every minute of delay costs satisfaction.
- **Fleet trade-off:** Motorbikes are fastest (18.5 min avg, 5% delay rate); E-bikes are ~40% cheaper to run but have the highest delay rate (16%).
- **Bottleneck zone:** Zone A (Central) has the highest delay rate (14%) despite not having the longest average distance — a congestion/SLA design issue, not a distance issue.
- **Weather impact:** Rain increases average delivery time by ~32% and roughly triples the delay rate versus clear weather.

Full analysis, all 8 visualizations, and recommendations are in
[`report/Week3_Logistics_Analysis_Report.docx`](report/Week3_Logistics_Analysis_Report.docx).

---

## ▶️ How to Reproduce

```bash
pip install pandas numpy matplotlib seaborn

cd scripts
python 01_generate_data.py       # generates data/two_wheeler_logistics.csv
python 02_eda_analysis.py        # prints + saves EDA summary
python 03_visualizations.py      # generates all 8 charts
```

---

## 🛠️ Tech Stack

- **Python 3** — pandas, numpy (data simulation & analysis)
- **Matplotlib, Seaborn** — visualization
- **python-docx / docx-js** — report generation

---

## 📄 License

This is an educational/portfolio project using simulated data. Free to reference or reuse.
