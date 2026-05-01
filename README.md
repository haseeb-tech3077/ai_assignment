# Smart Rickshaw Route Planner

A Streamlit-based AI application that finds the optimal route for rickshaw drivers
across a simulated Rawalpindi/Islamabad city road network using the **A\* (A-Star)
Informed Search Algorithm**.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the App
```bash
streamlit run app.py
```

The app will open at localhost in your browser.

---

## File Structure

```
astar_route_planner/
│
├── app.py              Main working app
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## Algorithm Overview

### A* Search
- **Evaluation function:** `f(n) = g(n) + h(n)`
  - `g(n)` = actual cost from source to node n (road distance × traffic multiplier)
  - `h(n)` = Euclidean straight-line distance from n to goal (admissible heuristic)
- **Data structures:** Min-heap priority queue (open list) + hash set (closed list)
- **Guarantees:** Complete + Optimal (with admissible, consistent heuristic)

### Heuristic Function
```
h(n) = sqrt((x_n - x_goal)^2 + (y_n - y_goal)^2)
```

---

## Road Network

- **25 nodes:** Named intersections across Rawalpindi/Islamabad
  (e.g., Saddar, Committee Chowk, Faizabad, Blue Area, Bahria Town Gate...)
- **40+ edges:** Road segments with base distances in km
- **Traffic modes:**
  - Off-Peak (multiplier: 1.0×)
  - Morning Rush, 7–10 AM (multiplier: 1.6×)
  - Evening Rush, 5–8 PM (multiplier: 1.8×)

---

## Sample Output

**Query:** Saddar → Blue Area, Evening Rush

| Metric               | A*     
|----------------------|------
| Distance (km)        | 4.97
| Nodes Explored       | 8
| Path Length (stops)  | 4

**Optimal Path:** Saddar → Liaquat Bagh → Faizabad → G-9 → Blue Area

---

## 🛠️ Technical Notes

- All code follows OOP principles with separate classes for graph, solver, and visualizer
- Input validation handles same-source/destination and disconnected graph cases
- Traffic weights are multiplied onto base edge distances at graph construction time
