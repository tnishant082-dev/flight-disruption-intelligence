# Model card: Airport and route segmentation

## Intended use

Group airports / routes with similar operating profiles so network planners can compare like with like and target interventions by segment.

## Data

marts.airport_profile (172 airports with >= 3,000 departures) and marts.kpi_route (4457 routes with >= 300 flights), full study window.

## Method

StandardScaler + KMeans (n_init=20); k in 4-7 picked by silhouette. Labels are derived from the centroid: size tier plus the two most distinctive traits (largest absolute z-score).

## Airports segments (k=6, silhouette 0.195)

- **Regional - volatile delays, delay-prone**: 27 airports
- **Mid-size - ATC/NAS-constrained, low weather exposure**: 44 airports
- **Regional - short-haul mix, few NAS delays**: 35 airports
- **Major hub - evening-heavy bank, long taxi-out / congested**: 29 airports
- **Mid-size - stable delays, quick turn**: 14 airports
- **Regional - few knock-on delays, morning-heavy bank**: 23 airports

## Routes segments (k=4, silhouette 0.224)

- **Trunk route - multi-carrier, unreliable**: 1071 routes
- **Thin route - long delay tail, high average delay**: 886 routes
- **Thin route - reliable, short delay tail**: 1843 routes
- **Thin route - long-haul, rarely cancels**: 657 routes

## Limitations

Silhouette scores are modest - operating profiles form a continuum, so segments are a descriptive lens, not hard categories. Profiles use outcome data from the whole window (descriptive, not predictive).
