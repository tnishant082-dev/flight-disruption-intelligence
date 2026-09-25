from datetime import date

import pandas as pd

from flightops.features import ALL_FEATURES, ROTATION_FEATURES, holiday_table, prepare


def test_holiday_distance():
    h = holiday_table("2026-06-01", "2026-07-31").set_index("flight_date")
    assert h.loc[date(2026, 7, 3), "days_to_holiday"] == 0     # Independence Day observed (Fri)
    assert h.loc[date(2026, 7, 1), "days_to_holiday"] == 2
    assert h.loc[date(2026, 6, 19), "is_holiday"] == 1         # Juneteenth


def test_predeparture_features_exclude_rotation_and_outcomes():
    leaky = set(ROTATION_FEATURES) | {"arr_delay_min", "arr_del15", "cancelled", "dep_delay_min",
                                      "taxi_out_min", "day_of_month"}
    assert not leaky & set(ALL_FEATURES)


def test_prepare_uses_fixed_categories():
    df = prepare(pd.DataFrame({"carrier": ["AA"], "origin": ["ORD"], "dest": ["LGA"],
                               "distance_mi": [733]}))
    assert str(df["carrier"].dtype) == "category"
    assert "WN" in df["carrier"].cat.categories
    assert df["distance_mi"].dtype == "float32"
