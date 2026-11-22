# Model card: I-94 hourly traffic volume

| | |
|---|---|
| Models | `hgb-a5959e97b784` (calendar and weather) and `hgb-22eddc348a3d` (calendar only) |
| Estimator | HistGradientBoosting; settings: `l2_regularization` = 2.9154, `learning_rate` = 0.1019, `max_iter` = 201, `max_leaf_nodes` = 10, `min_samples_leaf` = 31 |
| Format | ONNX, 139 KB each |
| Endpoint | https://traffic-volume-i94.vercel.app |
| Trained on | Hours from 2012-10-02 to the end of 2017 |

## Intended use

Forecasting westbound vehicles per hour at MnDOT station 301 (I-94, between Minneapolis and St Paul) days or
weeks ahead, for planning: putting roadworks into quiet hours and preparing for busy ones.

**Not intended** for other roads or the eastbound direction, for detecting incidents as they happen, for
next-hour operations (a real-time model would use recent counts and is far more accurate at that horizon), or
for years long after 2018 without retraining.

## Inputs

| Input | Required? | Notes |
|---|---|---|
| Date and hour, local time | Yes | Gives the hour, the weekday and the month |
| Public holiday | No | Worked out from the date. Only six quiet holidays count: New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving Day, Christmas Day |
| Weather (temperature, type, cloud cover; rain and snow amounts) | No | When given, the calendar-and-weather model answers; otherwise the calendar-only model |

**Not used:** the year (the model cannot extrapolate a trend) and recent traffic counts (not known when
planning ahead).

## Data

Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository,
https://doi.org/10.24432/C5X60B, CC BY 4.0. 48,204 rows; 40,518 hours after
cleaning. 23% of the hours in the period are missing from the source. Cleaning removes exact
duplicates, sensor errors and 46 road-closure hours, keeps one row per hour, and
repairs the holiday marking.

## Performance on the test year (January to September 2018)

| | RMSE | MAE | R² |
|---|---|---|---|
| Hour-of-week average (baseline) | 468 | 265 | 0.944 |
| The same average, aware of quiet holidays | 396 | 237 | 0.960 |
| Calendar-only model | 377 | 223 | 0.963 |
| **Calendar-and-weather model (champion)** | **368** | **221** | **0.965** |

Champion, 95% intervals (block bootstrap over whole days): RMSE 312 to 427,
R² 0.953 to 0.975. Its RMSE is 21% below the
baseline's (9% to 33%) and
7% below the holiday-aware table's
(4% to 10%).

The champion's figures assume the weather of the hour is known exactly, as in the data. With a forecast the
truth lies between the champion's row and the calendar-only row.

**The test year was scored twice.** The first evaluation (a random forest, all eleven named days as holidays)
gave R² 0.964, RMSE 376. Four defects were then fixed using training and
validation data only, and the figures above are the second evaluation. See `docs/PROCESS.md`, section 8.1.

### By kind of hour (champion, test year)

| Kind of hour | Hours | Baseline RMSE | Holiday-aware table | Champion RMSE | RMSE reduction against the baseline | Share of champion's squared error |
|---|---|---|---|---|---|---|
| Weekday rush hours | 1,525 | 508 | 501 | 448 | 12% | 35% |
| Weekday midday | 955 | 313 | 318 | 285 | 9% | 9% |
| Weekday evening | 764 | 365 | 365 | 341 | 7% | 10% |
| Weekday night | 1,331 | 189 | 181 | 180 | 5% | 5% |
| Weekend | 1,862 | 453 | 453 | 438 | 3% | 40% |
| Quiet holiday | 96 | 2,072 | 405 | 352 | 83% | 1% |

### By weather (test year)

| Weather | Hours | Baseline RMSE | Calendar-only model | Champion RMSE | Share of champion's squared error |
|---|---|---|---|---|---|
| Clear or cloudy | 4,324 | 415 | 296 | 297 | 43% |
| Low visibility | 604 | 271 | 234 | 230 | 4% |
| Rain or storm | 1,089 | 438 | 264 | 265 | 9% |
| Snow | 516 | 917 | 926 | 875 | 45% |

## Limitations and risks

- **Snow.** 45% of the test error comes from the 8% of hours with snow; the
  data does not record how much snow fell, so the model cannot tell a flurry from a blizzard.
- **Unusual days.** The day after Thanksgiving, the days between Christmas and New Year, incidents and
  closures cause the largest errors and are invisible to the model.
- **One station and direction.** Patterns elsewhere differ.
- **The test year lacks October to December**, the hardest quarter on the validation year
  (RMSE 411 against 323),
  so a full year would probably score somewhat worse.
- **No trend.** A change to the road (a new lane or interchange) would shift the pattern until the model is
  retrained.

## Monitoring

Weekly RMSE on hours whose real count has arrived, against an alert line of 400 (validation RMSE
plus 15%); retrain after three weeks in a row above it, and every January. Input drift by PSI; latency and
error rate from the logs. Details in `docs/PROCESS.md`, section 10.
