# Benchmark

Synthetic fleets of 8 vehicles, 600 s drives per phase, 5 campaigns with 7 injected issues and 3 clean campaigns. Each pre/post drive has an independently random style (city / mixed / highway), ambient temperature and state of charge. Reproduce with `python scripts/benchmark.py`.

| Method | Recall | Precision | False alarms / faulty campaign | False alarms / clean campaign |
|---|---|---|---|---|
| otasentry (context-matched residuals + fleet logic) | 1.00 | 0.97 | 0.6 | 0.0 |
| Naive pre/post signal diff (KS test, same fleet logic) | 0.31 | 0.53 | 5.8 | 10.3 |

Per-issue detection rate (otasentry / naive) and domain attribution accuracy (otasentry):

| Injected issue | Detected | Naive detected | Domain correct |
|---|---|---|---|
| `inverter_efficiency` | 100% | 100% | 100% |
| `bms_cycle_jitter` | 100% | 0% | 100% |
| `hvac_new_dtc` | 100% | 0% | 100% |
| `park_at_speed` | 100% | 0% | 100% |
| `ivi_boot_delay` | 100% | 0% | 100% |
| `ivi_cpu_load` | 100% | 100% | 100% |
| `wheel_sensor_fault` | 100% | 20% | 100% |

Release gate per campaign (seed:decision): 1:NO-GO, 2:NO-GO, 3:NO-GO, 4:NO-GO, 5:NO-GO, 101:GO, 102:GO, 103:GO

Mean analysis time: 4.8 s for 2.67 h of CAN data per campaign.

otasentry false positives observed (kind:target:scope:vehicles):

- `behavior:CompressorPower:vehicle:SIM1004`
- `behavior:MotorTorqueAct:vehicle:SIM1003`
- `behavior:MotorTorqueReq:vehicle:SIM1003`
