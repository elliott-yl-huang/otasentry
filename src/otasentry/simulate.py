"""Synthetic EV fleet simulator.

Generates candump-format CAN logs for a fleet of vehicles before and after an
OTA update. Each log is a fresh, randomly styled drive cycle (city / mixed /
highway, different ambient temperature), so pre and post logs are *not*
directly comparable signal-by-signal -- exactly the problem real validation
campaigns face. Known regressions are injected into the post-update logs and
written to ``ground_truth.json`` so detectors can be scored.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cantools
import numpy as np

DT = 0.05  # physics step (s)

MASS = 2400.0
CDA = 0.62
RHO = 1.2
CRR = 0.009
G = 9.81
WHEEL_R = 0.36
GEAR = 9.0
N_SERIES = 96
CAPACITY_AH = 250.0
R_INT = 0.08

STYLES = {
    "city": {"speed": (20, 60), "accel": (1.0, 2.6), "stop_prob": 0.6, "cruise": (6, 20)},
    "mixed": {"speed": (35, 90), "accel": (0.9, 2.2), "stop_prob": 0.35, "cruise": (10, 35)},
    "highway": {"speed": (80, 125), "accel": (0.7, 1.6), "stop_prob": 0.12, "cruise": (25, 70)},
}

ECU_CODES = {"VCU": 1, "INV": 2, "BMS": 3, "HVAC": 4, "IVI": 5, "ABS": 6}


@dataclass
class InjectedRegression:
    id: str
    description: str
    kind: str
    target: str
    domain: str
    scope: str
    vehicles: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)


@dataclass
class ScenarioConfig:
    inverter_efficiency_drop: float = 0.08
    bms_cycle_jitter: bool = True
    hvac_new_dtc: bool = True
    park_at_speed_kmh: float = 4.0
    ivi_boot_delay_s: float = 5.0
    ivi_cpu_load_delta: float = 12.0
    wheel_sensor_fault_vehicle: int | None = 2  # index into fleet; None disables

    @classmethod
    def clean(cls) -> ScenarioConfig:
        return cls(
            inverter_efficiency_drop=0.0,
            bms_cycle_jitter=False,
            hvac_new_dtc=False,
            park_at_speed_kmh=0.0,
            ivi_boot_delay_s=0.0,
            ivi_cpu_load_delta=0.0,
            wheel_sensor_fault_vehicle=None,
        )


def _speed_profile(rng: np.random.Generator, n: int, style: dict) -> np.ndarray:
    """Piecewise drive cycle (m/s) that starts parked and ends parked.

    Speed is always continuous: every segment ramps with a bounded
    acceleration, and the final segment decelerates to standstill early
    enough to leave the vehicle parked for the last seconds of the log.
    """
    v: list[float] = [0.0] * int(rng.uniform(2.5, 4.0) / DT)
    budget = n - int(45.0 / DT)  # keep room for the final stop
    cur = 0.0

    def ramp(to: float, acc: float) -> None:
        nonlocal cur
        while abs(to - cur) > acc * DT:
            cur += acc * DT * np.sign(to - cur)
            v.append(max(cur, 0.0))
        cur = to
        v.append(cur)

    while len(v) < budget:
        ramp(rng.uniform(*style["speed"]) / 3.6, rng.uniform(*style["accel"]))
        ou = 0.0
        for _ in range(int(rng.uniform(*style["cruise"]) / DT)):
            if len(v) >= budget:
                break
            ou += -0.01 * ou + rng.normal(0, 0.012)
            v.append(max(cur + ou, 0.0))
        cur = v[-1]
        if len(v) < budget and rng.random() < style["stop_prob"]:
            ramp(0.0, rng.uniform(1.2, 2.6))
            v.extend([0.0] * int(rng.uniform(4, 12) / DT))
    ramp(0.0, rng.uniform(1.6, 2.6))
    arr = np.array(v[:n] + [0.0] * max(0, n - len(v)))
    # vehicle inertia / driveline compliance: low-pass the commanded profile
    k = int(0.8 / DT)
    smooth = np.convolve(np.pad(arr, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    smooth[smooth < 0.01] = 0.0
    return smooth


def _gear_profile(rng: np.random.Generator, v: np.ndarray, park_at_kmh: float) -> np.ndarray:
    """P at start/end and at some stops; optional early-park regression."""
    n = len(v)
    gear = np.full(n, 3, dtype=int)
    moving = np.flatnonzero(v > 0)
    first_move = moving[0] if len(moving) else n
    gear[: max(first_move - int(0.6 / DT), 0)] = 0
    stopped = v <= 0
    edges = np.diff(stopped.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if stopped[-1]:
        ends.append(n)
    starts = [s for s in starts if s > first_move]
    ends = [e for e in ends if e > first_move]
    for s, e in zip(starts, ends):
        final = e >= n
        if (e - s) * DT < 3.0:
            continue
        if not final and rng.random() > 0.4:
            continue
        p_start = s + int(1.0 / DT)
        if park_at_kmh > 0:
            j = s - 1
            while j > 0 and 0 < v[j] * 3.6 < park_at_kmh:
                j -= 1
            p_start = j + 1
        p_end = n if final else e - int(1.0 / DT)
        if p_end > p_start:
            gear[p_start:p_end] = 0
    return gear


def simulate_drive(
    rng: np.random.Generator,
    duration_s: float,
    ambient: float,
    style_name: str,
    efficiency_drop: float = 0.0,
    park_at_kmh: float = 0.0,
    cpu_load_delta: float = 0.0,
    wheel_fault: bool = False,
    setpoint: float = 21.0,
    wheel_bias: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    style = STYLES[style_name]
    n = int(duration_s / DT)
    v = _speed_profile(rng, n, style)
    gear = _gear_profile(rng, v, park_at_kmh)
    a = np.gradient(v, DT)
    a = np.convolve(a, np.ones(5) / 5, mode="same")

    f_aero = 0.5 * RHO * CDA * v**2
    f_roll = np.where(v > 0, CRR * MASS * G, 0.0)
    force = MASS * a + f_aero + f_roll
    motor_torque = np.clip(force * WHEEL_R / GEAR, -250, 600)
    motor_rpm = v / WHEEL_R * GEAR * 60 / (2 * np.pi)
    p_mech = motor_torque * motor_rpm * 2 * np.pi / 60

    eta = 0.94 - 0.07 * np.exp(-np.abs(p_mech) / 12000.0)
    eta = eta * (1.0 - efficiency_drop)

    pedal = np.where(
        (v > 0.3) & (a > -0.25),
        np.clip(6 + 28 * np.maximum(a, 0) + 0.22 * v * 3.6, 0, 100),
        0.0,
    )
    pedal = np.clip(pedal + rng.normal(0, 0.6, n) * (pedal > 0), 0, 100)
    brake = (a < -0.4).astype(int)

    soc = np.empty(n)
    volt = np.empty(n)
    cur = np.empty(n)
    t_inv = np.empty(n)
    t_mot = np.empty(n)
    t_cell = np.empty(n)
    cabin = np.empty(n)
    comp = np.empty(n)
    s = rng.uniform(55, 90) / 100
    ti = ambient + rng.uniform(0, 4)
    tm = ambient + rng.uniform(0, 4)
    tc = ambient + rng.uniform(1, 3)
    cab = ambient
    for i in range(n):
        need = abs(cab - setpoint)
        c = 250 + 180 * need if abs(ambient - setpoint) > 1.5 else 180.0
        c = min(c, 6000) + rng.normal(0, 25)
        cab += (setpoint - cab) * DT / 160.0 * (1 if c > 200 else 0.2)
        aux = 420 + c
        pm = p_mech[i]
        p_batt = (pm / eta[i] if pm >= 0 else pm * eta[i] * 0.92) + aux
        ocv = N_SERIES * (3.35 + 0.82 * s)
        disc = max(ocv**2 - 4 * R_INT * p_batt, 1.0)
        current = (ocv - np.sqrt(disc)) / (2 * R_INT)
        vt = ocv - current * R_INT
        s -= current * DT / (3600 * CAPACITY_AH)
        ti += (ambient + 9 + 0.05 * abs(current) - ti) * DT / 90.0
        tm += (ambient + 7 + 0.06 * abs(motor_torque[i]) - tm) * DT / 150.0
        tc += (ambient + 3 + 0.00005 * current**2 - tc) * DT / 500.0
        soc[i], volt[i], cur[i] = s, vt, current
        t_inv[i], t_mot[i], t_cell[i] = ti, tm, tc
        cabin[i], comp[i] = cab, max(c, 0)

    kmh = v * 3.6
    wheels = {}
    for name in ("FL", "FR", "RL", "RR"):
        bias = (wheel_bias or {}).get(name, 1.0) + rng.normal(0, 0.0003)
        w = kmh * bias + rng.normal(0, 0.05, n) * (kmh > 0)
        wheels[name] = w
    if wheel_fault:
        w = wheels["RL"] + rng.normal(0, 1.8, n) * (kmh > 0)
        t = 0
        while t < n:
            t += int(rng.uniform(6, 18) / DT)
            length = int(rng.uniform(0.3, 0.9) / DT)
            w[t : t + length] = 0.0
            t += length
        wheels["RL"] = w

    t_axis = np.arange(n) * DT
    load = 30 + 6 * np.sin(2 * np.pi * t_axis / 90) + rng.normal(0, 2.5, n) + cpu_load_delta
    load = np.clip(load, 0, 100)
    cpu_t = np.empty(n)
    ct = 42 + 0.3 * (ambient - 20)
    for i in range(n):
        ct += (40 + 0.35 * load[i] + 0.25 * (ambient - 20) - ct) * DT / 30.0
        cpu_t[i] = ct

    return {
        "VehicleSpeed": kmh,
        "AccelPedal": pedal,
        "BrakePressed": brake,
        "GearState": gear,
        "DriveMode": np.zeros(n, dtype=int),
        "MotorTorqueReq": motor_torque + rng.normal(0, 0.8, n),
        "MotorTorqueAct": motor_torque + rng.normal(0, 1.2, n),
        "MotorSpeed": motor_rpm,
        "InverterTemp": t_inv,
        "MotorTemp": t_mot,
        "PackVoltage": volt + rng.normal(0, 0.15, n),
        "PackCurrent": cur + rng.normal(0, 0.4, n),
        "SOC": soc * 100,
        "CellTempMax": t_cell,
        "CellVoltMin": volt / N_SERIES - 0.012 + rng.normal(0, 0.002, n),
        "WheelSpeed_FL": wheels["FL"],
        "WheelSpeed_FR": wheels["FR"],
        "WheelSpeed_RL": wheels["RL"],
        "WheelSpeed_RR": wheels["RR"],
        "CabinTemp": cabin,
        "AmbientTemp": np.full(n, ambient),
        "CompressorPower": comp,
        "HvacSetpoint": np.full(n, setpoint),
        "CpuTemp": cpu_t,
        "CpuLoad": load,
        "AppCrashCount": np.zeros(n, dtype=int),
    }


def _frame_times(rng, start, period, duration, jitter=False, drop=0.0):
    ts = np.arange(start, duration, period)
    if jitter:
        ts = ts + np.abs(rng.normal(0, period * 0.18, len(ts)))
        gap_starts = rng.choice(len(ts), size=max(1, len(ts) // 300), replace=False)
        keep = np.ones(len(ts), dtype=bool)
        for g in gap_starts:
            keep[g : g + int(rng.integers(2, 4))] = False
        keep &= rng.random(len(ts)) > drop
        ts = np.sort(ts[keep])
    else:
        ts = ts + rng.normal(0, 0.0004, len(ts))
    return ts[(ts >= 0) & (ts < duration)]


def write_log(
    path: Path,
    db: cantools.database.Database,
    rng: np.random.Generator,
    sig: dict[str, np.ndarray],
    duration_s: float,
    epoch: float,
    boot: dict[str, float],
    bms_jitter: bool,
    dtcs: list[tuple[str, int, float]],
) -> int:
    n = len(sig["VehicleSpeed"])
    lines: list[tuple[float, str]] = []
    counters = {"VCU_AliveCounter": 16, "INV_AliveCounter": 16, "IVI_AliveCounter": 256}
    for msg in db.messages:
        if not msg.cycle_time:
            continue
        period = msg.cycle_time / 1000.0
        sender = msg.senders[0]
        jitter = bms_jitter and msg.name == "BMS_Pack"
        ts = _frame_times(rng, boot.get(sender, 0.1), period, duration_s, jitter=jitter, drop=0.03)
        for k, t in enumerate(ts):
            idx = min(int(t / DT), n - 1)
            values = {}
            for s in msg.signals:
                if s.name in counters:
                    val = k % counters[s.name]
                else:
                    val = float(sig[s.name][idx])
                lo = s.minimum if s.minimum is not None else -np.inf
                hi = s.maximum if s.maximum is not None else np.inf
                values[s.name] = min(max(val, lo), hi)
            data = msg.encode(values, strict=False)
            lines.append((epoch + t, f"{msg.frame_id:03X}#{data.hex().upper()}"))

    dtc_msg = db.get_message_by_name("DIAG_DTC")
    for ecu, code, t_on in dtcs:
        for t in np.arange(t_on, duration_s, 1.0):
            data = dtc_msg.encode(
                {"DtcSourceEcu": ECU_CODES[ecu], "DtcCode": code, "DtcStatus": 0x2F}, strict=False
            )
            t2 = t + rng.uniform(0, 0.02)
            lines.append((epoch + t2, f"{dtc_msg.frame_id:03X}#{data.hex().upper()}"))

    lines.sort(key=lambda x: x[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for t, body in lines:
            fh.write(f"({t:.6f}) can0 {body}\n")
    return len(lines)


def simulate_campaign(
    out_dir: str | Path,
    dbc_path: str | Path,
    n_vehicles: int = 8,
    duration_s: float = 600.0,
    seed: int = 7,
    scenario: ScenarioConfig | None = None,
) -> dict:
    out = Path(out_dir)
    db = cantools.database.load_file(str(dbc_path))
    scenario = scenario or ScenarioConfig()
    rng = np.random.default_rng(seed)
    vins = [f"SIM{1000 + i:04d}" for i in range(n_vehicles)]
    styles = list(STYLES)

    hvac_vehicles = vins[:-1] if n_vehicles > 1 else vins
    fault_vin = (
        vins[scenario.wheel_sensor_fault_vehicle]
        if scenario.wheel_sensor_fault_vehicle is not None
        and scenario.wheel_sensor_fault_vehicle < n_vehicles
        else None
    )

    frames = 0
    for vi, vin in enumerate(vins):
        base_ambient = rng.uniform(4, 34)
        setpoint = float(rng.choice([20.0, 21.0, 22.0]))
        wheel_bias = {w: 1 + rng.normal(0, 0.002) for w in ("FL", "FR", "RL", "RR")}
        for phase in ("pre", "post"):
            post = phase == "post"
            ambient = float(np.clip(base_ambient + rng.normal(0, 4), -5, 40))
            style = str(rng.choice(styles))
            sig = simulate_drive(
                rng,
                duration_s,
                ambient,
                style,
                efficiency_drop=scenario.inverter_efficiency_drop if post else 0.0,
                park_at_kmh=scenario.park_at_speed_kmh if post else 0.0,
                cpu_load_delta=scenario.ivi_cpu_load_delta if post else 0.0,
                wheel_fault=post and vin == fault_vin,
                setpoint=setpoint,
                wheel_bias=wheel_bias,
            )
            ivi_boot = rng.uniform(1.4, 2.4) + (
                scenario.ivi_boot_delay_s + rng.uniform(-0.5, 0.8) if post and scenario.ivi_boot_delay_s else 0
            )
            boot = {"VCU": 0.05, "INV": 0.12, "BMS": 0.2, "ABS": 0.08, "HVAC": 0.6, "IVI": ivi_boot}
            dtcs: list[tuple[str, int, float]] = []
            if vi == 1:
                dtcs.append(("IVI", 0xC14100, rng.uniform(5, 60)))
            if post and scenario.hvac_new_dtc and vin in hvac_vehicles:
                dtcs.append(("HVAC", 0x9A2316, rng.uniform(30, duration_s / 2)))
            epoch = 1_760_000_000 + vi * 86400 + (7 * 86400 if post else 0)
            frames += write_log(
                out / phase / f"{vin}.log",
                db,
                rng,
                sig,
                duration_s,
                epoch,
                boot,
                bms_jitter=post and scenario.bms_cycle_jitter,
                dtcs=dtcs,
            )

    truth: list[InjectedRegression] = []
    if scenario.inverter_efficiency_drop > 0:
        truth.append(InjectedRegression(
            "inverter_efficiency", f"Inverter efficiency reduced by {scenario.inverter_efficiency_drop:.0%}",
            "behavior", "PackCurrent", "software", "fleet",
            related=["InverterTemp", "MotorTemp", "CellTempMax", "CellVoltMin", "PackVoltage", "SOC"],
        ))
    if scenario.bms_cycle_jitter:
        truth.append(InjectedRegression(
            "bms_cycle_jitter", "BMS_Pack transmit jitter and dropped frames", "timing", "BMS_Pack",
            "communication", "fleet", related=["REQ-NET-005"],
        ))
    if scenario.hvac_new_dtc:
        truth.append(InjectedRegression(
            "hvac_new_dtc", "New HVAC DTC 0x9A2316 on most of the fleet", "dtc", "HVAC:0x9A2316",
            "software", "fleet", related=["REQ-DIAG-007"],
        ))
    if scenario.park_at_speed_kmh > 0:
        truth.append(InjectedRegression(
            "park_at_speed", f"Park engages below {scenario.park_at_speed_kmh} km/h instead of at standstill",
            "requirement", "REQ-VC-001", "vehicle-system", "fleet", related=["GearState"],
        ))
    if scenario.ivi_boot_delay_s > 0:
        truth.append(InjectedRegression(
            "ivi_boot_delay", "Infotainment heartbeat delayed after wake", "startup", "IVI_Heartbeat",
            "software", "fleet", related=["REQ-IVI-006"],
        ))
    if scenario.ivi_cpu_load_delta > 0:
        truth.append(InjectedRegression(
            "ivi_cpu_load", "Infotainment CPU load increased", "behavior", "CpuLoad", "software", "fleet",
            related=["CpuTemp"],
        ))
    if fault_vin:
        truth.append(InjectedRegression(
            "wheel_sensor_fault", "Intermittent rear-left wheel speed sensor (hardware, not the update)",
            "behavior", "WheelSpeed_RL", "hardware", "vehicle", vehicles=[fault_vin],
            related=["REQ-CH-002"],
        ))

    meta = {
        "campaign": out.name,
        "vehicles": vins,
        "duration_s": duration_s,
        "seed": seed,
        "frames": frames,
        "injected": [asdict(t) for t in truth],
    }
    (out / "ground_truth.json").write_text(json.dumps(meta, indent=2))
    return meta
