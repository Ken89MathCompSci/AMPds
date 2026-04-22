# AMPds2 Dataset Overview

**Source:** The Almanac of Minutely Power Dataset (v2)
**Institution:** Simon Fraser University — Makonin, Popowich, Bartram, Gill, Bajic
**Location:** Burnaby, BC, Canada (49.269°N, 122.992°W)
**Contact:** stephen@makonin.com

---

## Coverage

| Property | Value |
|----------|-------|
| Time range | 2012-04-01 → 2014-03-31 |
| Duration | 2 years (exactly) |
| Resolution | 1 sample per minute |
| Rows per meter | 1,051,200 |
| Timezone | America/Vancouver |

---

## File Structure

```
AMPds2.h5
└── building1/
    └── elec/
        ├── meter1          ← Aggregate (site meter)
        ├── meter2–meter21  ← Individual sub-meters
        └── cache/
            └── meterN/
                └── total_energy
```

Each meter dataset contains **11 columns**:

| Column | Type |
|--------|------|
| voltage | apparent |
| current | apparent |
| frequency | apparent |
| power factor | apparent / real |
| power | active / reactive / apparent |
| energy | active / reactive / apparent |

---

## Appliance Map

| Meter | Code | Appliance | Room | 2-Year Energy |
|-------|------|-----------|------|--------------|
| 1 | — | **Aggregate (main meter)** | — | **19,488 kWh** |
| 2 | B1E | Lights & Plugs | North Bedroom | 18 kWh |
| 3 | B2E | Lights & Plugs | Master/South Bedroom | 427 kWh |
| 4 | BME | Lights & Plugs | Basement | 672 kWh |
| 5 | CDE | Clothes Dryer | Basement | 936 kWh |
| 6 | CWE | Clothes Washer | Basement | 77 kWh |
| 7 | DNE | Plugs | Dining Room | 12 kWh |
| 8 | DWE | Dishwasher | Kitchen | 262 kWh |
| 9 | EBE | Electronics Workbench | Home Office | 212 kWh |
| 10 | EQE | Security/Network Equipment | Basement | 699 kWh |
| 11 | FGE | Fridge | Kitchen | 884 kWh |
| 12 | FRE | Furnace Fan & Thermostat | Basement | 2,023 kWh |
| 13 | GRE | Garage Sub-Panel | Garage | 27 kWh |
| 14 | HPE | Heat Pump | Outside | 2,949 kWh |
| 15 | HTE | Instant Hot Water Unit | Basement | 128 kWh |
| 16 | OFE | Lights & Home Office | Home Office | 621 kWh |
| 17 | OUE | Outdoor Plugs | Outside | ~0 kWh |
| 18 | RSE | Rental Suite Sub-Panel | Basement | 4,389 kWh |
| 19 | TVE | TV / PVR / Amp | Rec Room | 718 kWh |
| 20 | UTE | Utility Plug | Basement | 822 kWh |
| 21 | WOE | Wall Oven | Kitchen | 138 kWh |

---

## Aggregate Power Statistics (meter1)

| Stat | Value |
|------|-------|
| Min | 0 W |
| Max | 12,260 W |
| Mean | 1,112 W |
| Median | 758 W |

---

## Top Energy Consumers (2-year share of aggregate)

| Appliance | Energy | Share |
|-----------|--------|-------|
| Rental Suite (RSE) | 4,389 kWh | 22.5% |
| Heat Pump (HPE) | 2,949 kWh | 15.1% |
| Furnace Fan (FRE) | 2,023 kWh | 10.4% |
| Fridge (FGE) | 884 kWh | 4.5% |
| Clothes Dryer (CDE) | 936 kWh | 4.8% |

---

## Meter Device

**DENT Instruments PowerScout 18 (PS18)** — Branch circuit power meter
- Sample period: 60 s
- Voltage range: 0–270 V
- Current range: 0–400 A

---

## Notes

- Meter 1 is the site-level aggregate; meters 2–21 are all direct sub-meters of the main panel (not nested).
- This is a standard benchmark dataset for **NILM (Non-Intrusive Load Monitoring)** research.
- Schema follows [nilmtk/nilm_metadata v0.2](https://github.com/nilmtk/nilm_metadata/tree/v0.2).
