"""Write the tiny hand-built raw files used by the tests.

Run ``python tests/fixtures/build_fixtures.py`` to regenerate. Every loan has a story
so the expected numbers can be checked on paper (all balances in dollars):

Release 47 layout (sample_orig_2015 / sample_svcg_2015), 8 months 2015-03 .. 2015-10:
  F15Q10000001  100,000, FICO 760, LTV 75   voluntary payoff (zero balance 01) in month 3
  F15Q10000002  100,000, FICO 700, LTV 85   performs all 8 months
  F15Q10000003  100,000, FICO 650, LTV 95   delinquent 00,00,01,..,06 -> defaults month 8
  F15Q10000004   10,000, FICO 720, LTV 50   8-month term; paid off at maturity (not a prepay)
  F15Q10000005  100,000, FICO n/a, LTV n/a  sentinels 9999 / 999 load as null; performs
  F15Q10000006  100,000, FICO 690, LTV 90   balance 80,000; 90+ days late from month 2; short sale
                                            (zero balance 03) in month 5, removal UPB
                                            80,000, actual loss -20,000
Legacy layout (historical_data_2019Q1 / historical_data_time_2019Q1): 2 loans, 3 months.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
PERIODS = ["201503", "201504", "201505", "201506", "201507", "201508", "201509", "201510"]


def orig_r47(fico, ltv, upb, rate, term, maturity, loan_id, purpose="P", occ="P"):
    f = [fico, "201503", "N", maturity, "12345", "0", "1", occ, ltv, "35", upb, ltv, rate,
         "R", "N", "FRM", "VA", "SF", "221", loan_id, purpose, term, "2", "OTHER", "N", "",
         "", "N", "2", "N", "9999"]
    assert len(f) == 31
    return "|".join(str(x) for x in f)


def perf_r47(loan_id, period, upb, dq, age, remaining, zb="", zb_date="", rate="4.000",
             loss="", removal=""):
    f = [loan_id, period, upb, dq, age, remaining, "", "", zb, zb_date, rate, "0.00",
         "", "", "", "", "", "", "", "", "", loss, "0.00", "", "", "75", removal, "", "", "",
         "", upb, "7", "OTHER", "0.00"]
    assert len(f) == 35
    return "|".join(str(x) for x in f)


def r47_files() -> tuple[list[str], list[str]]:
    orig = [
        orig_r47(760, 75, 100000, "4.000", 360, "204502", "F15Q10000001"),
        orig_r47(700, 85, 100000, "4.000", 360, "204502", "F15Q10000002", purpose="C"),
        orig_r47(650, 95, 100000, "4.500", 360, "204502", "F15Q10000003"),
        orig_r47(720, 50, 10000, "3.500", 8, "201510", "F15Q10000004", purpose="N"),
        orig_r47(9999, 999, 100000, "4.000", 360, "204502", "F15Q10000005", occ="I"),
        orig_r47(690, 90, 100000, "4.250", 360, "204502", "F15Q10000006"),
    ]
    perf = []
    # Loan 1: constant balance, voluntary payoff in month 3 (row shows zero balance).
    for i, period in enumerate(PERIODS[:3]):
        payoff = i == 2
        perf.append(perf_r47("F15Q10000001", period, "0.00" if payoff else "100000.00", "00",
                             i + 1, 359 - i, zb="01" if payoff else "",
                             zb_date=period if payoff else ""))
    # Loan 2: performs throughout.
    for i, period in enumerate(PERIODS):
        perf.append(perf_r47("F15Q10000002", period, "100000.00", "00", i + 1, 359 - i))
    # Loan 3: rolls 30, 60, ... to 180 days delinquent in month 8.
    for i, (period, dq) in enumerate(zip(PERIODS, ["00", "00", "01", "02", "03", "04", "05",
                                                   "06"], strict=True)):
        perf.append(perf_r47("F15Q10000003", period, "100000.00", dq, i + 1, 359 - i,
                             rate="4.500"))
    # Loan 4: amortises 10,000 -> 0 over its 8-month term; final payoff at maturity.
    for i, period in enumerate(PERIODS):
        last = i == 7
        perf.append(perf_r47("F15Q10000004", period, "0.00" if last else f"{8750 - 1250 * i}.00",
                             "00", i + 1, 7 - i, zb="01" if last else "",
                             zb_date=period if last else "", rate="3.500"))
    # Loan 5: performs throughout.
    for i, period in enumerate(PERIODS):
        perf.append(perf_r47("F15Q10000005", period, "100000.00", "00", i + 1, 359 - i))
    # Loan 6: short sale in month 5 with a 20,000 loss on 80,000 removed UPB.
    for i, period in enumerate(PERIODS[:5]):
        sale = i == 4
        perf.append(perf_r47("F15Q10000006", period, "0.00" if sale else "80000.00",
                             "03" if i >= 1 else "00", i + 1, 359 - i,
                             zb="03" if sale else "", zb_date=period if sale else "",
                             rate="4.250", loss="-20000.00" if sale else "",
                             removal="80000.00" if sale else ""))
    return orig, perf


def legacy_files() -> tuple[list[str], list[str]]:
    def orig(loan_id, fico):
        f = [fico, "201903", "N", "204902", "", "0", "1", "P", "80", "30", "200000", "80",
             "4.750", "R", "N", "FRM", "TX", "SF", "750", loan_id, "P", "360", "2", "Other sellers",
             "Other servicers", "", "", "9", "", "2", "N", "7"]
        assert len(f) == 32
        return "|".join(f)

    def perf(loan_id, period, age, dq):
        f = [loan_id, period, "200000.00", dq, str(age), str(360 - age), "", "", "", "",
             "4.750", "0.00", "", "", "", "", "", "", "", "", "", "", "", "", "", "80", "",
             "", "", "", "", "200000.00"]
        assert len(f) == 32
        return "|".join(f)

    origs = [orig("F19Q10000001", "740"), orig("F19Q10000002", "9999")]
    perfs = [perf(lid, p, a, dq) for lid in ("F19Q10000001", "F19Q10000002")
             for a, (p, dq) in enumerate([("201903", "0"), ("201904", "0"), ("201905", "1")], 1)]
    return origs, perfs


def pmms_weekly() -> list[str]:
    """Weekly PMMS for 2015 (FRED CSV format): 3.80% in Jan-Feb, 3.70% after, one gap."""
    import datetime as dt

    rows = ["observation_date,MORTGAGE30US"]
    day = dt.date(2015, 1, 1)
    while day.year == 2015:
        value = "3.80" if day.month <= 2 else "3.70"
        if day == dt.date(2015, 3, 5):
            value = "."  # FRED marks missing values with a dot
        rows.append(f"{day.isoformat()},{value}")
        day += dt.timedelta(days=7)
    return rows


def main() -> None:
    raw = HERE / "raw"
    legacy = HERE / "legacy"
    raw.mkdir(exist_ok=True)
    legacy.mkdir(exist_ok=True)
    o, p = r47_files()
    (raw / "sample_orig_2015.txt").write_text("\n".join(o) + "\n")
    (raw / "sample_svcg_2015.txt").write_text("\n".join(p) + "\n")
    o, p = legacy_files()
    (legacy / "historical_data_2019Q1.txt").write_text("\n".join(o) + "\n")
    (legacy / "historical_data_time_2019Q1.txt").write_text("\n".join(p) + "\n")
    (raw / "MORTGAGE30US.csv").write_text("\n".join(pmms_weekly()) + "\n")


if __name__ == "__main__":
    main()
