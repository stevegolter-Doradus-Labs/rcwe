"""
Awkward real-world inputs for `load_table`.

Each case here was a silent wrong result or an unhelpful failure before it was fixed: dates
read with the wrong order, thousands separators taken for decimals, clocks taken for sensors.
"""
import json

import numpy as np
import pytest

from rcwe.toolkit import DataError, load_table

EPOCH = 1767225600          # 2026-01-01T00:00:00Z


# ---- dates ---------------------------------------------------------------------------
def test_day_first_dates_are_read_with_one_order_for_the_whole_column():
    table = load_table(
        "Date,temp,act_valve,anomaly\n11/03/2026 12:00,0,10,0\n12/03/2026 00:00,1,11,0\n12/03/2026 12:00,2,12,0\n"
        "13/03/2026 00:00,3,13,1\n13/03/2026 12:00,4,14,1\n"
    )
    assert table.values.ravel().tolist() == [0, 1, 2, 3, 4]              # file order kept: nothing was "sorted"
    assert table.labels.tolist() == [0, 0, 0, 1, 1] and table.sample_interval == 43200.0
    assert any("day/month/year" in note for note in table.notes)

    american = load_table("when,v\n03/11/2026 12:00,0\n03/12/2026 00:00,1\n03/13/2026 00:00,2\n03/14/2026 00:00,3\n")
    assert american.values.ravel().tolist() == [0, 1, 2, 3] and any("month/day/year" in note for note in american.notes)

    # Every date reads both ways: the evenly spaced reading wins, and the doubt is reported.
    daily = load_table("when,v\n01/03/2026 00:00,0\n02/03/2026 00:00,1\n03/03/2026 00:00,2\n04/03/2026 00:00,3\n")
    assert daily.sample_interval == 86400.0 and any("day/month/year" in note for note in daily.notes)


def test_date_and_time_in_separate_columns_make_one_time_axis():
    table = load_table("Date,Time,temp\n2026-03-01,23:58:00,1\n2026-03-01,23:59:00,2\n2026-03-02,00:00:00,3\n2026-03-02,00:01:00,4\n")
    assert table.sample_interval == 60.0 and table.sensors == ["temp"] and table.timestamps[2] == "2026-03-02 00:00:00"
    compact = load_table("date,time,temp\n20260301,2350,1\n20260301,2355,2\n20260302,0000,3\n20260302,0005,4\n")
    assert compact.sample_interval == 300.0 and compact.sensors == ["temp"]


def test_the_repeated_hour_when_daylight_saving_ends_is_not_shuffled():
    clock = ["01:30", "01:45", "02:00", "02:15", "02:30", "02:45", "02:00", "02:15", "02:30", "02:45", "03:00"]
    table = load_table("when,v\n" + "\n".join(f"2026-10-25 {c},{i}" for i, c in enumerate(clock)))
    assert table.values.ravel().tolist() == list(range(11))
    assert np.all(np.diff(table.time_seconds) > 0) and any("daylight" in note for note in table.notes)


# ---- numeric clocks ------------------------------------------------------------------
@pytest.mark.parametrize("name, scale", [("timestamp", 1), ("unix_time", 1), ("timestamp_ms", 1e3), ("Time [ms]", 1e3),
                                         ("epoch_us", 1e6), ("Timestamp (ns)", 1e9)])
def test_unix_time_is_recognised_in_every_unit_and_under_longer_names(name, scale):
    table = load_table({name: [int((EPOCH + i) * scale) for i in range(5)], "v": [1, 2, 3, 4, 5]})
    assert table.time_column == name and table.sensors == ["v"]
    assert table.timestamps[0] == "2026-01-01T00:00:00Z" and table.sample_interval == 1.0


def test_numeric_clocks_out_of_order_with_gaps_or_written_compactly():
    shuffled = load_table({"timestamp": [EPOCH, EPOCH + 60, EPOCH + 180, EPOCH + 120, EPOCH + 240], "v": [1, 2, 4, 3, 5]})
    assert shuffled.sensors == ["v"] and shuffled.values.ravel().tolist() == [1, 2, 3, 4, 5]
    assert any("sorted" in note for note in shuffled.notes)

    stamps = [EPOCH + 60 * i for i in range(12)]
    stamps[5] = None
    gappy = load_table({"timestamp": stamps, "v": list(range(12))})
    assert gappy.sensors == ["v"] and gappy.timestamps[5] == "2026-01-01T00:05:00Z"

    compact = load_table("timestamp,temp\n20260301105800,21.5\n20260301105900,21.6\n20260301110000,21.7\n20260301110100,21.8\n")
    assert compact.timestamps[0] == "2026-03-01T10:58:00Z" and compact.sample_interval == 60.0

    fast = load_table({"ts": [EPOCH + 0.0002 * i for i in range(5)], "v": [1, 2, 3, 4, 5]})
    assert len(set(fast.timestamps)) == 5                                # 5 kHz samples keep distinct labels


def test_columns_that_only_look_like_clocks_or_counters():
    resetting = load_table({"step": [0, 1, 2, 3, 0, 1, 2, 3], "v": [1, 2, 3, 4, 5, 6, 7, 8]})
    assert resetting.sensors == ["v"] and resetting.time_column is None             # neither a sensor nor a sort key
    assert any("does not keep increasing" in note for note in resetting.notes)
    assert load_table({"response_time_ms": [120, 130, 90, 300], "v": [1, 2, 3, 4]}).sensors == ["response_time_ms", "v"]
    assert load_table({"t": [21.5, 21.7, 21.2, 22.0], "p": [1, 2, 3, 4]}).sensors == ["t", "p"]     # a temperature
    assert load_table({"id": [7, 7, 7, 7], "v": [1, 2, 3, 4]}).time_column is None
    numbered = load_table({"id": [123456789 + i for i in range(4)], "v": [1, 2, 3, 4]})
    assert numbered.timestamps == ["123456789", "123456790", "123456791", "123456792"]             # not dates in 1973


# ---- numbers -------------------------------------------------------------------------
def test_thousands_separators_are_told_apart_from_decimal_marks():
    quoted = load_table('time,flow\n2026-03-01 10:00,"994"\n2026-03-01 10:01,"998"\n2026-03-01 10:02,"1,002"\n2026-03-01 10:03,"1,006"\n')
    assert quoted.values.ravel().tolist() == [994.0, 998.0, 1002.0, 1006.0]
    assert any("thousands separator" in note for note in quoted.notes)

    german = load_table("Zeit;Leistung\n01.03.2026 10:00;950\n01.03.2026 10:01;990\n01.03.2026 10:02;1.030\n01.03.2026 10:03;1.070\n")
    assert german.values.ravel().tolist() == [950.0, 990.0, 1030.0, 1070.0]

    millions = load_table('a,b\n"1,234,567",1\n"2,345,678",2\n"1,002",3\n')
    assert millions.values[:, 0].tolist() == [1234567.0, 2345678.0, 1002.0]


def test_decimal_marks_are_not_mistaken_for_thousands_separators():
    comma = load_table("t;p\n2026-03-01 10:00;1,002\n2026-03-01 10:01;1,006\n2026-03-01 10:02;0,998\n2026-03-01 10:03;1,013\n")
    assert np.allclose(comma.values.ravel(), [1.002, 1.006, 0.998, 1.013])
    # Three decimals everywhere, semicolon separated: still decimal points.
    fixed = load_table("t;p;q\n2026-03-01 10:00;7.050;58.266\n2026-03-01 10:01;7.061;58.310\n2026-03-01 10:02;7.049;57.900\n")
    assert np.allclose(fixed.values[:, 0], [7.050, 7.061, 7.049])
    ratios = load_table({"r": ["1.002", "1.006", "2", "1.013"], "x": [1, 2, 3, 4]})
    assert np.allclose(ratios.values[:, 0], [1.002, 1.006, 2.0, 1.013])
    percent = load_table("Zeit;Feuchte;Last\n01.03.2026 10:00;40.5%;40,5%\n01.03.2026 10:05;41.5%;41,5%\n01.03.2026 10:10;42.5%;42,5%\n")
    assert percent.sensors == ["Feuchte", "Last"] and percent.values[:, 1].tolist() == [40.5, 41.5, 42.5]
    as_numpy_text = load_table({"v": np.array(["1.5", "", "2.5", "3.5", "", "4.5"]), "w": np.arange(6.0)})
    assert as_numpy_text.sensors == ["v", "w"]


# ---- file layout ---------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["clean", "header ends with ;", "every row ends with ;", "one row ends with ;", "one row has an extra field"])
def test_semicolon_files_with_decimal_commas_survive_irregular_lines(variant):
    rows = [f"01.03.2026 10:{i:02d};21,{i % 10};1013,{i % 10}" for i in range(30)]
    header = "Zeit;Temp;Druck"
    if variant == "header ends with ;":
        header += ";"
    elif variant == "every row ends with ;":
        rows = [row + ";" for row in rows]
    elif variant == "one row ends with ;":
        rows[5] += ";"
    elif variant == "one row has an extra field":
        rows[5] += ";note"
    table = load_table(header + "\n" + "\n".join(rows))
    assert table.sensors == ["Temp", "Druck"] and table.n_rows == 30 and table.values[3, 0] == pytest.approx(21.3)


def test_rows_wider_than_the_header_single_columns_and_missing_headers():
    with_row_names = load_table("temp,druck\n1,21.5,1013.2\n2,21.6,1014.2\n3,21.7,1015.2\n")        # R's write.table layout
    assert with_row_names.sensors == ["temp", "druck"] and with_row_names.values[:, 1].tolist() == [1013.2, 1014.2, 1015.2]

    single = load_table("Temp\n21,5\n21,6\n21,7\n22,0\n")
    assert single.sensors == ["Temp"] and single.values.ravel().tolist() == [21.5, 21.6, 21.7, 22.0]

    bare = load_table("21.5,1013.2\n21.6,1014.2\n21.7,1015.2\n21.8,1016.2\n")
    assert bare.n_rows == 4 and bare.sensors == ["column_0", "column_1"]                            # the first row is data
    assert any("no header row" in note for note in bare.notes)
    bare_with_time = load_table("2026-03-01 10:00,21.5,1013.2\n2026-03-01 10:01,21.6,1014.2\n2026-03-01 10:02,21.7,1015.2\n")
    assert bare_with_time.n_rows == 3 and bare_with_time.time_column == "column_0" and len(bare_with_time.sensors) == 2

    repeated = load_table("time,temp,temp,temp_2\n2026-03-01 10:00,1,2,3\n2026-03-01 10:01,2,3,4\n2026-03-01 10:02,3,4,5\n")
    assert len(repeated.sensors) == 3 and len(set(repeated.sensors)) == 3


def test_comment_lines_are_skipped_wherever_they_are():
    body = "\n".join(f"2026-03-01 10:{i:02d},{20 + i},{1000 + i}" for i in range(12))
    leading = load_table("# Station: Pump 3, exported 2026-03-01, fw 2.1\ntime,temp,druck\n" + body)
    assert leading.sensors == ["temp", "druck"] and leading.n_rows == 12 and leading.time_column == "time"
    assert load_table("time,temp,druck\n" + body + "\n# end of log\n").n_rows == 12
    commented_header = load_table("# time,temp,druck\n" + body)
    assert commented_header.sensors == ["temp", "druck"] and commented_header.time_column == "time"


def test_encodings_compression_and_json_lines_are_recognised_by_content(tmp_path):
    text = "time,temp\n2026-03-01 10:00,21.5\n2026-03-01 10:01,21.6\n2026-03-01 10:02,21.7\n"
    (tmp_path / "utf16.csv").write_text(text, encoding="utf-16")
    assert load_table(tmp_path / "utf16.csv").sensors == ["temp"]
    (tmp_path / "utf16_no_bom.csv").write_bytes(text.encode("utf-16-le"))
    assert load_table(tmp_path / "utf16_no_bom.csv").n_rows == 3

    (tmp_path / "plain.csv.gz").write_text(text)
    plain = load_table(tmp_path / "plain.csv.gz")
    assert plain.n_rows == 3 and any("not compressed" in note for note in plain.notes)

    lines = "\n".join(json.dumps({"ts": EPOCH + 60 * i, "a": 0.5 * i}) for i in range(5))
    for name in ("lines.json", "lines.log"):
        (tmp_path / name).write_text(lines)
        assert load_table(tmp_path / name).sensors == ["a"]
    assert load_table(lines).n_rows == 5
    with pytest.raises(DataError, match="not valid JSON"):
        load_table('{"a": [1, 2,\n 3')


# ---- labels and explicit choices ---------------------------------------------------------
def test_label_columns_never_become_sensors_and_text_labels_are_understood():
    two = load_table({"v": [1.0, 3, 2, 5, 4, 6], "anomaly": [0, 0, 0, 0, 1, 1], "attack": [0, 0, 0, 1, 1, 1]})
    assert two.sensors == ["v"] and two.label_column == "anomaly"                # the second one is ground truth too
    v = [1.0, 3, 2, 5, 4, 6]
    assert load_table({"v": v, "label": ["good"] * 4 + ["bad"] * 2}).labels.tolist() == [0, 0, 0, 0, 1, 1]
    assert load_table({"v": v, "label": ["normal", "normal", "NA", "-", "n/a", "attack"]}).labels.tolist() == [0, 0, 0, 0, 0, 1]
    by_frequency = load_table({"v": v, "label": ["run", "run", "run", "run", "trip", "run"]})
    assert by_frequency.labels.tolist() == [0, 0, 0, 0, 1, 0] and any("most frequent" in note for note in by_frequency.notes)
    assert load_table({"v": v, "y_true": [1, 1, 1, 1, -1, -1]}).labels.tolist() == [0, 0, 0, 0, 1, 1]
    assert load_table({"v": v, "regime": [0.2, 0.5, 0.1, 0.9, 0.3, 0.4]}).sensors == ["v", "regime"]   # a measurement


def test_explicit_choices_are_checked_and_apply_to_loaded_tables_too():
    with pytest.raises(DataError, match="given in ignore"):
        load_table({"v": [1, 2, 3], "druck": [3, 2, 1]}, ignore=["drukc"])
    table = load_table({"temp": [1, 2, 3], "press": [3, 2, 1], "act_v": [0, 1, 0]})
    picked = load_table(table, sensors=["temp"])
    assert picked.sensors == ["temp"] and picked.controls == ["act_v"] and "press" in picked.ignored_columns
    with pytest.raises(DataError, match="already has its time axis"):
        load_table(table, label_column="press")

    # Active and reactive power per feeder are not the momentum columns of old RCWE logs.
    feeders = load_table({"timestamp": [EPOCH + i for i in range(6)], "p_1": [1, 2, 3, 4, 5, 7], "q_1": [1, 2, 3, 4, 5, 8]})
    assert feeders.sensors == ["p_1", "q_1"]


def test_a_text_column_does_not_slow_loading_down():
    import time

    rows = 60000
    data = {"ts": [EPOCH + i for i in range(rows)], "v": [float(i % 97) for i in range(rows)],
            "status": [f"RUN-{i % 7}" for i in range(rows)]}
    started = time.perf_counter()
    table = load_table(data)
    assert table.sensors == ["v"] and "status" in table.ignored_columns
    assert time.perf_counter() - started < 10.0          # it used to take seconds per thousand rows of text


def test_dataframe_indexes_and_nullable_columns_are_kept():
    pd = pytest.importorskip("pandas")
    by_text = pd.DataFrame({"time": [f"2026-03-01 10:{i:02d}" for i in range(8)], "v": np.arange(8.0)}).set_index("time")
    assert load_table(by_text).time_column == "time" and load_table(by_text).sample_interval == 60.0
    by_epoch = pd.DataFrame({"v": np.arange(8.0)}, index=pd.Index([EPOCH + 60 * i for i in range(8)], name="epoch"))
    assert load_table(by_epoch).timestamps[0] == "2026-01-01T00:00:00Z"
    nullable = pd.DataFrame({"v": np.arange(8.0), "n": pd.array([1, 2, None, 4, None, 6, 7, 8], dtype="Int64")})
    table = load_table(nullable)
    assert table.sensors == ["v", "n"] and int(table.missing.sum()) == 2 and table.values[2, 1] == 3.0


def test_delimiter_bombs_and_unicode_line_separators():
    with pytest.raises(DataError, match="columns; RCWE reads up to"):
        load_table(("," * 30000 + "\n") * 3)
    odd = load_table("time,line sep,temp\n2026-03-01 10:00,1,21.5\n2026-03-01 10:01,2,21.6\n2026-03-01 10:02,4,21.7\n")
    assert len(odd.sensors) == 2 and odd.sensors[1] == "temp"            # U+2028 inside a name is not a line break


def test_exported_headers_cannot_become_spreadsheet_formulas(tmp_path):
    import csv

    from rcwe.toolkit import TelemetryModel

    rows = [{"=cmd|' /C calc'!A0": float(i % 7), "@sum": float((i * 3) % 5), "ok": float(i % 4)} for i in range(120)]
    model = TelemetryModel(max_steps=20, device="cpu").fit(rows)
    model.detect(rows).to_csv(tmp_path / "scores.csv")
    model.forecast(3).to_csv(tmp_path / "forecast.csv")
    for name in ("scores.csv", "forecast.csv"):
        with open(tmp_path / name, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert not [cell for cell in header if cell[:1] in ("=", "@", "+", "-")], name
