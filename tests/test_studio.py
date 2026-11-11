"""RCWE Studio: the local web server behind `rcwe serve` and `rcwe ui`."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from rcwe.toolkit.demo import make_pump_station
from rcwe.toolkit.studio import create_server

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # never route localhost through a proxy


def _call(base, path, payload=None, headers=None, raw=False):
    data = None if payload is None else (payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8"))
    request = urllib.request.Request(base + path, data=data, headers=headers or ({"Content-Type": "application/json"} if data else {}))
    try:
        with _OPENER.open(request, timeout=30) as response:
            body = response.read()
            return response.status, (body if raw else json.loads(body)), dict(response.headers)
    except urllib.error.HTTPError as err:
        body = err.read()
        return err.code, (body if raw else json.loads(body)), dict(err.headers)


@pytest.fixture(scope="module")
def studio(tmp_path_factory):
    server = create_server("127.0.0.1", 0, jobs_root=tmp_path_factory.mktemp("studio_jobs"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _wait(base, job, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, body, _ = _call(base, f"/api/jobs/{job}")
        assert status == 200
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.3)
    raise AssertionError("analysis did not finish")


def test_pages_and_health(studio):
    status, body, _ = _call(studio, "/api/health")
    assert status == 200 and body["ok"] is True
    status, page, headers = _call(studio, "/", raw=True)
    assert status == 200 and b"RCWE Studio" in page and headers["Content-Type"].startswith("text/html")
    status, page, _ = _call(studio, "/mission-control", raw=True)
    assert status == 200 and "RCWE MISSION CONTROL".encode() in page
    assert _call(studio, "/nope")[0] == 404


def test_upload_is_analysed_and_results_can_be_downloaded(studio):
    table = make_pump_station(rows=1100, faults=True, fault_start=650, seed=3).table
    rows = table.to_rows()
    header = list(rows[0].keys())
    text = ",".join(header) + "\n" + "\n".join(",".join(str(r[c]) for c in header) for r in rows)

    status, body, _ = _call(studio, "/api/analyze", {"name": "plant <b>7</b>.csv", "text": text, "train_fraction": 0.55, "forecast": 6})
    assert status == 202
    job = _wait(studio, body["job"])
    assert job["status"] == "done", job.get("error")
    assert job["events"] >= 3 and "anomaly event" in job["headline"]
    assert job["learned_from"] == "the first 605 of 1,100 rows"
    assert set(job["files"]) == {"report", "result.json", "scores.csv", "model.rcwe", "forecast.csv"}
    assert "folder" not in job                                              # server paths are not exposed

    status, report, headers = _call(studio, f"/api/jobs/{body['job']}/report", raw=True)
    assert status == 200 and report.startswith(b"<!DOCTYPE html>") and b"<b>7</b>" not in report
    status, result, headers = _call(studio, f"/api/jobs/{body['job']}/result.json")
    assert status == 200 and result["event_count"] == job["events"] and "attachment" in headers["Content-Disposition"]
    status, model, _ = _call(studio, f"/api/jobs/{body['job']}/model.rcwe", raw=True)
    assert status == 200 and len(model) > 10_000
    assert _call(studio, f"/api/jobs/{body['job']}/secrets.txt")[0] == 404
    assert _call(studio, "/api/jobs")[1]["jobs"][0]["job"] == body["job"]


def test_bad_requests_are_rejected_cleanly(studio):
    assert _call(studio, "/api/jobs/0123456789abcdef")[0] == 404
    assert _call(studio, "/api/jobs/..%2F..%2Fetc/passwd")[0] == 404
    assert _call(studio, "/api/analyze", {"text": ""})[0] == 400
    assert _call(studio, "/api/analyze", {"text": "a,b\n1,2\n3,4\n", "sensitivity": "extreme"})[0] == 400
    assert _call(studio, "/api/analyze", {"text": "a,b\n1,2\n3,4\n", "train_fraction": 5})[0] == 400
    assert _call(studio, "/api/analyze", b"not json", {"Content-Type": "application/json"})[0] == 400

    # Requests that a page on another site could forge are refused before anything runs.
    assert _call(studio, "/api/analyze", b"demo=1", {"Content-Type": "application/x-www-form-urlencoded"})[0] == 415
    assert _call(studio, "/api/analyze", {"demo": True}, {"Content-Type": "application/json", "Origin": "http://evil.example"})[0] == 403
    assert _call(studio, "/api/health", headers={"Host": "evil.example"})[0] == 403
    # Browsers label requests another site makes (an <img> or <iframe> pointed here): the API ignores those ...
    assert _call(studio, "/api/v10/tick", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert _call(studio, "/api/jobs", headers={"Sec-Fetch-Site": "same-site"})[0] == 403
    # ... while its own pages, direct visits, and following a link to the start page all work.
    assert _call(studio, "/api/health", headers={"Sec-Fetch-Site": "same-origin"})[0] == 200
    assert _call(studio, "/api/health", headers={"Sec-Fetch-Site": "none"})[0] == 200
    assert _call(studio, "/", headers={"Sec-Fetch-Site": "cross-site"}, raw=True)[0] == 200
    # A refusal closes the connection, so an unread upload cannot be mistaken for the next request.
    assert _call(studio, "/api/analyze", b"demo=1", {"Content-Type": "text/plain"})[2].get("Connection") == "close"

    # File content is never treated as a path on the server.
    status, body, _ = _call(studio, "/api/analyze", {"name": "x.csv", "text": "/etc/passwd"})
    assert status == 202
    job = _wait(studio, body["job"])
    assert job["status"] == "error" and "no data rows" in job["error"] and "root:" not in json.dumps(job)


def test_mission_control_api_is_live(studio):
    status, snap, _ = _call(studio, "/api/v10/state")
    assert status == 200 and snap["live"] is True and snap["psi_kb"] == 8.0
    status, tick, _ = _call(studio, f"/api/v10/tick?since={snap['log'][-1]['seq']}")
    assert status == 200 and tick["step"] == snap["step"] + 1 and tick["log"] == []
    status, done, _ = _call(studio, "/api/v10/action?type=trunk_fault", {})
    assert status == 200 and done["status"] == "ok"
    assert _call(studio, "/api/v10/state")[1]["regime"] == 2
    assert _call(studio, "/api/v10/action?type=launch_missiles", {})[0] == 400
    assert _call(studio, "/api/v10/action?type=paint&r=abc", {})[0] == 400


def test_pages_cannot_be_framed_by_another_site(studio):
    for path in ("/", "/mission-control", "/api/health"):
        headers = _call(studio, path, raw=True)[2]
        assert headers.get("Content-Security-Policy") == "frame-ancestors 'self'", path
        assert headers.get("X-Frame-Options") == "SAMEORIGIN", path


def test_wrongly_typed_fields_are_a_bad_request_not_a_server_error(studio):
    text = "a,b\n1,2\n3,4\n"
    for field, value in (("sensitivity", None), ("sensitivity", [1]), ("sensitivity", {}), ("sensitivity", True),
                         ("forecast", 1e999), ("train_fraction", "half")):
        assert _call(studio, "/api/analyze", {"text": text, field: value})[0] == 400, (field, value)
    deep = b"[" * 100000
    assert _call(studio, "/api/analyze", deep, {"Content-Type": "application/json"})[0] == 400


def test_host_header_rules():
    from rcwe.toolkit.studio import _host_allowed

    # Bound to this machine only: loopback names, with the right port (or none at all on port 80).
    assert _host_allowed("127.0.0.1:8765", "127.0.0.1", 8765) and _host_allowed("localhost:8765", "127.0.0.1", 8765)
    assert _host_allowed("[::1]:8765", "127.0.0.1", 8765) and _host_allowed("127.0.0.1", "127.0.0.1", 80)
    assert not _host_allowed("evil.example:8765", "127.0.0.1", 8765) and not _host_allowed("127.0.0.1:9999", "127.0.0.1", 8765)
    assert not _host_allowed("192.168.1.20:8765", "127.0.0.1", 8765) and not _host_allowed("", "127.0.0.1", 8765)
    # Bound to the network: any address literal is fine, a name that could be re-pointed at us is not.
    assert _host_allowed("192.168.1.20:8765", "0.0.0.0", 8765) and _host_allowed("localhost:8765", "0.0.0.0", 8765)
    assert not _host_allowed("evil.example:8765", "0.0.0.0", 8765)


def test_finished_analyses_make_room_and_running_ones_are_never_dropped(tmp_path):
    from rcwe.toolkit.studio import JobRunner

    (tmp_path / "0123456789abcdef").mkdir()                      # left behind by an earlier run
    (tmp_path / "fedcba9876543210").mkdir()
    (tmp_path / "my notes").mkdir()                              # not ours: never touched
    runner = JobRunner(tmp_path, keep=1)
    assert sorted(p.name for p in tmp_path.iterdir()) in (["0123456789abcdef", "my notes"], ["fedcba9876543210", "my notes"])

    rows = "\n".join(f"{i},{(i * 7) % 11},{(i * 3) % 5}" for i in range(60))
    ids = [runner.submit({"name": f"log{k}.csv", "text": "t,a,b\n" + rows}) for k in range(3)]
    assert all(runner.status(job) is not None for job in ids)   # queued and running jobs all stay, whatever `keep` says
    deadline = time.time() + 240
    while time.time() < deadline and any(runner.status(job)["status"] not in ("done", "error") for job in ids):
        time.sleep(0.2)
    assert [runner.status(job)["status"] for job in ids] == ["done", "done", "done"]
    assert not list((tmp_path / ids[-1]).glob("upload.*"))      # the results are kept, not a copy of the upload
    last = runner.submit({"name": "log3.csv", "text": "t,a,b\n" + rows})
    assert all(runner.status(job) is None for job in ids)       # finished ones made room ...
    assert not any((tmp_path / job).exists() for job in ids)    # ... and took their folders with them
    assert runner.status(last) is not None and (tmp_path / "my notes").exists()


def test_a_server_that_cannot_start_says_why_and_leaves_nothing_behind(tmp_path):
    import threading as _threading

    before = _threading.active_count()
    taken = create_server("127.0.0.1", 0, jobs_root=tmp_path / "jobs")
    try:
        with pytest.raises(OSError):
            create_server("127.0.0.1", taken.server_address[1], jobs_root=tmp_path / "jobs")
    finally:
        taken.server_close()
    assert _threading.active_count() == before and not (tmp_path / "jobs").exists()
