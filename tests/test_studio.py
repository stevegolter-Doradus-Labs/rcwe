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
