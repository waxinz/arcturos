"""J2 comparison tests — /api/compare/benchmarks + /api/compare/run-diff.

Covers: happy path (2 runs × metrics), missing run -> 404, bad metric -> 422,
mtp_acceptance computation + null when draft fields absent, run-diff deltas
for known fixture values, engine metadata diff, and latest-point-per-context
resolution under the append-only model.
"""

import pytest

from arcturos import ops

# ---------------------------------------------------------- fixtures --------

RUN_A = {
    "server_url": "http://fixt-host-a:8000",
    "model_fingerprint": (
        "~/.cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/"
        "snapshots/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-IQ3_XXS/"
        "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
    ),
    "engine": "llama.cpp",
    "context_size": 262144,
}

RUN_B = {
    "server_url": "http://fixt-host-b:8000",
    "model_fingerprint": "GLM-5.3-Flash (GGUF, demo GPU node)",
    "engine": "llama.cpp",
    "context_size": 262144,
}

# Two context points per run with fully populated metrics; mtp 96/83 = 0.864...
BENCH_A = [
    {
        "context_tokens": 4096,
        "prefill_tps": 476.2, "decode_tps": 56.8, "ttft_ms": 214.0,
        "wall_s": 7.4, "output_tokens": 32, "mtp_draft_n": 96,
        "mtp_accepted": 83, "power_watts": 249.5, "power_host": "h@10.0.0.1", "power_gpu_index": 0,
    },
    {
        "context_tokens": 65536,
        "prefill_tps": 458.1, "decode_tps": 61.2, "ttft_ms": 1429.0,
        "wall_s": 8.1, "output_tokens": 32, "mtp_draft_n": 96,
        "mtp_accepted": 83, "power_watts": 250.1, "power_host": "h@10.0.0.1", "power_gpu_index": 0,
    },
]

BENCH_B = [
    {
        "context_tokens": 4096,
        "prefill_tps": 512.7, "decode_tps": 74.3, "ttft_ms": 189.0,
        "wall_s": 6.2, "output_tokens": 32, "mtp_draft_n": 96,
        "mtp_accepted": 86, "power_watts": 248.2, "power_host": "h@10.0.0.2", "power_gpu_index": 0,
    },
    {
        "context_tokens": 65536,
        "prefill_tps": 493.5, "decode_tps": 68.9, "ttft_ms": 1327.0,
        "wall_s": 7.0, "output_tokens": 32, "mtp_draft_n": 96,
        "mtp_accepted": 84, "power_watts": 249.9, "power_host": "h@10.0.0.2", "power_gpu_index": 0,
    },
]


def _create_run(client, run: dict) -> int:
    res = client.post("/api/runs", json=run)
    assert res.status_code == 201, res.text
    return res.json()["id"]


def _add_benchmarks(client, run_id: int, points: list[dict]) -> None:
    for point in points:
        res = client.post(f"/api/runs/{run_id}/benchmarks", json=point)
        assert res.status_code == 201, res.text


def _seed_two_runs(client) -> tuple[int, int]:
    a = _create_run(client, RUN_A)
    b = _create_run(client, RUN_B)
    _add_benchmarks(client, a, BENCH_A)
    _add_benchmarks(client, b, BENCH_B)
    return a, b


# --------------------------------------------- compare benchmarks ----------


def test_compare_benchmarks_happy_path(client):
    a, b = _seed_two_runs(client)
    res = client.get(
        f"/api/compare/benchmarks?runs={a},{b}&metrics=decode_tps,prefill_tps,mtp_acceptance"
    )
    assert res.status_code == 200, res.text
    data = res.json()

    assert [r["id"] for r in data["runs"]] == [a, b]
    assert data["runs"][0]["engine"] == "llama.cpp"
    # Path fingerprint -> short legend name (basename minus shard suffix).
    assert data["runs"][0]["model_fingerprint_short"] == "DeepSeek-V4-Flash-0731-UD-IQ3_XXS"
    assert data["runs"][1]["model_fingerprint_short"] == "GLM-5.3-Flash (GGUF, demo GPU node)"

    assert data["metrics"] == ["decode_tps", "prefill_tps", "mtp_acceptance"]

    dec_a = data["series"]["decode_tps"][str(a)]
    assert [p["context_tokens"] for p in dec_a] == [4096, 65536]  # sorted
    assert dec_a[0]["value"] == 56.8
    assert dec_a[1]["value"] == 61.2

    mtp_b = data["series"]["mtp_acceptance"][str(b)]
    assert mtp_b[0]["value"] == pytest.approx(86 / 96)
    assert mtp_b[1]["value"] == pytest.approx(84 / 96)


def test_compare_missing_run_404(client):
    a, _ = _seed_two_runs(client)
    res = client.get(f"/api/compare/benchmarks?runs={a},9999&metrics=decode_tps")
    assert res.status_code == 404
    assert "9999" in res.json()["detail"]


def test_compare_bad_metric_422(client):
    a, _ = _seed_two_runs(client)
    res = client.get(f"/api/compare/benchmarks?runs={a}&metrics=decode_tps,bogus_metric")
    assert res.status_code == 422
    assert "bogus_metric" in res.json()["detail"]


def test_compare_empty_runs_422(client):
    res = client.get("/api/compare/benchmarks?runs=&metrics=decode_tps")
    assert res.status_code == 422


def test_compare_bad_run_id_422(client):
    res = client.get("/api/compare/benchmarks?runs=abc&metrics=decode_tps")
    assert res.status_code == 422


def test_compare_metrics_default_to_all_when_omitted(client):
    a, _ = _seed_two_runs(client)
    res = client.get(f"/api/compare/benchmarks?runs={a}")
    assert res.status_code == 200, res.text
    assert set(res.json()["metrics"]) == {
        "decode_tps", "prefill_tps", "ttft_ms", "wall_s", "output_tokens",
        "mtp_acceptance", "power_watts", "streams",
        "decode_tps_combined", "prefill_tps_combined",
    }


def test_compare_combined_throughput_series(client):
    """Combined (Σ across streams) metrics are selectable and plot their
    stored values (2026-09-22: exposed as compare-page checkboxes)."""
    a, b = _seed_two_runs(client)
    # The shared fixtures leave combined columns NULL (single-stream class);
    # seed explicit multi-stream points so the series has real values.
    _add_benchmarks(client, a, [
        {"context_tokens": 4096, "decode_tps": 56.8, "streams": 4,
         "decode_tps_combined": 227.2, "prefill_tps_combined": 1904.8},
    ])
    res = client.get(
        f"/api/compare/benchmarks?runs={a},{b}"
        "&metrics=decode_tps_combined,prefill_tps_combined"
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["metrics"] == ["decode_tps_combined", "prefill_tps_combined"]
    dec = data["series"]["decode_tps_combined"][str(a)]
    assert [p["context_tokens"] for p in dec] == [4096, 65536]
    assert dec[0]["value"] == 227.2
    assert dec[1]["value"] is None  # 65536 point has no combined value
    pre = data["series"]["prefill_tps_combined"][str(a)]
    assert pre[0]["value"] == 1904.8


def test_compare_combined_honest_null_when_not_multi_stream(client):
    """A single-stream point stores NULL for the combined columns — the
    series keeps the point with value null (never dropped, never zeroed)."""
    a = _create_run(client, RUN_A)
    _add_benchmarks(client, a, BENCH_A)  # no streams field -> defaults to 1
    res = client.get(
        f"/api/compare/benchmarks?runs={a}&metrics=decode_tps_combined"
    )
    assert res.status_code == 200, res.text
    points = res.json()["series"]["decode_tps_combined"][str(a)]
    assert points == [
        {"context_tokens": 4096, "value": None},
        {"context_tokens": 65536, "value": None},
    ]


def test_mtp_acceptance_null_when_draft_fields_absent(client):
    run_id = _create_run(client, RUN_A)
    # Point with no MTP counters at all -> acceptance must be null, point kept.
    _add_benchmarks(client, run_id, [{"context_tokens": 4096, "decode_tps": 42.0}])
    res = client.get(
        f"/api/compare/benchmarks?runs={run_id}&metrics=mtp_acceptance,decode_tps"
    )
    assert res.status_code == 200, res.text
    series = res.json()["series"]
    assert series["decode_tps"][str(run_id)] == [
        {"context_tokens": 4096, "value": 42.0}
    ]
    # Point present with an honest null — never dropped, never zeroed.
    assert series["mtp_acceptance"][str(run_id)] == [
        {"context_tokens": 4096, "value": None}
    ]


def test_compare_series_uses_latest_point_per_context(client):
    run_id = _create_run(client, RUN_A)
    _add_benchmarks(client, run_id, [{"context_tokens": 4096, "decode_tps": 50.0}])
    # Append-only re-run at the same context supersedes (PRD §J1): the
    # comparison must resolve to the latest measurement, not the first.
    _add_benchmarks(client, run_id, [{"context_tokens": 4096, "decode_tps": 60.0}])
    res = client.get(f"/api/compare/benchmarks?runs={run_id}&metrics=decode_tps")
    assert res.status_code == 200, res.text
    points = res.json()["series"]["decode_tps"][str(run_id)]
    assert points == [{"context_tokens": 4096, "value": 60.0}]


# ------------------------------------------------------------- run diff ----


def test_run_diff_deltas_and_metadata(client):
    a, b = _seed_two_runs(client)
    res = client.get(f"/api/compare/run-diff/{a}/{b}")
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["run_a"]["id"] == a
    assert data["run_b"]["id"] == b
    assert data["run_a"]["model_fingerprint_short"] == "DeepSeek-V4-Flash-0731-UD-IQ3_XXS"

    # Engine metadata diff: same n_ctx, different model path.
    assert data["engine_metadata"]["n_ctx"] == {"a": 262144, "b": 262144, "same": True}
    assert data["engine_metadata"]["model_path"]["same"] is False
    assert data["engine_metadata"]["model_path"]["a"] == RUN_A["model_fingerprint"]
    assert data["engine_metadata"]["model_path"]["b"] == RUN_B["model_fingerprint"]

    # Known fixture deltas at ctx 4096: decode 74.3 vs 56.8 -> Δ=17.5, pct≈30.8%
    ctx4096 = data["deltas"][0]
    assert ctx4096["context_tokens"] == 4096
    dec = ctx4096["metrics"]["decode_tps"]
    assert dec["a"] == 56.8 and dec["b"] == 74.3
    assert dec["delta"] == pytest.approx(74.3 - 56.8)
    assert dec["pct"] == pytest.approx((74.3 - 56.8) / 56.8 * 100)

    # TTFT is lower-is-better: 189 vs 214 -> negative delta.
    ttft = ctx4096["metrics"]["ttft_ms"]
    assert ttft["delta"] == pytest.approx(189.0 - 214.0)
    assert ttft["pct"] == pytest.approx((189.0 - 214.0) / 214.0 * 100)

    # mtp_acceptance is computed per side before the delta.
    mtp = ctx4096["metrics"]["mtp_acceptance"]
    va, vb = 83 / 96, 86 / 96
    assert mtp["a"] == pytest.approx(va)
    assert mtp["b"] == pytest.approx(vb)
    assert mtp["delta"] == pytest.approx(vb - va)

    # Both context points present, sorted.
    assert [d["context_tokens"] for d in data["deltas"]] == [4096, 65536]

    # Direction map is exposed so the UI can color deltas correctly.
    assert data["directions"]["decode_tps"] == "higher"
    assert data["directions"]["ttft_ms"] == "lower"
    assert data["directions"]["mtp_acceptance"] == "higher"


def test_run_diff_missing_points_become_null(client):
    a = _create_run(client, RUN_A)
    b = _create_run(client, RUN_B)
    # Run B only has the 4096 point; run A has both.
    _add_benchmarks(client, a, BENCH_A)
    _add_benchmarks(client, b, [BENCH_B[0]])
    res = client.get(f"/api/compare/run-diff/{a}/{b}")
    assert res.status_code == 200, res.text
    data = res.json()

    rows = {d["context_tokens"]: d for d in data["deltas"]}
    assert set(rows) == {4096, 65536}  # union, not intersection
    missing = rows[65536]["metrics"]["decode_tps"]
    assert missing["a"] == 61.2 and missing["b"] is None
    assert missing["delta"] is None and missing["pct"] is None


def test_run_diff_404_when_run_missing(client):
    a, _ = _seed_two_runs(client)
    assert client.get(f"/api/compare/run-diff/{a}/9999").status_code == 404
    assert client.get("/api/compare/run-diff/9999/1").status_code == 404


def test_run_diff_combined_metrics_direction_and_deltas(client):
    """Combined (Σ) throughput appears in run diffs as higher-is-better,
    and stays null when a side has no multi-stream point."""
    a, b = _seed_two_runs(client)
    _add_benchmarks(client, a, [
        {"context_tokens": 4096, "decode_tps": 56.8, "streams": 4,
         "decode_tps_combined": 227.2},
    ])
    _add_benchmarks(client, b, [
        {"context_tokens": 4096, "decode_tps": 74.3, "streams": 2,
         "decode_tps_combined": 148.6},
    ])
    res = client.get(f"/api/compare/run-diff/{a}/{b}")
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["directions"]["decode_tps_combined"] == "higher"
    assert data["directions"]["prefill_tps_combined"] == "higher"
    ctx4096 = data["deltas"][0]
    dec = ctx4096["metrics"]["decode_tps_combined"]
    assert dec["a"] == 227.2 and dec["b"] == 148.6
    assert dec["delta"] == pytest.approx(148.6 - 227.2)
    assert dec["pct"] == pytest.approx((148.6 - 227.2) / 227.2 * 100)


def test_compare_page_lists_combined_metric_checkboxes(client):
    """Compare Runs page exposes the combined metrics as selectable
    checkboxes so they can be plotted like any other metric (static
    source: the checkbox values live in the METRICS array literal,
    labels in METRIC_LABELS)."""
    html = client.get("/compare").text
    assert "'decode_tps_combined'" in html
    assert "'prefill_tps_combined'" in html
    assert "Σ decode tok/s (combined)" in html
    assert "Σ prefill tok/s (combined)" in html


def test_compare_page_merged_defaults_and_order(client):
    """Compare page (owner renames 2026-10-11: ex-"Runs", ex-"Compare
    Runs"): ttft / wall / prefill / decode checkboxes are pre-checked;
    the runs list renders latest-first (render-time reversal of the
    ascending API); the former Runs-page controls (CSV export + Hide
    selected) live on the page; a client-side pager with an owner-settable
    page size (30/50/100/All, persisted in localStorage) keeps Compare
    reachable; the integrated A/B diff renders under the charts
    (skip-blank)."""
    html = client.get("/compare").text
    assert "DEFAULT_METRICS = new Set(['decode_tps', 'prefill_tps', 'ttft_ms', 'wall_s'])" in html
    assert "b.id - a.id" in html            # latest up top
    assert 'id="export-runs"' in html       # runs CSV export carried over
    assert 'id="hide-selected"' in html     # bulk hide carried over
    assert "<h2>Compare</h2>" in html
    # Client-side pagination with a persisted page-size dropdown.
    assert "PAGE_SIZE_KEY = 'arcturos-run-page-size'" in html
    assert "[30, 50, 100].includes(+v)" in html
    assert "o.value = opt[0]; o.textContent = opt[1];" in html
    assert "['30', '30 / page'], ['50', '50 / page']" in html
    assert "['100', '100 / page'], ['all', 'All']" in html
    assert "localStorage.setItem(PAGE_SIZE_KEY, v)" in html
    # Compare sits in the runs toolbar (owner call: next to Hide selected,
    # not a floating bar) with a live selected-count hint beside it.
    assert 'id="compare-btn"' in html
    assert 'id="abar-count"' in html
    assert 'id="action-bar"' not in html
    assert 'id="charts"' in html            # results section above controls
    assert "renderAbDiff" in html           # integrated Run Diff
    assert "run-diff/" in html              # …fed by the run-diff payload
    # Owner skip-blank rule: a metric missing on either side shows no row.
    assert "d.a === null || d.a === undefined ||" in html


def test_runs_root_redirect_to_compare(client):
    """/ and /runs redirect (308) to /compare after the merge; unknown
    pages still 404 rather than silently redirecting."""
    for path in ("/", "/runs"):
        res = client.get(path, follow_redirects=False)
        assert res.status_code == 308, path
        assert res.headers["location"] == "/compare", path
    assert client.get("/nope").status_code == 404


# ------------------------------------------------- display precision -------


def test_ui_floats_render_at_2dp(client):
    """All numeric rendering paths round floats to 2dp (2026-09-22: run 30
    showed 30.105000000000004). Presentation-only — storage/CSV keep full
    fidelity; integers and strings pass through untouched."""
    # run_detail: cellText rounds floats, keeps '—' for nulls
    detail = client.get("/runs/1").text
    assert "v.toFixed(2)" in detail
    assert "Number.isInteger(v)" in detail
    # compare: fmtNum + chart tooltip callbacks
    compare = client.get("/compare").text
    assert compare.count("toFixed(2)") >= 2  # fmtNum + tooltip label
    assert "ctx.parsed.y" in compare
    # create: live-table cell() rounds floats too
    create = client.get("/create").text
    assert "v.toFixed(2)" in create
    # the integrated A/B diff rounds via fmtNum on the compare page
    # (the standalone /diff page is retired; fmt() lived there)
    assert "toFixed(2)" in client.get("/compare").text


# ------------------------------------------------------------ views ---------


def test_compare_views_served(client):
    """/compare is the merged Runs+Compare page; /diff is retired as a
    page (2026-10-11): it 308-redirects to /compare, where the A/B
    tables render under the charts."""
    assert client.get("/compare").status_code == 200
    diff = client.get("/diff", follow_redirects=False)
    assert diff.status_code == 308
    assert diff.headers["location"] == "/compare"
    assert client.get("/runs").status_code == 200
    assert "Arcturos" in client.get("/compare").text
    assert "Chart" in client.get("/compare").text


def test_vendored_chartjs_served(client):
    res = client.get("/static/vendor/chart.umd.min.js")
    assert res.status_code == 200
    assert len(res.content) > 100_000  # real library, not a stub
    assert "Chart" in res.text[:2000] or b"Chart" in res.content[:2000]


def test_compare_payload_carries_run_names(client):
    """Compare + diff payloads expose each run's name (2026-09-22 naming
    feature) so legends and headers can prefer it over the fingerprint."""
    import sqlite3
    a, b = _seed_two_runs(client)
    # seed names directly (label column is trigger-whitelisted)
    for run_id, name in [(a, "alpha sweep"), (b, "beta sweep")]:
        conn = sqlite3.connect(client.app.state.db_path)
        conn.execute("UPDATE runs SET name = ? WHERE id = ?", (name, run_id))
        conn.commit()
        conn.close()
    payload = client.get(f"/api/compare/benchmarks?runs={a},{b}").json()
    names = {r["id"]: r["name"] for r in payload["runs"]}
    assert names.get(a) == "alpha sweep", (a, b, names)
    assert names.get(b) == "beta sweep", (a, b, names)
    diff = client.get(f"/api/compare/run-diff/{a}/{b}").json()
    assert diff["run_a"]["name"] == "alpha sweep"
    assert diff["run_b"]["name"] == "beta sweep"
