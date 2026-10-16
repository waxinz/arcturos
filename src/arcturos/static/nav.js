// nav.js — top-nav state painter (2026-10-11 top-nav round).
//
// Each page declares its identity + section on <body>:
//   <body class="shell" data-page="compare" data-section="Benchmarks">
// The top bar has two rows:
//   row 1: section tabs (.nav-section[data-nav-section], .active = filled)
//   row 2: ALL sections' page pills, ALWAYS visible (constant strip,
//          2026-10-15 — the per-section swap read as "the menu changed"
//          when moving between pages). This painter tags the active
//          section's head (.active) and the current page's pill (.active).
// theme.js still owns the <html data-theme> attribute + chart re-theme;
// this file only paints nav state.
(function () {
  'use strict';

  function paint() {
    var body = document.body;
    var page = body.getAttribute('data-page');
    var section = body.getAttribute('data-section');

    if (section) {
      var sec = document.querySelector(
        '.nav-section[data-nav-section="' + section + '"]');
      if (sec) sec.classList.add('active');
    }
    if (page) {
      var link = document.querySelector(
        '.topnav-sub a[data-page="' + page + '"]');
      if (link) link.classList.add('active');
    }
  }

  function themeLabel() {
    var btn = document.querySelector('[data-theme-toggle]');
    if (!btn) return;
    btn.textContent = document.documentElement.getAttribute('data-theme') === 'dark'
      ? '☀︎' : '☾';
    btn.title = 'Toggle light/dark mode';
    btn.setAttribute('aria-label', 'Toggle light/dark mode');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      paint();
      themeLabel();
    });
  } else {
    paint();
    themeLabel();
  }
})();
