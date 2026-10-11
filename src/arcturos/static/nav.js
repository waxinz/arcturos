// nav.js — top-nav state painter (2026-10-11 top-nav round).
//
// Each page declares its identity + section on <body>:
//   <body class="shell" data-page="compare" data-section="Benchmarks">
// The top bar has two rows:
//   row 1: section tabs (.nav-section[data-nav-section], .active = filled)
//   row 2: ALL sections' page pills, hidden by CSS default (.topnav-sub a
//          { display: none }); this painter tags the active section's
//          pills .in-section so exactly one section's pages show.
// It also marks the current page's pill (.active) — same contract as the
// sidebar era. theme.js still owns the <html data-theme> attribute +
// chart re-theme; this file only paints nav state.
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

      // Row 2: reveal this section's pills only.
      var pills = document.querySelectorAll(
        '.topnav-sub a[data-nav-section="' + section + '"]');
      for (var i = 0; i < pills.length; i++) {
        pills[i].classList.add('in-section');
      }
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
