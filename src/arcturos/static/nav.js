// nav.js — sidebar state painter (2026-09-30 UX round 2).
// Each page declares its identity + section on <body>:
//   <body class="shell" data-page="compare" data-section="Benchmarks">
// nav.js then marks the current section (.active on .nav-section) and
// the current link (.active on the matching data-page anchor). The
// theme toggle lives in the rail footer; theme.js still owns the
// <html data-theme> attribute + chart re-theme, so nothing changes
// there — this file only paints nav state.
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
        '.nav-section a[data-page="' + page + '"]');
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