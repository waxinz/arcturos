// theme.js — light/dark mode toggle (2026-09-22 UX round).
// Injects a toggle at the right edge of the nav bar, persists the choice
// in localStorage, defaults to the system preference, and re-themes
// Chart.js instances on toggle. Dark palette: Dracula (draculatheme.com)
// — background #282A36, text #F8F8F2, accents purple/cyan/green/orange.
(function () {
  'use strict';

  var STORAGE_KEY = 'arcturos-theme';

  function apply(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    // Chart.js: re-color defaults for charts created later…
    if (window.Chart) {
      var dark = theme === 'dark';
      var grid = dark ? '#44475A' : '#e5e5e5';
      var tick = dark ? '#F8F8F2' : '#1a1a1a';
      window.Chart.defaults.color = tick;
      window.Chart.defaults.borderColor = grid;
      // …and live charts already on the page.
      Object.values(window.Chart.instances || {}).forEach(function (c) {
        if (!c || !c.options || !c.options.scales) return;
        Object.values(c.options.scales).forEach(function (s) {
          if (s.title) s.title.color = tick;
          if (s.ticks) s.ticks.color = tick;
          if (s.grid) s.grid.color = grid;
        });
        c.update('none');
      });
    }
  }

  function initial() {
    var saved = null;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) {}
    if (saved === 'dark' || saved === 'light') return saved;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  }

  function inject() {
    var nav = document.querySelector('nav');
    if (!nav) return;
    var btn = document.createElement('button');
    btn.className = 'theme-toggle';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Toggle light/dark mode');
    btn.title = 'Toggle light/dark mode';
    function label() {
      btn.textContent = document.documentElement.getAttribute('data-theme') === 'dark'
        ? '☀︎' : '☾';
    }
    btn.onclick = function () {
      var next = document.documentElement.getAttribute('data-theme') === 'dark'
        ? 'light' : 'dark';
      apply(next);
      try { localStorage.setItem(STORAGE_KEY, next); } catch (e) {}
      label();
    };
    nav.appendChild(btn);
    label();
  }

  var current = initial();
  apply(current);
  // Chart.js may load after this script (vendor bundle at page bottom):
  // re-apply once the DOM is settled so chart defaults pick up the theme.
  function settle() { apply(current); inject(); }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', settle);
  } else {
    settle();
  }
})();
