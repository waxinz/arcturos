// app-time.js — local-timezone datetime rendering (2026-09-22 UX round).
// All datetimes render in the viewer's local timezone; when the browser
// cannot report one, Pacific time (America/Los_Angeles) is used. The raw
// UTC ISO string stays in the cell tooltip for exactness.
(function () {
  'use strict';

  function viewerTimeZone() {
    try {
      return Intl.DateTimeFormat().resolvedOptions().timeZone || null;
    } catch (e) {
      return null;
    }
  }

  window.fmtLocalTime = function (value) {
    if (value === null || value === undefined || value === '') return '—';
    var d = new Date(value);
    if (isNaN(d.getTime())) return String(value);  // not a date: pass through
    var zone = viewerTimeZone() || 'America/Los_Angeles';
    var text;
    try {
      text = new Intl.DateTimeFormat('en-US', {
        dateStyle: 'medium', timeStyle: 'short', timeZone: zone
      }).format(d);
    } catch (e) {
      text = d.toLocaleString();
    }
    return text;
  };

  window.fmtLocalTimeTitle = function (value) {
    if (value === null || value === undefined || value === '') return null;
    var d = new Date(value);
    if (isNaN(d.getTime())) return null;
    return 'UTC: ' + d.toISOString();
  };
})();
