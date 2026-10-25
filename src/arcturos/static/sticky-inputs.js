/* Sticky inputs (2026-10-15): every text box on every Arcturos page
 * remembers what the operator last typed and restores it as the default
 * value on the next page load.
 *
 * Storage: browser localStorage under "arcturos-sticky:<field-id>". This
 * is the temp store the operator asked for — nothing is sent to the server.
 * Sticky values SHADOW the form's shipped defaults; the config-file
 * defaults ([eval]/[demo]/bench defaults) remain the server-side fallback
 * for any field left empty.
 *
 * Rules:
 * - Text-like inputs only (text/url/number/password/email/search/tel and
 *   typeless inputs) plus <textarea>. Password fields are included by
 *   design (operator convenience); localStorage is not encrypted, so the
 *   stored API keys live client-side in plain text.
 * - Emptying a field UN-sticks it (the key is removed, the form default
 *   applies again next load).
 * - Fields with no id are skipped (nothing stable to key on).
 * - Restore happens on DOMContentLoaded, after page scripts attached
 *   their handlers; a synthetic 'input' event is dispatched so dependent
 *   UI (kick-off button labels, etc.) reacts to the restored value.
 */
(function () {
  'use strict';
  var PREFIX = 'arcturos-sticky:';
  var SELECTOR =
    'input:not([type]), input[type="text"], input[type="url"], ' +
    'input[type="number"], input[type="password"], input[type="email"], ' +
    'input[type="search"], input[type="tel"], textarea';

  function stickyFields(root) {
    return Array.prototype.filter.call(
      (root || document).querySelectorAll(SELECTOR),
      function (el) { return !!el.id && !el.disabled; });
  }

  function storageGet(key) {
    try { return localStorage.getItem(key); } catch (_) { return null; }
  }
  function storageSet(key, value) {
    try { localStorage.setItem(key, value); } catch (_) {}
  }
  function storageRemove(key) {
    try { localStorage.removeItem(key); } catch (_) {}
  }

  function save(el) {
    if (!el.id) return;
    if (el.value === '') storageRemove(PREFIX + el.id);
    else storageSet(PREFIX + el.id, el.value);
  }

  function restoreAll() {
    var fields = stickyFields();
    for (var i = 0; i < fields.length; i++) {
      var el = fields[i];
      var saved = storageGet(PREFIX + el.id);
      // Sticky values SHADOW the form's shipped defaults (the header
      // contract) — restore even when the input ships a non-empty
      // value= (the Run page's Target URL shipped localhost, which
      // blocked the restore and read as "this box doesn't remember").
      // Emptying a field un-sticks it (save() drops the key), so the
      // shipped default still wins whenever the operator clears it.
      if (saved === null || saved === '') continue;
      el.value = saved;
      // let dependent UI react (labels, enable/disable logic, previews)
      try {
        el.dispatchEvent(new Event('input', {bubbles: false}));
        el.dispatchEvent(new Event('change', {bubbles: false}));
      } catch (_) {}
    }
  }

  document.addEventListener('input', function (ev) {
    var el = ev.target;
    if (!el || !el.matches || !el.matches(SELECTOR)) return;
    save(el);
  });
  // 'change' catches autofill/spellcheck edits that skip 'input' events
  document.addEventListener('change', function (ev) {
    var el = ev.target;
    if (!el || !el.matches || !el.matches(SELECTOR)) return;
    save(el);
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', restoreAll);
  } else {
    restoreAll();
  }
})();
