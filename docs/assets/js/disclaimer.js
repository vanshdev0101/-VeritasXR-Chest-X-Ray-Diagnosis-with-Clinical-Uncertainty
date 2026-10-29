/* The "not a medical device" notice, as a toast.
 *
 * It slides in, waits, and leaves. Three things keep that from being a
 * downgrade on a page that prints clinical-sounding output:
 *   - a close button, because timed content needs a manual escape (WCAG 2.2.1);
 *   - hover and focus pause the timer, so it cannot vanish mid-read;
 *   - a standing note stays beside the result for the life of the page.
 */
(function () {
  const HOLD_MS = 11000;

  const el = document.createElement('div');
  el.className = 'toast';
  el.setAttribute('role', 'alert');
  el.innerHTML = `
    <svg class="icon" width="22" height="22" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" aria-hidden="true">
      <path d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>
    </svg>
    <p><strong>Not a medical device.</strong> A student research artifact trained
       on the RSNA Pneumonia dataset (test AUPRC 0.64). Not validated for clinical
       use, must not inform any diagnosis or treatment decision, and its output
       carries no clinical authority.</p>
    <button class="close" type="button" aria-label="Dismiss this notice">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>
    </button>`;
  document.body.appendChild(el);

  let timer = null;
  const hide = () => {
    el.classList.remove('in');
    el.addEventListener('transitionend', () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 800);          // reduced-motion has no transition
  };
  const arm = () => { clearTimeout(timer); timer = setTimeout(hide, HOLD_MS); };

  requestAnimationFrame(() => requestAnimationFrame(() => { el.classList.add('in'); arm(); }));
  el.querySelector('.close').addEventListener('click', hide);
  ['mouseenter', 'focusin'].forEach((e) => el.addEventListener(e, () => clearTimeout(timer)));
  ['mouseleave', 'focusout'].forEach((e) => el.addEventListener(e, arm));
})();
