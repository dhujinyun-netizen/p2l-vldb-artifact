/* Schematic execution walkthrough. No model calls or experiment data are generated. */
(() => {
  'use strict';
  const demo = document.querySelector('.execution-demo');
  if (!demo) return;
  const descriptions = [
    'Search the Candidate Trie to depth s = 3, including the modality-routing code. Retain the top-B prefixes; leaves outside these prefixes are no longer reachable.',
    'Enumerate all stored descendants of the retained prefixes. This index operation exposes valid complete IDs without adding a neural decoding round.',
    'Predict unresolved-position scores conditioned on the query and each prefix. Suffix positions do not attend to each other. Descendants gather their code scores from the shared matrix.',
    'Add prefix and gathered suffix scores, then select top-K complete IDs across the retained frontier. Exact-ID expansion and frozen embedding reranking follow.'
  ];
  const steps = [...demo.querySelectorAll('[data-step]')];
  const play = document.getElementById('demo-play');
  const next = document.getElementById('demo-next');
  let stage = 0;
  let timer = null;
  const stop = () => {
    clearInterval(timer);
    timer = null;
    play.textContent = stage === 3 ? 'Replay animation' : 'Play animation';
    play.setAttribute('aria-pressed', 'false');
  };
  const render = () => {
    demo.dataset.stage = String(stage);
    steps.forEach((button, i) => {
      if (i === stage) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    document.getElementById('demo-count').textContent = `0${stage + 1} / 04`;
    document.getElementById('demo-description').textContent = descriptions[stage];
    next.disabled = stage === 3;
  };
  play.addEventListener('click', () => {
    if (timer !== null) { stop(); return; }
    if (stage === 3) stage = 0;
    render();
    play.textContent = 'Pause';
    play.setAttribute('aria-pressed', 'true');
    timer = setInterval(() => {
      stage += 1;
      render();
      if (stage === 3) stop();
    }, 2600);
  });
  document.getElementById('demo-reset').addEventListener('click', () => { stage = 0; stop(); render(); });
  next.addEventListener('click', () => { stage = Math.min(stage + 1, 3); stop(); render(); });
  steps.forEach((button, i) => {
    button.disabled = false;
    button.addEventListener('click', () => { stage = i; stop(); render(); });
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  window.addEventListener('pagehide', stop);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => { if (!entries[0].isIntersecting) stop(); }).observe(demo);
  }
  demo.querySelector('.demo-controls').hidden = false;
  demo.classList.add('enhanced');
  render();
})();
