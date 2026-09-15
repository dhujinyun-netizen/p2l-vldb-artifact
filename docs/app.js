(() => {
  const panel = document.querySelector('.demo-panel');
  const modeButtons = [...document.querySelectorAll('.mode-button')];
  const playButton = document.querySelector('#play-toggle');
  const title = document.querySelector('#demo-title');
  const description = document.querySelector('#demo-description');
  const counter = document.querySelector('#round-counter');
  const heroStatus = document.querySelector('#hero-status');
  const roundCopy = document.querySelector('#round-copy');
  const teaserCopy = document.querySelector('#teaser-copy');
  let mode = 'p2l';
  let playing = true;

  const copy = {
    p2l: {
      title: 'Prefix → leaf frontier',
      description: 'The prefix is fixed first. Then every valid descendant receives suffix evidence before ranking.',
      counter: 'round 1 / 4',
      hero: 'P2L active',
      round: '3 prefix decisions',
      teaser: 'all valid descendants'
    },
    sequential: {
      title: 'Level-wise beam search',
      description: 'Each new code is committed and pruned before the next position can contribute evidence.',
      counter: 'round 1 / 9',
      hero: 'Sequential active',
      round: '9 dependent decisions',
      teaser: 'one next code at a time'
    }
  };

  function render(nextMode) {
    mode = nextMode;
    panel.classList.toggle('sequential', mode === 'sequential');
    modeButtons.forEach(button => button.classList.toggle('active', button.dataset.mode === mode));
    const state = copy[mode];
    title.textContent = state.title;
    description.textContent = state.description;
    counter.textContent = state.counter;
    heroStatus.textContent = state.hero;
    roundCopy.textContent = state.round;
    teaserCopy.textContent = state.teaser;
  }

  modeButtons.forEach(button => button.addEventListener('click', () => render(button.dataset.mode)));
  playButton.addEventListener('click', () => {
    playing = !playing;
    document.body.classList.toggle('is-paused', !playing);
    playButton.textContent = playing ? 'Pause' : 'Play';
    playButton.setAttribute('aria-pressed', String(playing));
  });

  const caseView = document.querySelector('.case-view');
  const caseTabs = [...document.querySelectorAll('.case-tab')];
  const caseKicker = document.querySelector('#case-kicker');
  const caseTitle = document.querySelector('#case-title');
  const caseText = document.querySelector('#case-text');
  const seqRank = document.querySelector('#seq-rank');
  const p2lRank = document.querySelector('#p2l-rank');
  const cases = {
    recovery: {
      image: 'assets/case_recovery.png',
      kicker: 'case A · recovery',
      title: 'Complete-leaf evidence recovers the relevant item.',
      text: 'The query asks for a dog with a human instead of another dog. Sequential ranks the relevant item below the cutoff, while Independent-Suffix P2L returns it at rank 1.',
      seq: 'GT rank >10',
      p2l: 'GT rank 1'
    },
    counter: {
      image: 'assets/case_counterexample.png',
      kicker: 'case B · counterexample',
      title: 'A wrong prefix cannot be recovered later.',
      text: 'The query asks for one panda with one hand on its mouth. If the relevant prefix is removed during localization, complete-leaf scoring cannot bring it back.',
      seq: 'GT rank 1',
      p2l: 'GT rank >10'
    }
  };
  caseTabs.forEach(tab => tab.addEventListener('click', () => {
    const state = cases[tab.dataset.case];
    caseTabs.forEach(item => item.classList.toggle('active', item === tab));
    caseView.classList.toggle('counter', tab.dataset.case === 'counter');
    caseKicker.textContent = state.kicker;
    caseTitle.textContent = state.title;
    caseText.textContent = state.text;
    seqRank.textContent = state.seq;
    p2lRank.textContent = state.p2l;
    const caseImage = document.querySelector('#case-image');
    caseImage.src = state.image;
    caseImage.alt = `${state.kicker}: query, ground truth, Sequential result, and Independent-Suffix P2L result`;
  }));

  const walkSteps = [...document.querySelectorAll('.walk-step')];
  const walkProgress = document.querySelector('#walk-progress');
  const walkNext = document.querySelector('#walk-next');
  const walkTerminal = document.querySelector('#walk-terminal-text');
  let walkIndex = 0;
  const walkMessages = ['query memory ready', 'prefix y₀:y₂ retained', 'complete frontier exposed', 'paths ranked and returned'];
  function renderWalkthrough() {
    walkSteps.forEach((step, index) => {
      step.classList.toggle('active', index === walkIndex);
      step.classList.toggle('done', index < walkIndex);
    });
    walkProgress.style.width = `${((walkIndex + 1) / walkSteps.length) * 100}%`;
    walkTerminal.textContent = walkMessages[walkIndex];
    walkNext.innerHTML = walkIndex === walkSteps.length - 1 ? 'Replay <span>↻</span>' : 'Next step <span>→</span>';
  }
  walkNext.addEventListener('click', () => {
    walkIndex = (walkIndex + 1) % walkSteps.length;
    renderWalkthrough();
  });
  walkSteps.forEach((step, index) => step.addEventListener('click', () => { walkIndex = index; renderWalkthrough(); }));
  renderWalkthrough();
  render(mode);
})();
