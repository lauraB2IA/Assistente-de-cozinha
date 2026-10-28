// ── Scroll Reveal ─────────────────────────────────────────────────────────────
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      revealObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

// ── Navbar scroll effect ──────────────────────────────────────────────────────
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
});

// ── Animated counters ─────────────────────────────────────────────────────────
function animateCounter(el, target, suffix = '') {
  const duration = 1600;
  const start = performance.now();
  const update = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(eased * target) + suffix;
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const el = e.target;
      const target = parseInt(el.dataset.target);
      const suffix = el.dataset.suffix || '';
      animateCounter(el, target, suffix);
      counterObserver.unobserve(el);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.stat-num').forEach(el => counterObserver.observe(el));

// ── Waveform canvas animation ─────────────────────────────────────────────────
const canvas = document.getElementById('waveCanvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

let time = 0;
function drawWave() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const waves = [
    { amp: 40, freq: 0.008, speed: 0.018, color: 'rgba(139,92,246,0.25)', y: 0.45 },
    { amp: 28, freq: 0.012, speed: 0.025, color: 'rgba(34,211,238,0.18)',  y: 0.55 },
    { amp: 18, freq: 0.018, speed: 0.032, color: 'rgba(236,72,153,0.12)',  y: 0.50 },
  ];
  waves.forEach(w => {
    ctx.beginPath();
    ctx.strokeStyle = w.color;
    ctx.lineWidth = 1.5;
    for (let x = 0; x <= canvas.width; x += 2) {
      const y = canvas.height * w.y + Math.sin(x * w.freq + time * w.speed * 60) * w.amp;
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  time += 1;
  requestAnimationFrame(drawWave);
}
drawWave();

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-content-' + tab).classList.add('active');
  });
});

// ── Accuracy circle ───────────────────────────────────────────────────────────
function updateAccuracy() {
  const val = parseFloat(document.getElementById('accInput').value);
  if (isNaN(val) || val < 0 || val > 100) return;
  const display = document.getElementById('accDisplay');
  const circle = document.getElementById('accCircle');
  display.textContent = val.toFixed(1) + '%';
  circle.style.setProperty('--pct', val);

  // Inject SVG gradient (needed for circle stroke)
  let defs = document.querySelector('#acc-svg-defs');
  if (!defs) {
    const svg = document.querySelector('.acc-svg');
    defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.id = 'acc-svg-defs';
    const grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    grad.id = 'grad';
    grad.setAttribute('x1', '0%'); grad.setAttribute('y1', '0%');
    grad.setAttribute('x2', '100%'); grad.setAttribute('y2', '0%');
    const s1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    s1.setAttribute('offset', '0%'); s1.setAttribute('stop-color', '#8b5cf6');
    const s2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    s2.setAttribute('offset', '100%'); s2.setAttribute('stop-color', '#22d3ee');
    grad.appendChild(s1); grad.appendChild(s2);
    defs.appendChild(grad);
    svg.insertBefore(defs, svg.firstChild);
    circle.setAttribute('stroke', 'url(#grad)');
  }
}

// ── Image fallback: show placeholder if image fails to load ──────────────────
function handleImageLoad(imgId, phId) {
  const img = document.getElementById(imgId);
  const ph = document.getElementById(phId);
  if (!img) return;
  const show = () => { img.style.display = 'none'; ph.style.display = 'flex'; };
  img.addEventListener('error', show);
  // Also check if already broken
  if (img.complete && img.naturalWidth === 0) show();
}
handleImageLoad('img-matriz', 'ph-matriz');
handleImageLoad('img-curvas', 'ph-curvas');

// ── Active nav link on scroll ─────────────────────────────────────────────────
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-links a');

const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(a => a.style.color = '');
      const active = document.querySelector(`.nav-links a[href="#${e.target.id}"]`);
      if (active) active.style.color = '#8b5cf6';
    }
  });
}, { rootMargin: '-40% 0px -55% 0px' });

sections.forEach(s => sectionObserver.observe(s));

// ── Demo ao Vivo ──────────────────────────────────────────────────────────────
function loadDemo() {
  const start = document.getElementById('demoStart');
  const frame = document.getElementById('demoFrame');
  const iframe = document.getElementById('demoIframe');

  iframe.src = 'http://localhost:7860';
  start.style.display = 'none';
  frame.style.display = 'flex';
  frame.style.flexDirection = 'column';
}

function closeDemo() {
  const start = document.getElementById('demoStart');
  const frame = document.getElementById('demoFrame');
  const iframe = document.getElementById('demoIframe');

  iframe.src = '';
  frame.style.display = 'none';
  start.style.display = 'flex';
}

function copyCmd() {
  const text = document.getElementById('cmdText').textContent;
  const btn = document.getElementById('copyBtn');
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✓ Copiado!';
    btn.style.color = '#10b981';
    btn.style.borderColor = '#10b981';
    setTimeout(() => {
      btn.textContent = 'Copiar';
      btn.style.color = '';
      btn.style.borderColor = '';
    }, 2000);
  });
}
