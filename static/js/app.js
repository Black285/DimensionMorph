/**
 * DimensionMorph front end.
 *
 * The form is built from /api/catalog rather than being hard-coded, so adding a
 * source or a coordinate system on the server is enough to make it appear here.
 *
 * Drawing rules that keep this responsive:
 *   - exactly one panel is on screen, and only that panel is requested and
 *     rendered; the others are marked stale and redrawn when selected;
 *   - each panel is owned by at most one in-flight request, and a newer request
 *     aborts whatever it supersedes, so a dragged slider cannot have a stale
 *     response land on top of a fresh one;
 *   - updates go through Plotly.react, which diffs against the current figure
 *     instead of tearing the plot down and rebuilding it.
 */

const PLOT_CONFIG = {
  responsive: false,
  displaylogo: false,
  scrollZoom: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toggleSpikelines', 'hoverClosestCartesian',
                           'hoverCompareCartesian'],
  toImageButtonOptions: { format: 'png', scale: 2, filename: 'dimensionmorph' },
};

const DEBOUNCE_FAST = 160;
const DEBOUNCE_HEAVY = 280;
const MAX_PANELS = 3;

const dom = {
  body: document.body,
  root: document.documentElement,
  source: document.getElementById('source'),
  sourceSummary: document.getElementById('source-summary'),
  expressionFields: document.getElementById('expression-fields'),
  rangeFields: document.getElementById('range-fields'),
  samples: document.getElementById('samples'),
  samplesLabel: document.getElementById('samples-label'),
  samplesValue: document.getElementById('samples-value'),
  params: document.getElementById('params'),
  addParameter: document.getElementById('add-parameter'),
  examples: document.getElementById('examples'),
  referenceBody: document.getElementById('reference-body'),
  panels: document.getElementById('panels'),
  panelTemplate: document.getElementById('panel-template'),
  caption: document.getElementById('caption'),
  status: document.getElementById('status'),
  error: document.getElementById('error'),
  panelPicker: document.getElementById('panel-picker'),
  themeToggle: document.getElementById('theme-toggle'),
  sidebarToggle: document.getElementById('sidebar-toggle'),
};

const state = {
  source: 'explicit2d',
  expressions: {},
  ranges: {},
  samples: null,
  parameters: [],
  views: [],
  panel: 0,          // which panel is on screen
  theme: 'light',
};

let catalog = null;
let debounceTimer = null;
let urlTimer = null;
const panels = [];            // { root, select, equations, note, plot, rendered }
const inflight = new Map();   // panel index -> the AbortController that owns it

/* ------------------------------------------------------------------ utils */

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

function currentSource() {
  return catalog.sources.find((s) => s.key === state.source) || catalog.sources[0];
}

function visibleIndices() {
  return [state.panel];
}

/** Mark every panel as needing a redraw; the hidden ones wait until selected. */
function invalidateAll() {
  for (const panel of panels) panel.dirty = true;
}

function whenPlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  return new Promise((resolve) => {
    const poll = () => (window.Plotly ? resolve(window.Plotly) : requestAnimationFrame(poll));
    poll();
  });
}

function setStatus(text, busy = false) {
  dom.status.textContent = text;
  dom.status.parentElement.classList.toggle('busy', busy);
}

function showError(message) {
  dom.error.textContent = message;
  dom.error.hidden = false;
}

function clearError() {
  dom.error.hidden = true;
  dom.error.textContent = '';
}

/* ------------------------------------------------------------------ theme */

function applyTheme(name) {
  state.theme = name;
  dom.root.dataset.theme = name;
  try { localStorage.setItem('dm-theme', name); } catch { /* private mode */ }
}

function initialTheme() {
  try {
    const saved = localStorage.getItem('dm-theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* private mode */ }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/* ------------------------------------------------------------- URL state */

function encodeState() {
  const compact = {
    s: state.source,
    e: state.expressions,
    r: state.ranges,
    n: state.samples,
    p: state.parameters.map((p) => [p.name, p.value, p.min, p.max]),
    v: state.views,
    a: state.panel,
  };
  return btoa(unescape(encodeURIComponent(JSON.stringify(compact))))
    .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function decodeState(hash) {
  const padded = hash.replace(/-/g, '+').replace(/_/g, '/');
  return JSON.parse(decodeURIComponent(escape(atob(padded))));
}

function syncUrl() {
  clearTimeout(urlTimer);
  urlTimer = setTimeout(() => {
    history.replaceState(null, '', `#${encodeState()}`);
  }, 600);
}

function restoreFromUrl() {
  const hash = location.hash.slice(1);
  if (!hash) return false;
  try {
    const saved = decodeState(hash);
    const source = catalog.sources.find((s) => s.key === saved.s);
    if (!source) return false;
    state.source = saved.s;
    state.expressions = saved.e || {};
    state.ranges = saved.r || {};
    state.samples = saved.n || source.samples.default;
    state.views = Array.isArray(saved.v) && saved.v.length ? saved.v : source.defaultViews;
    state.panel = clamp(Number(saved.a) || 0, 0, MAX_PANELS - 1);
    state.parameters = (saved.p || []).map(([name, value, min, max]) => ({
      name, value: Number(value), min: Number(min), max: Number(max),
    }));
    return true;
  } catch {
    return false;
  }
}

/* -------------------------------------------------------------- the form */

function loadSourceDefaults(source) {
  state.source = source.key;
  state.expressions = Object.fromEntries(source.expressions.map((f) => [f.name, f.default]));
  state.ranges = Object.fromEntries(source.ranges.map((r) => [r.name, { low: r.low, high: r.high }]));
  state.samples = source.samples.default;
  state.views = source.defaultViews.slice();
}

function autosize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight + 2, 160)}px`;
}

function buildExpressionFields(source) {
  dom.expressionFields.replaceChildren();
  for (const field of source.expressions) {
    const wrap = document.createElement('div');
    wrap.className = 'expression-field';

    const label = document.createElement('label');
    label.textContent = field.label;
    label.htmlFor = `expr-${field.name}`;

    const input = document.createElement('textarea');
    input.id = `expr-${field.name}`;
    input.className = 'control';
    input.rows = 1;
    input.spellcheck = false;
    input.placeholder = field.placeholder;
    input.value = state.expressions[field.name] ?? field.default;

    input.addEventListener('input', () => {
      state.expressions[field.name] = input.value;
      autosize(input);
      schedulePlot(DEBOUNCE_FAST);
    });
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        redraw();
      }
    });

    wrap.append(label, input);
    dom.expressionFields.append(wrap);
    requestAnimationFrame(() => autosize(input));
  }
}

function buildRangeFields(source) {
  dom.rangeFields.replaceChildren();
  for (const spec of source.ranges) {
    const row = document.createElement('div');
    row.className = 'range-field';

    const name = document.createElement('span');
    name.textContent = spec.label;

    const dash = document.createElement('span');
    dash.className = 'dash';
    dash.textContent = 'to';

    const current = state.ranges[spec.name] || { low: spec.low, high: spec.high };
    const inputs = ['low', 'high'].map((bound) => {
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'control';
      input.spellcheck = false;
      input.value = current[bound];
      input.setAttribute('aria-label', `${spec.label} ${bound}`);
      input.addEventListener('input', () => {
        state.ranges[spec.name] = { ...state.ranges[spec.name], [bound]: input.value };
        schedulePlot(DEBOUNCE_HEAVY);
      });
      return input;
    });

    state.ranges[spec.name] = current;
    row.append(name, inputs[0], dash, inputs[1]);
    dom.rangeFields.append(row);
  }
}

function buildSamples(source) {
  const spec = source.samples;
  dom.samplesLabel.textContent = spec.label;
  dom.samples.min = spec.min;
  dom.samples.max = spec.max;
  dom.samples.step = spec.max > 500 ? 25 : 1;
  dom.samples.value = state.samples ?? spec.default;
  dom.samplesValue.textContent = dom.samples.value;
}

function buildExamples(source) {
  dom.examples.replaceChildren();
  for (const example of source.examples) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = example.label;
    chip.addEventListener('click', () => {
      state.expressions = { ...state.expressions, ...example.expressions };
      for (const [name, bounds] of Object.entries(example.ranges || {})) {
        state.ranges[name] = { ...bounds };
      }
      buildExpressionFields(source);
      buildRangeFields(source);
      redraw();
    });
    dom.examples.append(chip);
  }
}

function buildReference() {
  dom.referenceBody.replaceChildren();
  for (const group of catalog.functions) {
    const block = document.createElement('div');
    block.className = 'reference-group';

    const heading = document.createElement('h4');
    heading.textContent = group.group;

    const list = document.createElement('div');
    for (const name of group.names) {
      const tag = document.createElement('code');
      tag.textContent = name;
      list.append(tag);
    }

    block.append(heading, list);
    dom.referenceBody.append(block);
  }
}

function buildSourceForm({ replot = true } = {}) {
  const source = currentSource();
  dom.source.value = source.key;
  dom.sourceSummary.textContent = source.summary;
  buildExpressionFields(source);
  buildRangeFields(source);
  buildSamples(source);
  buildExamples(source);
  syncPanelSelects();
  invalidateAll();
  if (replot) requestPlot();
}

/* ------------------------------------------------------------ parameters */

function nextParameterName() {
  const taken = new Set(state.parameters.map((p) => p.name));
  for (const code of 'abcdkmnrs') {
    if (!taken.has(code)) return code;
  }
  return `p${state.parameters.length + 1}`;
}

function renderParameters() {
  dom.params.replaceChildren();

  state.parameters.forEach((param, index) => {
    const row = document.createElement('div');
    row.className = 'param';

    const name = document.createElement('input');
    name.className = 'param-name';
    name.value = param.name;
    name.maxLength = 12;
    name.setAttribute('aria-label', 'Parameter name');

    const equals = document.createElement('span');
    equals.className = 'param-eq';
    equals.textContent = '=';

    const value = document.createElement('input');
    value.type = 'number';
    value.className = 'param-value';
    value.step = 'any';
    value.value = param.value;
    value.setAttribute('aria-label', `Value of ${param.name}`);

    const track = document.createElement('div');
    track.className = 'param-track';

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'track';
    slider.min = param.min;
    slider.max = param.max;
    slider.step = (param.max - param.min) / 400;
    slider.value = param.value;
    slider.setAttribute('aria-label', `Slider for ${param.name}`);

    const bounds = document.createElement('div');
    bounds.className = 'param-bounds';

    const boundInputs = ['min', 'max'].map((which) => {
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'param-bound';
      input.value = param[which];
      input.setAttribute('aria-label', `${which} of ${param.name}`);
      input.addEventListener('change', () => {
        const low = Number(boundInputs[0].value);
        const high = Number(boundInputs[1].value);
        if (!Number.isFinite(low) || !Number.isFinite(high) || low >= high) {
          boundInputs[0].value = param.min;
          boundInputs[1].value = param.max;
          return;
        }
        param.min = low;
        param.max = high;
        param.value = clamp(param.value, low, high);
        slider.min = low;
        slider.max = high;
        slider.step = (high - low) / 400;
        slider.value = param.value;
        value.value = Number(param.value.toFixed(4));
        schedulePlot(DEBOUNCE_FAST);
      });
      return input;
    });

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'param-remove';
    remove.innerHTML = '&times;';
    remove.title = `Remove ${param.name}`;

    name.addEventListener('input', () => {
      param.name = name.value.trim();
      name.style.width = `${Math.max(param.name.length, 2)}ch`;
      schedulePlot(DEBOUNCE_FAST);
    });
    name.style.width = `${Math.max(param.name.length, 2)}ch`;

    slider.addEventListener('input', () => {
      param.value = Number(slider.value);
      value.value = Number(param.value.toFixed(4));
      schedulePlot(DEBOUNCE_FAST);
    });

    value.addEventListener('input', () => {
      const next = Number(value.value);
      if (!Number.isFinite(next)) return;
      param.value = next;
      slider.value = clamp(next, param.min, param.max);
      schedulePlot(DEBOUNCE_FAST);
    });

    remove.addEventListener('click', () => {
      state.parameters.splice(index, 1);
      renderParameters();
      redraw();
    });

    bounds.append(boundInputs[0], boundInputs[1]);
    track.append(slider, bounds);
    row.append(name, equals, value, track, remove);
    dom.params.append(row);
  });
}

function addParameter() {
  state.parameters.push({ name: nextParameterName(), value: 1, min: -10, max: 10 });
  renderParameters();
  redraw();
}

/* ---------------------------------------------------------------- panels */

function createPanel(index) {
  const fragment = dom.panelTemplate.content.cloneNode(true);
  const root = fragment.querySelector('.panel');
  const select = fragment.querySelector('.panel-view');
  const equations = fragment.querySelector('.panel-equations');
  const note = fragment.querySelector('.panel-note');
  const plot = fragment.querySelector('.panel-plot');

  select.addEventListener('change', () => {
    state.views[index] = select.value;
    panel.dirty = true;
    syncUrl();
    if (index === state.panel) requestPlot([index]);
  });

  const panel = { root, select, equations, note, plot, rendered: false, dirty: true };
  panels.push(panel);
  dom.panels.append(fragment);

  const observer = new ResizeObserver(() => {
    // A hidden panel measures 0x0; resizing a figure to that ruins it, and it
    // is re-fitted on reveal anyway.
    if (!panel.rendered || !window.Plotly || root.hidden || !plot.offsetWidth) return;
    cancelAnimationFrame(panel.resizeFrame);
    panel.resizeFrame = requestAnimationFrame(() => window.Plotly.Plots.resize(plot));
  });
  observer.observe(plot);
  return panel;
}

function syncPanelSelects() {
  const source = currentSource();
  const options = catalog.transforms[source.domain] || [];

  panels.forEach((panel, index) => {
    const wanted = state.views[index] || options[index % options.length]?.key;
    if (panel.domain !== source.domain) {
      panel.select.replaceChildren();
      for (const option of options) {
        const node = document.createElement('option');
        node.value = option.key;
        node.textContent = option.label;
        node.title = option.summary;
        panel.select.append(node);
      }
      panel.domain = source.domain;
    }
    const valid = options.some((o) => o.key === wanted) ? wanted : options[0]?.key;
    state.views[index] = valid;
    panel.select.value = valid;
  });
}

/**
 * Put one panel on screen. A panel that is already up to date is simply
 * revealed and re-fitted -- no request, no re-render.
 */
function showPanel(index) {
  state.panel = clamp(index, 0, MAX_PANELS - 1);

  panels.forEach((panel, at) => {
    panel.root.hidden = at !== state.panel;
  });
  for (const button of dom.panelPicker.querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(Number(button.dataset.panel) === state.panel));
  }
  syncUrl();

  const panel = panels[state.panel];
  if (!panel) return;
  if (panel.dirty || !panel.rendered) {
    requestPlot([state.panel]);
  } else if (window.Plotly) {
    // It was measured while hidden, so give it the current size.
    requestAnimationFrame(() => window.Plotly.Plots.resize(panel.plot));
  }
}

async function renderFigure(panel, figure) {
  const Plotly = await whenPlotly();
  const layout = { ...figure.layout, autosize: true };
  await Plotly.react(panel.plot, figure.data, layout, PLOT_CONFIG);
  Plotly.Plots.resize(panel.plot);
  panel.rendered = true;
  panel.dirty = false;
  panel.note.textContent = figure.note || '';
  panel.equations.textContent = (figure.equations || []).join('   ');
  panel.equations.title = (figure.equations || []).join('\n');
}

/* --------------------------------------------------------------- request */

function buildPayload(indices) {
  const parameters = {};
  for (const param of state.parameters) {
    if (param.name) parameters[param.name] = param.value;
  }
  return {
    source: state.source,
    expressions: state.expressions,
    ranges: state.ranges,
    samples: Number(dom.samples.value) || state.samples,
    parameters,
    views: indices.map((index) => state.views[index]),
    theme: state.theme,
  };
}

function schedulePlot(delay = DEBOUNCE_FAST, indices = null) {
  invalidateAll();
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => requestPlot(indices), delay);
}

/**
 * Work out which panels this request must cover.
 *
 * Aborting an in-flight request kills it for *every* panel it was fetching, so
 * if the new request overlaps an older one it has to adopt the older one's
 * other panels too -- otherwise those panels would keep a stale figure and
 * never be redrawn.
 */
function expandTargets(indices) {
  const targets = new Set(indices.filter((index) => panels[index]));
  let grew = true;
  while (grew) {
    grew = false;
    for (const [index, controller] of inflight) {
      if (!targets.has(index)) continue;
      for (const [other, owner] of inflight) {
        if (owner === controller && !targets.has(other)) {
          targets.add(other);
          grew = true;
        }
      }
    }
  }
  return [...targets].sort((a, b) => a - b);
}

function release(targets, controller) {
  for (const index of targets) {
    if (inflight.get(index) !== controller) continue;   // a newer request owns it
    inflight.delete(index);
    panels[index]?.root.classList.remove('busy');
  }
}

/** Everything changed: redraw what is on screen, mark the rest stale. */
function redraw() {
  invalidateAll();
  return requestPlot();
}

async function requestPlot(indices = null) {
  clearTimeout(debounceTimer);
  if (!catalog) return;

  const targets = expandTargets(indices || visibleIndices());
  if (!targets.length) return;

  const controller = new AbortController();
  const superseded = new Set(targets.map((index) => inflight.get(index)).filter(Boolean));
  for (const old of superseded) old.abort();
  for (const index of targets) {
    inflight.set(index, controller);
    panels[index].root.classList.add('busy');
  }
  setStatus('Drawing…', true);

  let text;
  try {
    const response = await fetch('/api/plot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload(targets)),
      signal: controller.signal,
    });
    text = await response.text();
  } catch (error) {
    release(targets, controller);
    if (error.name === 'AbortError') return;          // superseded by a newer request
    setStatus('Offline');
    showError(`Could not reach the server: ${error.message}`);
    return;
  }

  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    release(targets, controller);
    setStatus('Failed');
    showError('The server returned a malformed response.');
    return;
  }

  if (!payload.ok) {
    release(targets, controller);
    setStatus('Failed');
    showError(payload.error || 'Unknown error.');
    return;
  }

  clearError();
  dom.caption.textContent = payload.caption || '';

  await Promise.all(payload.figures.map((figure, slot) => {
    const panel = panels[targets[slot]];
    return panel ? renderFigure(panel, figure) : null;
  }));

  release(targets, controller);
  const size = text.length < 1024 * 1024
    ? `${Math.round(text.length / 1024)} KB`
    : `${(text.length / 1048576).toFixed(1)} MB`;
  const count = payload.figures.length > 1 ? `${payload.figures.length} views · ` : '';
  setStatus(`${count}${payload.elapsedMs} ms · ${size}`);
  syncUrl();
}

/* ------------------------------------------------------------- listeners */

function bindListeners() {
  dom.source.addEventListener('change', () => {
    loadSourceDefaults(catalog.sources.find((s) => s.key === dom.source.value));
    buildSourceForm();
  });

  dom.samples.addEventListener('input', () => {
    state.samples = Number(dom.samples.value);
    dom.samplesValue.textContent = dom.samples.value;
    schedulePlot(DEBOUNCE_HEAVY);
  });

  dom.addParameter.addEventListener('click', addParameter);

  dom.panelPicker.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-panel]');
    if (button) showPanel(Number(button.dataset.panel));
  });

  dom.themeToggle.addEventListener('click', () => {
    applyTheme(state.theme === 'dark' ? 'light' : 'dark');
    redraw();                 // every figure carries its theme's colours
  });

  dom.sidebarToggle.addEventListener('click', () => {
    const hidden = dom.body.classList.toggle('sidebar-hidden');
    dom.sidebarToggle.setAttribute('aria-expanded', String(!hidden));
    dom.sidebarToggle.title = hidden ? 'Show controls (\\)' : 'Hide controls (\\)';
  });

  document.addEventListener('keydown', (event) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);

    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      redraw();
      return;
    }
    if (typing || event.ctrlKey || event.metaKey || event.altKey) return;

    if (event.key >= '1' && event.key <= '3') {
      showPanel(Number(event.key) - 1);
    } else if (event.key.toLowerCase() === 'd') {
      dom.themeToggle.click();
    } else if (event.key === '\\') {
      dom.sidebarToggle.click();
    }
  });

  window.addEventListener('resize', () => {
    const panel = panels[state.panel];
    if (window.Plotly && panel?.rendered) window.Plotly.Plots.resize(panel.plot);
  });
}

/* ------------------------------------------------------------------ boot */

async function start() {
  applyTheme(initialTheme());

  let response;
  try {
    response = await fetch('/api/catalog');
    catalog = await response.json();
  } catch (error) {
    setStatus('Failed');
    showError(`Could not load the catalogue: ${error.message}`);
    return;
  }

  for (const source of catalog.sources) {
    const option = document.createElement('option');
    option.value = source.key;
    option.textContent = source.label;
    dom.source.append(option);
  }

  buildReference();
  for (let index = 0; index < MAX_PANELS; index += 1) createPanel(index);

  if (!restoreFromUrl()) loadSourceDefaults(catalog.sources[0]);

  renderParameters();
  bindListeners();
  buildSourceForm({ replot: false });
  showPanel(state.panel);
}

start();
