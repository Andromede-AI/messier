const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  displayModeBar: false,
};

const AXIS_TITLE_FONT = { size: 13 };
const AXIS_TICK_FONT = { size: 11 };

const PLOT_LAYOUT = {
  autosize: true,
  height: 300,
  margin: { l: 54, r: 16, t: 12, b: 50 },
  font: { family: 'Noto Sans, sans-serif', color: '#363636', size: 14 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  hoverlabel: { bgcolor: '#ffffff', bordercolor: '#363636', font: { color: '#363636' } },
  xaxis: { gridcolor: '#e4e4e4', zerolinecolor: '#b5b5b5', automargin: true, tickangle: 0, tickfont: AXIS_TICK_FONT },
  yaxis: { gridcolor: '#e4e4e4', zerolinecolor: '#b5b5b5', automargin: true, tickfont: AXIS_TICK_FONT },
};

const CATEGORICAL_X_AXIS = {
  ...PLOT_LAYOUT.xaxis,
  tickangle: -30,
  showgrid: false,
  zeroline: false,
  fixedrange: true,
};

const GROUP_LABELS = {
  programming: 'Programming',
  enterprise: 'Enterprise',
  research: 'Research',
  gui: 'GUI',
  function_calling: 'Function<br>calling',
};

const GROUP_COLORS = {
  programming: '#3A55D6',
  enterprise: '#B1592C',
  research: '#4A7355',
  gui: '#7A5D8A',
  function_calling: '#A85257',
};

function fittedLine(x, y) {
  const xMean = x.reduce((sum, value) => sum + value, 0) / x.length;
  const yMean = y.reduce((sum, value) => sum + value, 0) / y.length;
  const numerator = x.reduce((sum, value, index) => sum + (value - xMean) * (y[index] - yMean), 0);
  const denominator = x.reduce((sum, value) => sum + (value - xMean) ** 2, 0);
  const slope = numerator / denominator;
  const intercept = yMean - slope * xMean;
  const bounds = [Math.min(...x), Math.max(...x)];
  return { x: bounds, y: bounds.map((value) => slope * value + intercept) };
}

function summarize(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return { mean, std: Math.sqrt(variance) };
}

function densityCurve(values, start, end, steps = 160) {
  const { std } = summarize(values);
  const bandwidth = 1.06 * std * values.length ** -0.2;
  const scale = values.length * bandwidth * Math.sqrt(2 * Math.PI);
  const x = Array.from({ length: steps + 1 }, (_, index) => start + (end - start) * index / steps);
  const y = x.map((point) => values.reduce((sum, value) => sum + Math.exp(-0.5 * ((point - value) / bandwidth) ** 2), 0) / scale);
  return { x, y };
}

function drawCompositionBar(id, labels, values, options = {}) {
  const plot = document.getElementById(id);
  const trace = {
    x: labels,
    y: values,
    type: 'bar',
    marker: { color: '#3A55D6', line: { color: '#252525', width: 0.6 } },
    hovertemplate: `%{x}<br>%{y:,} ${options.unit || ''}<extra></extra>`,
  };
  const layout = {
    ...PLOT_LAYOUT,
    height: plot.clientHeight,
    margin: { l: 54, r: 10, t: 8, b: 82 },
    bargap: 0.28,
    showlegend: false,
    xaxis: {
      ...CATEGORICAL_X_AXIS,
    },
    yaxis: {
      ...PLOT_LAYOUT.yaxis,
      type: options.log ? 'log' : 'linear',
      title: { text: options.axisTitle || '', font: AXIS_TITLE_FONT },
      dtick: options.log ? 1 : undefined,
      tickformat: '~s',
      rangemode: 'tozero',
      fixedrange: true,
    },
  };
  Plotly.newPlot(plot, [trace], layout, PLOT_CONFIG);
}

function drawComposition(result) {
  const groupLabels = result.benchmark_groups.map((entry) => GROUP_LABELS[entry.group].toLowerCase());
  drawCompositionBar(
    'plot-composition-benchmarks',
    groupLabels,
    result.benchmark_groups.map((entry) => entry.benchmarks),
    { axisTitle: 'benchmarks', unit: 'benchmarks' },
  );
  drawCompositionBar(
    'plot-composition-tasks',
    groupLabels,
    result.benchmark_groups.map((entry) => entry.tasks),
    { axisTitle: 'tasks', unit: 'tasks' },
  );
  drawCompositionBar(
    'plot-composition-identities',
    ['models', 'scaffolds', 'agents'],
    [result.identities.models, result.identities.scaffolds, result.identities.agents],
    { axisTitle: 'count' },
  );
  drawCompositionBar(
    'plot-composition-scoring',
    ['direct', 'all-pass', 'threshold'],
    [result.scoring_rules.direct, result.scoring_rules.all_pass, result.scoring_rules.threshold],
    { axisTitle: 'tasks', unit: 'tasks' },
  );
  drawCompositionBar(
    'plot-composition-verifiers',
    ['llm<br>judge', 'script', 'exact<br>match', 'human<br>label'],
    [
      result.verifier_types.llm_judge,
      result.verifier_types.script,
      result.verifier_types.exact_match,
      result.verifier_types.human_label,
    ],
    { axisTitle: 'verifiers', unit: 'verifiers', log: true },
  );

  const frontier = result.frontier.map((series) => ({
    ...series,
    points: series.points.filter((point) => point.quarter >= '2024Q1'),
  }));
  const frontierPlot = document.getElementById('plot-composition-frontier');
  const frontierTraces = frontier.map((series) => ({
    x: series.points.map((point) => point.quarter),
    y: series.points.map((point) => point.mean),
    customdata: series.points.map((point) => point.se),
    type: 'scatter',
    mode: 'lines+markers',
    name: GROUP_LABELS[series.group],
    line: { color: GROUP_COLORS[series.group], width: 2.5, shape: 'spline' },
    marker: { color: GROUP_COLORS[series.group], size: 6 },
    error_y: {
      type: 'data',
      array: series.points.map((point) => point.se),
      color: GROUP_COLORS[series.group],
      thickness: 1,
      width: 2,
      visible: true,
    },
    hovertemplate: `${GROUP_LABELS[series.group]}<br>%{x}<br>Frontier result %{y:.3f}<br>SE %{customdata:.3f}<extra></extra>`,
  }));
  Plotly.newPlot(
    frontierPlot,
    frontierTraces,
    {
      ...PLOT_LAYOUT,
      height: frontierPlot.clientHeight,
      margin: { l: 58, r: 18, t: 12, b: 52 },
      legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.13, yanchor: 'bottom', font: { size: 11 } },
      xaxis: {
        ...PLOT_LAYOUT.xaxis,
        title: { text: 'Model release quarter', font: AXIS_TITLE_FONT },
        tickmode: 'array',
        tickvals: ['2024Q1', '2025Q1', '2026Q1'],
        ticktext: ['2024', '2025', '2026'],
        fixedrange: true,
      },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: { text: 'Frontier result', font: AXIS_TITLE_FONT }, range: [0, 1.03], fixedrange: true },
    },
    PLOT_CONFIG,
  );

  const changePlot = document.getElementById('plot-composition-frontier-change');
  const changeTraces = [];
  frontier.forEach((series) => {
    const first = series.points[0];
    const latest = series.points[series.points.length - 1];
    const label = GROUP_LABELS[series.group];
    const color = GROUP_COLORS[series.group];
    changeTraces.push({
      x: [first.mean, latest.mean],
      y: [label, label],
      type: 'scatter',
      mode: 'lines',
      line: { color, width: 2 },
      hoverinfo: 'skip',
      showlegend: false,
    });
    changeTraces.push({
      x: [first.mean, latest.mean],
      y: [label, label],
      customdata: [first.quarter, latest.quarter],
      text: ['', `${latest.mean.toFixed(2)} (+${(latest.mean - first.mean).toFixed(2)})`],
      type: 'scatter',
      mode: 'markers+text',
      marker: { color: ['#ffffff', color], size: 10, line: { color, width: 2 } },
      textposition: 'middle right',
      textfont: { size: 11, color: '#363636' },
      cliponaxis: false,
      hovertemplate: '%{y}<br>%{customdata}<br>Frontier result %{x:.3f}<extra></extra>',
      showlegend: false,
    });
  });
  Plotly.newPlot(
    changePlot,
    changeTraces,
    {
      ...PLOT_LAYOUT,
      height: changePlot.clientHeight,
      margin: { l: 116, r: 106, t: 12, b: 52 },
      showlegend: false,
      xaxis: { ...PLOT_LAYOUT.xaxis, title: { text: 'Frontier result', font: AXIS_TITLE_FONT }, range: [0, 1.03], fixedrange: true },
      yaxis: { ...PLOT_LAYOUT.yaxis, showgrid: false, fixedrange: true },
    },
    PLOT_CONFIG,
  );
}

function drawIrtCoverage(result) {
  const plot = document.getElementById('plot-irt-coverage');
  const coverage = result.coverage;
  const groups = Object.keys(GROUP_LABELS);
  const groupPresence = groups.map((group) => {
    const benchmarkIndexes = coverage.benchmark_groups
      .map((benchmarkGroup, index) => benchmarkGroup === group ? index : -1)
      .filter((index) => index >= 0);
    return coverage.models.map((_, modelIndex) => (
      benchmarkIndexes.some((benchmarkIndex) => coverage.observed[modelIndex][benchmarkIndex] > 0)
    ));
  });
  const overlap = groups.map((_, row) => (
    groups.map((__, column) => groupPresence[row].reduce((count, present, modelIndex) => (
      count + Number(present && groupPresence[column][modelIndex])
    ), 0))
  ));
  const labels = groups.map((group) => GROUP_LABELS[group]);
  const offDiagonal = overlap.map((row, rowIndex) => row.map((value, columnIndex) => (
    rowIndex === columnIndex ? null : value
  )));
  const diagonal = overlap.map((row, rowIndex) => row.map((value, columnIndex) => (
    rowIndex === columnIndex ? value : null
  )));
  const trace = {
    x: labels,
    y: labels,
    z: offDiagonal,
    text: overlap.map((row, rowIndex) => row.map((value, columnIndex) => (
      rowIndex === columnIndex ? '' : value.toLocaleString()
    ))),
    customdata: overlap.map((row, rowIndex) => row.map((value, columnIndex) => (
      rowIndex === columnIndex ? '' : `${labels[rowIndex]} and ${labels[columnIndex]}<br>${value} shared models`
    ))),
    texttemplate: '%{text}',
    textfont: { size: 13 },
    type: 'heatmap',
    colorscale: [[0, '#f1f1f1'], [0.12, '#e2e6f7'], [1, '#3A55D6']],
    zmin: 0,
    zmax: Math.max(...offDiagonal.flat().filter(Number.isFinite)),
    showscale: false,
    hovertemplate: '%{customdata}<extra></extra>',
  };
  const diagonalTrace = {
    x: labels,
    y: labels,
    z: diagonal,
    text: diagonal.map((row) => row.map((value) => value === null ? '' : `n=${value.toLocaleString()}`)),
    customdata: diagonal.map((row, rowIndex) => row.map((value) => (
      value === null ? '' : `${labels[rowIndex]}<br>${value} evaluated models`
    ))),
    texttemplate: '%{text}',
    textfont: { size: 13 },
    type: 'heatmap',
    colorscale: [[0, '#d9d9d9'], [1, '#d9d9d9']],
    showscale: false,
    hovertemplate: '%{customdata}<extra></extra>',
  };
  const layout = {
    ...PLOT_LAYOUT,
    height: plot.clientHeight,
    margin: { l: 112, r: 12, t: 48, b: 92 },
    xaxis: {
      ...CATEGORICAL_X_AXIS,
    },
    yaxis: {
      ...PLOT_LAYOUT.yaxis,
      tickfont: AXIS_TICK_FONT,
      showgrid: false,
      zeroline: false,
      autorange: 'reversed',
      scaleanchor: 'x',
      scaleratio: 1,
      fixedrange: true,
    },
  };
  Plotly.newPlot(plot, [trace, diagonalTrace], layout, PLOT_CONFIG);
  document.getElementById('stat-irt-coverage').textContent = 'Diagonal cells show the models evaluated in each benchmark group. Off-diagonal cells show models shared by the corresponding groups. A model may appear in several cells.';
}

function drawIrt(result) {
  const plot = document.getElementById('plot-irt-distributions');
  const betaDensity = densityCurve(result.beta, -7, 6);
  const thetaDensity = densityCurve(result.theta, -7, 6);
  const traces = [
    {
      x: betaDensity.x,
      y: betaDensity.y,
      type: 'scatter',
      mode: 'lines',
      name: `Task difficulty β (${result.beta.length.toLocaleString()})`,
      line: { color: '#59606C', width: 2.5 },
      fill: 'tozeroy',
      fillcolor: 'rgba(89, 96, 108, 0.22)',
      hovertemplate: 'Task difficulty %{x:.2f}<br>Density %{y:.3f}<extra></extra>',
    },
    {
      x: thetaDensity.x,
      y: thetaDensity.y,
      type: 'scatter',
      mode: 'lines',
      name: `Model ability θ (${result.theta.length.toLocaleString()})`,
      line: { color: '#3A55D6', width: 2.5 },
      fill: 'tozeroy',
      fillcolor: 'rgba(58, 85, 214, 0.22)',
      hovertemplate: 'Model ability %{x:.2f}<br>Density %{y:.3f}<extra></extra>',
    },
  ];
  const layout = {
    ...PLOT_LAYOUT,
    height: plot.clientHeight,
    margin: { l: 54, r: 16, t: 48, b: 92 },
    legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.12, yanchor: 'top', font: { size: 12 } },
    xaxis: { ...PLOT_LAYOUT.xaxis, title: { text: 'Logit', font: AXIS_TITLE_FONT }, range: [-7, 6] },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: { text: 'Density', font: AXIS_TITLE_FONT }, rangemode: 'tozero' },
  };
  Plotly.newPlot(plot, traces, layout, PLOT_CONFIG);

  const theta = summarize(result.theta);
  const beta = summarize(result.beta);
  document.getElementById('stat-irt').textContent = `The density curves summarize estimates for ${result.theta.length.toLocaleString()} models and ${result.beta.length.toLocaleString()} tasks. Model ability θ has mean ${theta.mean.toFixed(2)} and SD ${theta.std.toFixed(2)}. Task difficulty β has mean ${beta.mean.toFixed(2)} and SD ${beta.std.toFixed(2)}.`;
}

function drawEci(result) {
  const x = result.points.map((point) => point.theta);
  const y = result.points.map((point) => point.eci);
  const line = fittedLine(x, y);
  const traces = [
    {
      x: line.x,
      y: line.y,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#7a7a7a', width: 1.5 },
      hoverinfo: 'skip',
      showlegend: false,
    },
    {
      x,
      y,
      customdata: result.points.map((point) => point.model),
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#3A55D6', size: 8, line: { color: '#252525', width: 0.7 } },
      hovertemplate: '<b>%{customdata}</b><br>Messier ability %{x:.2f}<br>Epoch ECI %{y:.1f}<extra></extra>',
      showlegend: false,
    },
  ];
  const layout = {
    ...PLOT_LAYOUT,
    xaxis: { ...PLOT_LAYOUT.xaxis, title: { text: 'Messier ability', font: AXIS_TITLE_FONT } },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: { text: 'Epoch ECI', font: AXIS_TITLE_FONT } },
  };
  Plotly.newPlot(`plot-eci-${result.slug}`, traces, layout, PLOT_CONFIG);
  document.getElementById(`stat-eci-${result.slug}`).textContent = `ρ = ${result.rho.toFixed(2)}, ${result.n} models`;
}

function drawIndustry(result, axisRange, color) {
  const x = result.points.map((point) => point.overall_theta);
  const y = result.points.map((point) => point.industry_theta);
  const traces = [
    {
      x: axisRange,
      y: axisRange,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#7a7a7a', width: 1.5, dash: 'dot' },
      hoverinfo: 'skip',
      showlegend: false,
    },
    {
      x,
      y,
      customdata: result.points.map((point) => [point.agent_id, point.n_tasks, point.n_benchmark_groups]),
      type: 'scatter',
      mode: 'markers',
      marker: { color, size: 8, line: { color: '#252525', width: 0.7 } },
      hovertemplate: '<b>%{customdata[0]}</b><br>Overall ability %{x:.2f}<br>Industry ability %{y:.2f}<br>%{customdata[1]} tasks across %{customdata[2]} benchmark groups<extra></extra>',
      showlegend: false,
    },
  ];
  const layout = {
    ...PLOT_LAYOUT,
    xaxis: { ...PLOT_LAYOUT.xaxis, title: { text: 'Overall ability', font: AXIS_TITLE_FONT }, range: axisRange },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: { text: 'Industry ability', font: AXIS_TITLE_FONT }, range: axisRange },
  };
  Plotly.newPlot(`plot-industry-${result.slug}`, traces, layout, PLOT_CONFIG);
  document.getElementById(`stat-industry-${result.slug}`).textContent = `ρ = ${result.rho.toFixed(2)}, ${result.n} models`;
}

async function drawResults() {
  const response = await fetch('./static/data/results.json');
  if (!response.ok) {
    throw new Error(`Could not load result data (${response.status})`);
  }
  const results = await response.json();
  drawComposition(results.composition);
  drawIrtCoverage(results.irt);
  drawIrt(results.irt);
  results.eci.forEach((result) => drawEci(result));

  const industryValues = results.industries.flatMap((result) => result.points.flatMap((point) => [point.overall_theta, point.industry_theta]));
  const padding = 0.2;
  const axisRange = [Math.min(...industryValues) - padding, Math.max(...industryValues) + padding];
  const industryColors = ['#2F7D77', '#B44C5C', '#B17A18'];
  results.industries.forEach((result, index) => drawIndustry(result, axisRange, industryColors[index]));
}

document.addEventListener('DOMContentLoaded', () => {
  drawResults().catch((error) => {
    document.querySelectorAll('.interactive-plot').forEach((plot) => {
      plot.textContent = 'Result data could not be loaded.';
    });
    console.error(error);
  });
});
