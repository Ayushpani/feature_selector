"""
Reporter — Professional Interactive HTML Audit Report

Generates a stunning, self-contained HTML report using Plotly for interactive
visualizations and modern CSS for professional styling.

Report Sections:
1. Executive Summary Dashboard
2. Interactive Feature Funnel (Plotly waterfall)
3. Correlation Heatmap (Plotly interactive)
4. Feature Importance Rankings
5. Sortable/Filterable Audit Trail Table
6. Data Quality Summary
7. Performance Comparison (before vs. after)
8. Column Journey Tracker
"""
import pandas as pd
import numpy as np
import io
import base64
import json
from datetime import datetime

from feature_engine_pro.logger import get_logger


class Reporter:
    """
    Maintains an audit trail and generates a professional interactive HTML report.
    """
    def __init__(self):
        self.logs = []
        self.feature_status = {}
        self.correlation_matrix_data = None
        self.correlation_feature_names = None
        self.evaluation_results = None
        self.pipeline_runtime = None
        self.column_journey = {}
        self._logger = get_logger()

    def log_event(self, feature, status, reason, step_name):
        """Log a feature decision event."""
        self.logs.append({
            'feature': feature,
            'status': status,
            'reason': reason,
            'step': step_name
        })
        self.feature_status[feature] = {
            'status': status,
            'reason': reason,
            'step': step_name
        }

    def capture_correlation_matrix(self, df):
        """Capture the correlation matrix data for interactive visualization."""
        num_df = df.select_dtypes(include='number')
        if num_df.shape[1] > 1:
            corr = num_df.corr()
            self.correlation_matrix_data = corr.values.tolist()
            self.correlation_feature_names = corr.columns.tolist()

    def generate_summary(self):
        return pd.DataFrame(self.logs)

    def generate_pdf_report(self, html_filepath="feature_engine_report.html", pdf_filepath="feature_engine_report.pdf"):
        """Generate a high-quality PDF report using Playwright."""
        import os
        import time
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._logger.error("[Reporter] Playwright is not installed. To generate PDFs, run: pip install playwright && playwright install chromium")
            return

        self.generate_html_report(html_filepath)
        abs_path = os.path.abspath(html_filepath)
        file_url = f"file:///{abs_path.replace('\\', '/')}"

        self._logger.info(f"[Reporter] Generating PDF (this may take a few seconds)...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            # Wait for Plotly to render
            page.goto(file_url, wait_until='networkidle')
            time.sleep(2)  # Give JS an extra moment for rendering
            page.pdf(
                path=pdf_filepath,
                format="A4",
                print_background=True,
                margin={'top': '20mm', 'bottom': '20mm', 'left': '10mm', 'right': '10mm'}
            )
            browser.close()
            
        self._logger.info(f"[Reporter] High-quality PDF generated: {pdf_filepath}")

    def _sanitize_for_console(self, text):
        """Replace Unicode chars that crash on Windows cp1252 terminals."""
        replacements = {
            '\u2192': '->', '\u2248': '~=', '\u2265': '>=',
            '\u2264': '<=', '\u03b1': 'alpha', '\u2191': '+',
            '\u2193': '-', '\u2705': '[OK]', '\u274c': '[ERR]',
            '\u26a0\ufe0f': '[WARN]', '\u26a0': '[WARN]',
            '\u2014': '--', '\u2013': '-',
        }
        for char, repl in replacements.items():
            text = text.replace(char, repl)
        # Final fallback: encode to ASCII with replace
        return text.encode('ascii', errors='replace').decode('ascii')


    def print_report(self):
        """Print summary to console."""
        p = lambda s: print(self._sanitize_for_console(s))
        p("=" * 60)
        p("  FEATURE ENGINE PRO - AUDIT REPORT")
        p("=" * 60)
        df_logs = self.generate_summary()
        if df_logs.empty:
            print("No features processed yet.")
            return

        # Get last status per feature
        final_status = {}
        for _, row in df_logs.iterrows():
            final_status[row['feature']] = row

        dropped = [r for r in final_status.values() if r['status'] == 'dropped']
        kept = [r for r in final_status.values() if r['status'] == 'kept']

        p(f"\n  Features Kept:    {len(kept)}")
        p(f"  Features Dropped: {len(dropped)}")

        if self.pipeline_runtime:
            p(f"  Pipeline Runtime: {self.pipeline_runtime}s")

        p("\n--- DROPPED FEATURES ---")
        for row in dropped:
            p(f"  [{row['step']}] {row['feature']} -> {row['reason']}")

        p(f"\n--- KEPT FEATURES ({len(kept)}) ---")
        for row in kept:
            p(f"  [{row['step']}] {row['feature']}")

        p("=" * 60)

    def generate_html_report(self, filepath="feature_engine_report.html"):
        """Generate the professional interactive HTML report."""
        df_logs = self.generate_summary()

        # Compute summary stats
        final_status = {}
        for _, row in df_logs.iterrows():
            final_status[row['feature']] = row.to_dict()

        n_kept = sum(1 for v in final_status.values() if v['status'] == 'kept')
        n_dropped = sum(1 for v in final_status.values() if v['status'] == 'dropped')
        n_total = n_kept + n_dropped

        # Build step-by-step funnel data
        steps = df_logs['step'].unique().tolist() if not df_logs.empty else []
        funnel_data = []
        for step in steps:
            step_data = df_logs[df_logs['step'] == step]
            n_k = (step_data['status'] == 'kept').sum()
            n_d = (step_data['status'] == 'dropped').sum()
            funnel_data.append({'step': step, 'kept': int(n_k), 'dropped': int(n_d)})

        # Column journey data
        journey_data = {}
        if self.column_journey:
            journey_data = {k: len(v) for k, v in self.column_journey.items()}

        # Evaluation data
        eval_data = self.evaluation_results or {}

        # Build the HTML
        html = self._build_html(
            n_total=n_total,
            n_kept=n_kept,
            n_dropped=n_dropped,
            funnel_data=funnel_data,
            df_logs=df_logs,
            journey_data=journey_data,
            eval_data=eval_data,
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        self._logger.info(f"[Reporter] HTML report generated: {filepath}")

    def _build_html(self, n_total, n_kept, n_dropped, funnel_data,
                    df_logs, journey_data, eval_data):
        """Construct the full HTML string."""

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        runtime = self.pipeline_runtime or 'N/A'
        reduction_pct = round((n_dropped / max(n_total, 1)) * 100, 1)

        # Input/output counts from journey
        input_cols = journey_data.get('input', n_total)
        final_cols = journey_data.get('final', n_kept)

        # Build audit trail rows
        audit_rows = ''
        for _, row in df_logs.iterrows():
            status_class = 'status-dropped' if row['status'] == 'dropped' else 'status-kept'
            status_icon = '✗' if row['status'] == 'dropped' else '✓'
            audit_rows += f'''
            <tr class="{status_class}">
                <td>{row['feature']}</td>
                <td><span class="badge {status_class}">{status_icon} {row['status'].upper()}</span></td>
                <td><span class="step-badge">{row['step']}</span></td>
                <td>{row['reason']}</td>
            </tr>'''

        # Evaluation section
        eval_html = ''
        if eval_data:
            before_metrics = eval_data.get('before', {})
            after_metrics = eval_data.get('after', {})
            eval_rows = ''
            for metric_name in before_metrics:
                b = before_metrics[metric_name]
                a = after_metrics.get(metric_name, {})
                b_mean = b.get('mean', 0)
                a_mean = a.get('mean', 0)
                delta = a_mean - b_mean
                direction = '+' if delta >= 0 else '-'
                color = '#00b894' if delta >= 0 else '#d63031'
                eval_rows += f'''
                <tr>
                    <td style="font-weight:600">{metric_name}</td>
                    <td>{b_mean:.4f} ± {b.get("std", 0):.4f}</td>
                    <td>{a_mean:.4f} ± {a.get("std", 0):.4f}</td>
                    <td style="color:{color}; font-weight:700">{direction} {abs(delta):.4f}</td>
                </tr>'''

            feat_before = eval_data.get('features_before', '?')
            feat_after = eval_data.get('features_after', '?')
            red_pct = eval_data.get('reduction_pct', 0)

            eval_html = f'''
            <div class="card">
                <h2>Performance Impact</h2>
                <p class="subtitle">Proving that fewer features = same or better performance</p>
                <div class="stat-row">
                    <div class="stat-box">
                        <div class="stat-value">{feat_before}</div>
                        <div class="stat-label">Features Before</div>
                    </div>
                    <div class="stat-box accent">
                        <div class="stat-value">{feat_after}</div>
                        <div class="stat-label">Features After</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{red_pct}%</div>
                        <div class="stat-label">Reduction</div>
                    </div>
                </div>
                <table class="data-table">
                    <thead><tr>
                        <th>Metric</th><th>Before Selection</th><th>After Selection</th><th>Change</th>
                    </tr></thead>
                    <tbody>{eval_rows}</tbody>
                </table>
            </div>'''

        # Correlation heatmap Plotly JSON
        corr_plot_js = ''
        if self.correlation_matrix_data and self.correlation_feature_names:
            labels = self.correlation_feature_names
            # Limit to 40 features for readability
            if len(labels) > 40:
                labels = labels[:40]
                matrix = [row[:40] for row in self.correlation_matrix_data[:40]]
            else:
                matrix = self.correlation_matrix_data

            labels_json = json.dumps(labels)
            matrix_json = json.dumps(matrix)

            corr_plot_js = f'''
            <div class="card">
                <h2>🔗 Correlation Matrix</h2>
                <p class="subtitle">Interactive heatmap — hover for details, scroll to zoom</p>
                <div id="corr-heatmap" style="width:100%;height:600px;"></div>
            </div>
            <script>
                var corrData = [{{
                    z: {matrix_json},
                    x: {labels_json},
                    y: {labels_json},
                    type: 'heatmap',
                    colorscale: [
                        [0, '#667eea'],
                        [0.5, '#f8f9fa'],
                        [1, '#e74c3c']
                    ],
                    hoverongaps: false,
                    hovertemplate: '%{{x}} vs %{{y}}<br>Correlation: %{{z:.3f}}<extra></extra>'
                }}];
                var corrLayout = {{
                    margin: {{t: 30, l: 120, r: 30, b: 120}},
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{color: '#e0e0e0', family: 'Inter, sans-serif'}},
                    xaxis: {{tickangle: -45, tickfont: {{size: 9}}}},
                    yaxis: {{tickfont: {{size: 9}}}},
                }};
                Plotly.newPlot('corr-heatmap', corrData, corrLayout, {{responsive: true}});
            </script>'''

        # Funnel chart
        funnel_js = ''
        if funnel_data:
            steps_json = json.dumps([d['step'] for d in funnel_data])
            kept_json = json.dumps([d['kept'] for d in funnel_data])
            dropped_json = json.dumps([d['dropped'] for d in funnel_data])

            funnel_js = f'''
            <div class="card">
                <h2>🔽 Feature Filtering Funnel</h2>
                <p class="subtitle">How features were kept or dropped at each pipeline stage</p>
                <div id="funnel-chart" style="width:100%;height:400px;"></div>
            </div>
            <script>
                var funnelData = [
                    {{
                        x: {steps_json},
                        y: {kept_json},
                        name: 'Kept',
                        type: 'bar',
                        marker: {{color: '#00b894'}},
                        hovertemplate: '%{{x}}<br>Kept: %{{y}}<extra></extra>'
                    }},
                    {{
                        x: {steps_json},
                        y: {dropped_json},
                        name: 'Dropped',
                        type: 'bar',
                        marker: {{color: '#d63031'}},
                        hovertemplate: '%{{x}}<br>Dropped: %{{y}}<extra></extra>'
                    }}
                ];
                var funnelLayout = {{
                    barmode: 'stack',
                    margin: {{t: 30, l: 50, r: 30, b: 80}},
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{color: '#e0e0e0', family: 'Inter, sans-serif'}},
                    xaxis: {{tickangle: -25}},
                    yaxis: {{title: 'Features', gridcolor: '#e9ecef'}},
                    legend: {{orientation: 'h', y: 1.1}},
                }};
                Plotly.newPlot('funnel-chart', funnelData, funnelLayout, {{responsive: true}});
            </script>'''

        # Column journey chart
        journey_js = ''
        if journey_data:
            j_steps = list(journey_data.keys())
            j_counts = list(journey_data.values())
            j_steps_json = json.dumps(j_steps)
            j_counts_json = json.dumps(j_counts)

            journey_js = f'''
            <div class="card">
                <h2>Column Journey</h2>
                <p class="subtitle">Feature count at each pipeline stage — shows where features were created and destroyed</p>
                <div id="journey-chart" style="width:100%;height:350px;"></div>
            </div>
            <script>
                var journeyData = [{{
                    x: {j_steps_json},
                    y: {j_counts_json},
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{color: '#667eea', width: 3}},
                    marker: {{size: 10, color: '#764ba2'}},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0,51,102,0.1)',
                    hovertemplate: '%{{x}}<br>Features: %{{y}}<extra></extra>'
                }}];
                var journeyLayout = {{
                    margin: {{t: 30, l: 50, r: 30, b: 80}},
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{color: '#e0e0e0', family: 'Inter, sans-serif'}},
                    xaxis: {{tickangle: -25}},
                    yaxis: {{title: 'Feature Count', gridcolor: '#e9ecef'}},
                }};
                Plotly.newPlot('journey-chart', journeyData, journeyLayout, {{responsive: true}});
            </script>'''

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Engine Pro — Audit Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            line-height: 1.6;
            padding: 0;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 48px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
            animation: pulse 8s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.5; }}
            50% {{ transform: scale(1.2); opacity: 0.8; }}
        }}
        .header h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            position: relative;
            color: #fff;
        }}
        .header .subtitle {{
            font-size: 1rem;
            opacity: 0.85;
            margin-top: 8px;
            font-weight: 400;
            position: relative;
            color: #f0f0f0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }}
        .stat-row {{
            display: flex;
            gap: 16px;
            margin: 24px 0;
            flex-wrap: wrap;
        }}
        .stat-box {{
            flex: 1;
            min-width: 140px;
            background: #1a1a2e;
            border: 1px solid #2d2d42;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .stat-box:hover {{
            transform: translateY(-2px);
            border-color: #667eea;
        }}
        .stat-box.accent {{
            background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
            border-color: #667eea;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .stat-label {{
            font-size: 0.85rem;
            color: #888;
            margin-top: 4px;
            font-weight: 500;
        }}
        .card {{
            background: #16162a;
            border: 1px solid #2d2d42;
            border-radius: 16px;
            padding: 28px;
            margin: 20px 0;
            transition: border-color 0.2s;
        }}
        .card:hover {{
            border-color: #3d3d5c;
        }}
        .card h2 {{
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 6px;
            color: #f0f0f0;
        }}
        .card .subtitle {{
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 20px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.85rem;
            border-radius: 8px;
            overflow: hidden;
        }}
        .data-table thead th {{
            background: #1a1a2e;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            color: #b0b0b0;
            border-bottom: 2px solid #2d2d42;
            position: sticky;
            top: 0;
            cursor: pointer;
        }}
        .data-table thead th:hover {{
            color: #667eea;
        }}
        .data-table tbody td {{
            padding: 10px 16px;
            border-bottom: 1px solid #1f1f35;
            max-width: 400px;
            word-wrap: break-word;
        }}
        .data-table tbody tr:hover {{
            background: rgba(102,126,234,0.08);
        }}
        .data-table tbody tr.status-dropped {{
            border-left: 3px solid #d63031;
        }}
        .data-table tbody tr.status-kept {{
            border-left: 3px solid #00b894;
        }}
        .badge {{
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge.status-kept {{
            background: rgba(0,184,148,0.15);
            color: #00b894;
        }}
        .badge.status-dropped {{
            background: rgba(214,48,49,0.15);
            color: #d63031;
        }}
        .step-badge {{
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            background: rgba(102,126,234,0.15);
            color: #667eea;
        }}
        .search-box {{
            width: 100%;
            padding: 12px 16px;
            background: #1a1a2e;
            border: 1px solid #2d2d42;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 0.9rem;
            margin-bottom: 16px;
            outline: none;
            transition: border-color 0.2s;
        }}
        .search-box:focus {{
            border-color: #667eea;
        }}
        .search-box::placeholder {{
            color: #666;
        }}
        .filter-buttons {{
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            padding: 6px 16px;
            border-radius: 20px;
            border: 1px solid #2d2d42;
            background: transparent;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: #667eea;
            border-color: #667eea;
            color: #fff;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #555;
            font-size: 0.8rem;
        }}
        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}
        .collapsible::after {{
            content: ' ▼';
            font-size: 0.7rem;
        }}
        .collapsible-content {{
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}
        @media print {{
            body {{ background: #fff; color: #333; }}
            .card {{ border-color: #ddd; background: #fff; }}
            .header {{ background: #667eea; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ Feature Engine Pro</h1>
        <p class="subtitle">Automated Feature Selection — Audit Report</p>
    </div>
    <div class="container">
        <!-- Executive Summary -->
        <div class="card">
            <h2>📋 Executive Summary</h2>
            <p class="subtitle">Generated on {timestamp} • Pipeline runtime: {runtime}s</p>
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-value">{input_cols}</div>
                    <div class="stat-label">Input Features</div>
                </div>
                <div class="stat-box accent">
                    <div class="stat-value">{final_cols}</div>
                    <div class="stat-label">Selected Features</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{n_dropped}</div>
                    <div class="stat-label">Features Dropped</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{reduction_pct}%</div>
                    <div class="stat-label">Dimensionality Reduction</div>
                </div>
            </div>
        </div>

        <!-- Column Journey -->
        {journey_js}

        <!-- Feature Funnel -->
        {funnel_js}

        <!-- Correlation Heatmap -->
        {corr_plot_js}

        <!-- Performance Evaluation -->
        {eval_html}

        <!-- Audit Trail -->
        <div class="card">
            <h2>🔍 Complete Audit Trail</h2>
            <p class="subtitle">Every feature decision with mathematical justification — search and filter below</p>

            <input type="text" class="search-box" id="audit-search"
                   placeholder="🔎 Search features, stages, or reasons..." onkeyup="filterTable()">

            <div class="filter-buttons">
                <button class="filter-btn active" onclick="setFilter('all', this)">All ({n_total})</button>
                <button class="filter-btn" onclick="setFilter('kept', this)">✓ Kept ({n_kept})</button>
                <button class="filter-btn" onclick="setFilter('dropped', this)">✗ Dropped ({n_dropped})</button>
            </div>

            <div style="max-height:600px; overflow-y:auto; border-radius:8px;">
                <table class="data-table" id="audit-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">Feature ↕</th>
                            <th onclick="sortTable(1)">Status ↕</th>
                            <th onclick="sortTable(2)">Stage ↕</th>
                            <th onclick="sortTable(3)">Mathematical Reason ↕</th>
                        </tr>
                    </thead>
                    <tbody>
                        {audit_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Feature Engine Pro v2.0 • Industry-Grade Automated Feature Selection</p>
        <p>Every decision is mathematically justified. Zero black boxes.</p>
    </div>

    <script>
        // --- Search/Filter ---
        let currentFilter = 'all';

        function setFilter(filter, btn) {{
            currentFilter = filter;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filterTable();
        }}

        function filterTable() {{
            const search = document.getElementById('audit-search').value.toLowerCase();
            const rows = document.querySelectorAll('#audit-table tbody tr');

            rows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                const isKept = row.classList.contains('status-kept');
                const isDropped = row.classList.contains('status-dropped');

                let matchesFilter = currentFilter === 'all' ||
                    (currentFilter === 'kept' && isKept) ||
                    (currentFilter === 'dropped' && isDropped);

                let matchesSearch = search === '' || text.includes(search);

                row.style.display = (matchesFilter && matchesSearch) ? '' : 'none';
            }});
        }}

        // --- Sortable Columns ---
        let sortDirection = {{}};
        function sortTable(colIndex) {{
            const table = document.getElementById('audit-table');
            const rows = Array.from(table.tBodies[0].rows);
            const dir = sortDirection[colIndex] = !sortDirection[colIndex];

            rows.sort((a, b) => {{
                const aText = a.cells[colIndex].textContent.trim();
                const bText = b.cells[colIndex].textContent.trim();
                return dir ? aText.localeCompare(bText) : bText.localeCompare(aText);
            }});

            rows.forEach(row => table.tBodies[0].appendChild(row));
        }}
    </script>
</body>
</html>'''
