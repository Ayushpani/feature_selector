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
import time
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
        """Generate a high-quality PDF report using Playwright, auto-installing browsers if needed.
        
        Automatically detects Colab/Jupyter (running asyncio loop) and uses the
        Playwright Async API in that environment. Falls back to the Sync API for
        normal Python scripts.
        """
        import os
        import subprocess
        import sys
        import asyncio

        self.generate_html_report(html_filepath)
        abs_path = os.path.abspath(html_filepath)
        file_url = f"file:///{abs_path.replace(os.sep, '/')}"

        self._logger.info("[Reporter] Preparing PDF generation...")

        # Detect if we're inside a running asyncio loop (Colab / Jupyter).
        _in_async_env = False
        try:
            # Playwright checks both running_loop and event_loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            if loop.is_running():
                _in_async_env = True
        except Exception:
            pass

        if _in_async_env:
            self._generate_pdf_async(file_url, pdf_filepath)
        else:
            self._generate_pdf_sync(file_url, pdf_filepath)

    # ------------------------------------------------------------------
    # Sync path — used in normal Python scripts
    # ------------------------------------------------------------------
    def _generate_pdf_sync(self, file_url, pdf_filepath):
        """Generate PDF using Playwright Sync API (for normal scripts)."""
        import time
        import subprocess
        import sys

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._logger.error("[Reporter] Playwright is not installed. Run: pip install playwright && playwright install chromium")
            return

        with sync_playwright() as p:
            browser = self._launch_browser_sync(p)
            if browser is None:
                return

            page = browser.new_page()
            self._logger.info("[Reporter] Generating PDF (this may take a few seconds)...")
            page.goto(file_url, wait_until='networkidle')
            time.sleep(2)
            page.pdf(
                path=pdf_filepath,
                format="A4",
                print_background=True,
                margin={'top': '20mm', 'bottom': '20mm', 'left': '10mm', 'right': '10mm'}
            )
            browser.close()

        self._logger.info(f"[Reporter] High-quality PDF generated: {pdf_filepath}")

    def _launch_browser_sync(self, playwright_ctx):
        """Try to launch Chromium; auto-install if missing."""
        import subprocess, sys
        try:
            return playwright_ctx.chromium.launch(headless=True)
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e).lower():
                self._logger.info("[Reporter] Chromium not found. Installing (one-time)...")
                try:
                    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
                    return playwright_ctx.chromium.launch(headless=True)
                except Exception as ie:
                    self._logger.error(f"[Reporter] Auto-install failed: {ie}")
                    self._logger.error("Please run 'playwright install chromium' manually.")
                    return None
            else:
                self._logger.error(f"[Reporter] Failed to launch browser: {e}")
                return None

    # ------------------------------------------------------------------
    # Async path — used inside Colab / Jupyter (running event loop)
    # ------------------------------------------------------------------
    def _generate_pdf_async(self, file_url, pdf_filepath):
        """Generate PDF using Playwright Async API (for Colab/Jupyter)."""
        import asyncio

        async def _run():
            import subprocess, sys
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                self._logger.error("[Reporter] Playwright is not installed. Run: pip install playwright && playwright install chromium")
                return

            async with async_playwright() as p:
                browser = await self._launch_browser_async(p)
                if browser is None:
                    return

                page = await browser.new_page()
                self._logger.info("[Reporter] Generating PDF (this may take a few seconds)...")
                await page.goto(file_url, wait_until='networkidle')
                await page.wait_for_timeout(2000)
                await page.pdf(
                    path=pdf_filepath,
                    format="A4",
                    print_background=True,
                    margin={'top': '20mm', 'bottom': '20mm', 'left': '10mm', 'right': '10mm'}
                )
                await browser.close()

            self._logger.info(f"[Reporter] High-quality PDF generated: {pdf_filepath}")

        # In Colab/Jupyter the loop is already running, so we can use
        # nest_asyncio to allow nested run(), or schedule via ensure_future.
        import asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.get_event_loop().run_until_complete(_run())
        except ImportError:
            # nest_asyncio not available — try ensure_future + manual blocking
            import concurrent.futures
            future = asyncio.ensure_future(_run())
            loop = asyncio.get_event_loop()
            loop.run_until_complete(future)

    async def _launch_browser_async(self, playwright_ctx):
        """Try to launch Chromium async; auto-install if missing."""
        import subprocess, sys
        try:
            return await playwright_ctx.chromium.launch(headless=True)
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e).lower():
                self._logger.info("[Reporter] Chromium not found. Installing (one-time)...")
                try:
                    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
                    return await playwright_ctx.chromium.launch(headless=True)
                except Exception as ie:
                    self._logger.error(f"[Reporter] Auto-install failed: {ie}")
                    self._logger.error("Please run 'playwright install chromium' manually.")
                    return None
            else:
                self._logger.error(f"[Reporter] Failed to launch browser: {e}")
                return None

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
                    paper_bgcolor: 'rgba(255,255,255,1)',
                    plot_bgcolor: 'rgba(255,255,255,1)',
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
                    paper_bgcolor: 'rgba(255,255,255,1)',
                    plot_bgcolor: 'rgba(255,255,255,1)',
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
                    line: {{color: '#1e6091', width: 3}},
                    marker: {{size: 10, color: '#184e77'}},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(30,96,145,0.1)',
                    hovertemplate: '%{{x}}<br>Features: %{{y}}<extra></extra>'
                }}];
                var journeyLayout = {{
                    margin: {{t: 30, l: 50, r: 30, b: 80}},
                    paper_bgcolor: 'rgba(255,255,255,1)',
                    plot_bgcolor: 'rgba(255,255,255,1)',
                    font: {{color: '#333333', family: 'Inter, sans-serif'}},
                    xaxis: {{tickangle: -25}},
                    yaxis: {{title: 'Feature Count', gridcolor: '#e9ecef'}},
                }};
                Plotly.newPlot('journey-chart', journeyData, journeyLayout, {{responsive: true}});
            </script>'''

        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Engine Pro - Audit Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f8f9fa;
            color: #333333;
            line-height: 1.6;
            padding: 0;
        }
        .header {
            background: #ffffff;
            padding: 48px 40px;
            text-align: center;
            border-bottom: 1px solid #e9ecef;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #1a202c;
        }
        .header .subtitle {
            font-size: 1.1rem;
            opacity: 0.8;
            margin-top: 10px;
            font-weight: 500;
            color: #4a5568;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .stat-row {
            display: flex;
            gap: 20px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        .stat-box {
            flex: 1;
            min-width: 150px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
        }
        .stat-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.06);
            border-color: #3182ce;
        }
        .stat-box.accent {
            border-top: 4px solid #3182ce;
            background: #ebf8ff;
        }
        .stat-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: #2b6cb0;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #4a5568;
            margin-top: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 32px;
            margin: 24px 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        }
        .card h2 {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 8px;
            color: #1a202c;
            border-bottom: 2px solid #edf2f7;
            padding-bottom: 12px;
        }
        .card .subtitle {
            font-size: 0.95rem;
            color: #718096;
            margin-bottom: 24px;
        }
        .data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.95rem;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }
        .data-table thead th {
            background: #f7fafc;
            padding: 16px 20px;
            text-align: left;
            font-weight: 600;
            color: #4a5568;
            border-bottom: 2px solid #e2e8f0;
            position: sticky;
            top: 0;
            cursor: pointer;
        }
        .data-table thead th:hover {
            background: #edf2f7;
            color: #2b6cb0;
        }
        .data-table tbody td {
            padding: 14px 20px;
            border-bottom: 1px solid #edf2f7;
            color: #2d3748;
            max-width: 400px;
            word-wrap: break-word;
        }
        .data-table tbody tr:hover {
            background: #f7fafc;
        }
        .data-table tbody tr.status-dropped {
            border-left: 4px solid #e53e3e;
        }
        .data-table tbody tr.status-kept {
            border-left: 4px solid #38a169;
        }
        .badge {
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge.status-kept {
            background: #f0fff4;
            color: #2f855a;
            border: 1px solid #c6f6d5;
        }
        .badge.status-dropped {
            background: #fff5f5;
            color: #c53030;
            border: 1px solid #fed7d7;
        }
        .step-badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            background: #edf2f7;
            color: #4a5568;
            border: 1px solid #e2e8f0;
        }
        .search-box {
            width: 100%;
            padding: 16px 20px;
            background: #ffffff;
            border: 1px solid #cbd5e0;
            border-radius: 8px;
            color: #2d3748;
            font-size: 1rem;
            margin-bottom: 20px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .search-box:focus {
            border-color: #3182ce;
            box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.15);
        }
        .search-box::placeholder {
            color: #a0aec0;
        }
        .filter-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 10px 20px;
            background: #ffffff;
            border: 1px solid #cbd5e0;
            border-radius: 8px;
            color: #4a5568;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        .filter-btn:hover, .filter-btn.active {
            background: #3182ce;
            border-color: #3182ce;
            color: #ffffff;
            box-shadow: 0 2px 4px rgba(49, 130, 206, 0.2);
        }
        .footer {
            text-align: center;
            padding: 40px;
            color: #718096;
            font-size: 0.9rem;
            border-top: 1px solid #e2e8f0;
            margin-top: 48px;
        }
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        .collapsible::after {
            content: ' \u25BC';
            font-size: 0.7rem;
            color: #a0aec0;
        }
        .collapsible-content {
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        @media print {
            body { background: #fff; color: #000; }
            .card { border: 1px solid #ddd; box-shadow: none; break-inside: avoid; }
            .header { background: #f8f9fa; border-bottom: 2px solid #333; }
        }
    </style>
</head>
<body>

    <div class="header">
        <h1>Feature Engine Pro</h1>
        <p class="subtitle">Automated Feature Selection - Audit Report</p>
    </div>

    <div class="container">
        
        <div class="card" style="padding: 0; background: transparent; border: none; box-shadow: none;">
            <h2 style="border: none; margin-bottom: 0;">Executive Summary</h2>
            <p class="subtitle">Generated on ''' + f'{timestamp}' + ''' • Pipeline runtime: ''' + f'{runtime}' + '''s</p>
            
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-value">''' + f'{n_total}' + '''</div>
                    <div class="stat-label">Input Features</div>
                </div>
                <div class="stat-box accent">
                    <div class="stat-value">''' + f'{n_kept}' + '''</div>
                    <div class="stat-label">Selected Features</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">''' + f'{n_dropped}' + '''</div>
                    <div class="stat-label">Features Dropped</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">''' + f'{reduction_pct:.1f}' + '''%</div>
                    <div class="stat-label">Dimensionality Reduction</div>
                </div>
            </div>
        </div>

        ''' + f'{journey_js}' + '''

        ''' + f'{eval_html}' + '''

        ''' + f'{corr_plot_js}' + '''

        <div class="card">
            <h2>The Audit Trail</h2>
            <p class="subtitle">Complete history of feature transformations and selection reasoning</p>
            
            <input type="text" class="search-box" id="searchInput" placeholder="Search features, steps, or reasons...">
            
            <div class="filter-buttons">
                <button class="filter-btn active" onclick="filterTable('all')">All Features (''' + f'{n_total}' + ''')</button>
                <button class="filter-btn" onclick="filterTable('kept')">Kept (''' + f'{n_kept}' + ''')</button>
                <button class="filter-btn" onclick="filterTable('dropped')">Dropped (''' + f'{n_dropped}' + ''')</button>
            </div>

            <table class="data-table" id="auditTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">Feature</th>
                        <th onclick="sortTable(1)">Final Status</th>
                        <th onclick="sortTable(2)">Pipeline Step</th>
                        <th onclick="sortTable(3)">Action & Reasoning</th>
                    </tr>
                </thead>
                <tbody>
                    ''' + f'{audit_rows}' + '''
                </tbody>
            </table>
        </div>

        <div class="footer">
            Feature Engine Pro v2.0 • Corporate Audit Report
        </div>
    </div>

    <script>
        // Search functionality
        document.getElementById('searchInput').addEventListener('keyup', function() {
            let filter = this.value.toUpperCase();
            let rows = document.getElementById("auditTable").getElementsByTagName("tr");
            
            for (let i = 1; i < rows.length; i++) {
                let text = rows[i].textContent || rows[i].innerText;
                if (text.toUpperCase().indexOf(filter) > -1) {
                    let currentFilter = document.querySelector('.filter-btn.active').innerText.toLowerCase();
                    if (currentFilter.includes('all') || rows[i].className.includes(currentFilter.split(' ')[0])) {
                        rows[i].style.display = "";
                    }
                } else {
                    rows[i].style.display = "none";
                }
            }
        });

        // Status Filtering
        function filterTable(status) {
            let btns = document.getElementsByClassName('filter-btn');
            for (let b of btns) b.classList.remove('active');
            event.target.classList.add('active');

            let rows = document.getElementById("auditTable").getElementsByTagName("tr");
            let searchFilter = document.getElementById('searchInput').value.toUpperCase();

            for (let i = 1; i < rows.length; i++) {
                let text = rows[i].textContent || rows[i].innerText;
                let matchesSearch = text.toUpperCase().indexOf(searchFilter) > -1;
                
                if (status === 'all') {
                    rows[i].style.display = matchesSearch ? "" : "none";
                } else {
                    if (rows[i].className.indexOf(status) > -1 && matchesSearch) {
                        rows[i].style.display = "";
                    } else {
                        rows[i].style.display = "none";
                    }
                }
            }
        }

        // Basic Sorting
        function sortTable(n) {
            let table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
            table = document.getElementById("auditTable");
            switching = true;
            dir = "asc"; 
            while (switching) {
                switching = false;
                rows = table.rows;
                for (i = 1; i < (rows.length - 1); i++) {
                    shouldSwitch = false;
                    x = rows[i].getElementsByTagName("TD")[n];
                    y = rows[i + 1].getElementsByTagName("TD")[n];
                    if (dir == "asc") {
                        if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
                            shouldSwitch = true;
                            break;
                        }
                    } else if (dir == "desc") {
                        if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {
                            shouldSwitch = true;
                            break;
                        }
                    }
                }
                if (shouldSwitch) {
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount ++; 
                } else {
                    if (switchcount == 0 && dir == "asc") {
                        dir = "desc";
                        switching = true;
                    }
                }
            }
        }
    </script>
</body>
</html>'''