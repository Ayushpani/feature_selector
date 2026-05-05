import re

with open('feature_engine_pro/reporter.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace HTML emojis
content = content.replace('<h2>📊 Performance Impact</h2>', '<h2>Performance Impact</h2>')
content = content.replace('<h2>📉 Attrition Funnel</h2>', '<h2>Attrition Funnel</h2>')
content = content.replace('<h2>🗺️ Column Journey</h2>', '<h2>Column Journey</h2>')
content = content.replace('<h2>🌡️ Correlation Heatmap (Pre-Filter)</h2>', '<h2>Correlation Heatmap (Pre-Filter)</h2>')
content = content.replace('<h2>🔍 Detailed Audit Trail</h2>', '<h2>Detailed Audit Trail</h2>')

# Change CSS
css_old = r'''        body \{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            line-height: 1\.6;
            padding: 0;
        \}
        \.header \{
            background: linear-gradient\(135deg, #667eea 0%, #764ba2 100%\);
            padding: 48px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        \}
        \.header::before \{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient\(circle, rgba\(255,255,255,0\.05\) 0%, transparent 60%\);
            animation: pulse 8s ease-in-out infinite;
        \}
        @keyframes pulse \{
            0%, 100% \{ transform: scale\(1\); opacity: 0\.5; \}
            50% \{ transform: scale\(1\.2\); opacity: 0\.8; \}
        \}
        \.header h1 \{
            font-size: 2\.2rem;
            font-weight: 800;
            letter-spacing: -0\.5px;
            position: relative;
            color: #fff;
        \}
        \.header \.subtitle \{
            font-size: 1rem;
            opacity: 0\.85;
            margin-top: 8px;
            font-weight: 400;
            position: relative;
            color: #f0f0f0;
        \}
        \.container \{
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        \}
        \.stat-row \{
            display: flex;
            gap: 16px;
            margin: 24px 0;
            flex-wrap: wrap;
        \}
        \.stat-box \{
            flex: 1;
            min-width: 140px;
            background: #1a1a2e;
            border: 1px solid #2d2d42;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0\.2s, border-color 0\.2s;
        \}
        \.stat-box:hover \{
            transform: translateY\(-2px\);
            border-color: #667eea;
        \}
        \.stat-box\.accent \{
            background: linear-gradient\(135deg, rgba\(102,126,234,0\.15\), rgba\(118,75,162,0\.15\)\);
            border-color: #667eea;
        \}
        \.stat-value \{
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient\(135deg, #667eea, #764ba2\);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        \}
        \.stat-label \{
            font-size: 0\.85rem;
            color: #888;
            margin-top: 4px;
            font-weight: 500;
        \}
        \.card \{
            background: #16162a;
            border: 1px solid #2d2d42;
            border-radius: 16px;
            padding: 28px;
            margin: 20px 0;
            transition: border-color 0\.2s;
        \}
        \.card:hover \{
            border-color: #3d3d5c;
        \}
        \.card h2 \{
            font-size: 1\.3rem;
            font-weight: 700;
            margin-bottom: 6px;
            color: #f0f0f0;
        \}
        \.card \.subtitle \{
            font-size: 0\.85rem;
            color: #888;
            margin-bottom: 20px;
        \}
        \.data-table \{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0\.85rem;
            border-radius: 8px;
            overflow: hidden;
        \}
        \.data-table thead th \{
            background: #1a1a2e;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            color: #b0b0b0;
            border-bottom: 2px solid #2d2d42;
            position: sticky;
            top: 0;
            cursor: pointer;
        \}
        \.data-table thead th:hover \{
            color: #667eea;
        \}
        \.data-table tbody td \{
            padding: 10px 16px;
            border-bottom: 1px solid #1f1f35;
            max-width: 400px;
            word-wrap: break-word;
        \}
        \.data-table tbody tr:hover \{
            background: rgba\(102,126,234,0\.08\);
        \}
        \.data-table tbody tr\.status-dropped \{
            border-left: 3px solid #d63031;
        \}
        \.data-table tbody tr\.status-kept \{
            border-left: 3px solid #00b894;
        \}
        \.badge \{
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0\.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0\.5px;
        \}
        \.badge\.status-kept \{
            background: rgba\(0,184,148,0\.15\);
            color: #00b894;
        \}
        \.badge\.status-dropped \{
            background: rgba\(214,48,49,0\.15\);
            color: #d63031;
        \}
        \.step-badge \{
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0\.75rem;
            font-weight: 500;
            background: rgba\(102,126,234,0\.15\);
            color: #667eea;
        \}
        \.search-box \{
            width: 100%;
            padding: 12px 16px;
            background: #1a1a2e;
            border: 1px solid #2d2d42;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 0\.9rem;
            margin-bottom: 16px;
            outline: none;
            transition: border-color 0\.2s;
        \}
        \.search-box:focus \{
            border-color: #667eea;
        \}
        \.search-box::placeholder \{
            color: #666;
        \}
        \.filter-buttons \{
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        \}
        \.filter-btn \{
            padding: 6px 16px;
            border-radius: 20px;
            border: 1px solid #2d2d42;
            background: transparent;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 0\.8rem;
            font-weight: 500;
            transition: all 0\.2s;
        \}
        \.filter-btn:hover, \.filter-btn\.active \{
            background: #667eea;
            border-color: #667eea;
            color: #fff;
        \}'''

css_new = '''        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f8f9fa;
            color: #333333;
            line-height: 1.6;
            padding: 0;
        }
        .header {
            background: #ffffff;
            border-bottom: 1px solid #e9ecef;
            padding: 40px 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #003366;
        }
        .header .subtitle {
            font-size: 1rem;
            color: #6c757d;
            margin-top: 8px;
            font-weight: 400;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .stat-row {
            display: flex;
            gap: 16px;
            margin: 24px 0;
            flex-wrap: wrap;
        }
        .stat-box {
            flex: 1;
            min-width: 140px;
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        .stat-box.accent {
            background: #f0f7ff;
            border-color: #cce5ff;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #003366;
        }
        .stat-label {
            font-size: 0.85rem;
            color: #6c757d;
            margin-top: 4px;
            font-weight: 500;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 28px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .card h2 {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 6px;
            color: #003366;
        }
        .card .subtitle {
            font-size: 0.85rem;
            color: #6c757d;
            margin-bottom: 20px;
        }
        .data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.85rem;
            border-radius: 8px;
            overflow: hidden;
        }
        .data-table thead th {
            background: #f8f9fa;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
            position: sticky;
            top: 0;
            cursor: pointer;
        }
        .data-table thead th:hover {
            color: #003366;
        }
        .data-table tbody td {
            padding: 10px 16px;
            border-bottom: 1px solid #e9ecef;
            max-width: 400px;
            word-wrap: break-word;
        }
        .data-table tbody tr:hover {
            background: #f8f9fa;
        }
        .data-table tbody tr.status-dropped {
            border-left: 3px solid #d93025;
        }
        .data-table tbody tr.status-kept {
            border-left: 3px solid #1e8e3e;
        }
        .badge {
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge.status-kept {
            background: #e6f4ea;
            color: #1e8e3e;
        }
        .badge.status-dropped {
            background: #fce8e6;
            color: #d93025;
        }
        .step-badge {
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            background: #e8f0fe;
            color: #1a73e8;
        }
        .search-box {
            width: 100%;
            padding: 12px 16px;
            background: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 8px;
            color: #495057;
            font-size: 0.9rem;
            margin-bottom: 16px;
            outline: none;
            transition: border-color 0.2s;
        }
        .search-box:focus {
            border-color: #003366;
        }
        .search-box::placeholder {
            color: #6c757d;
        }
        .filter-buttons {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 6px 16px;
            border-radius: 20px;
            border: 1px solid #ced4da;
            background: #ffffff;
            color: #495057;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .filter-btn:hover, .filter-btn.active {
            background: #003366;
            border-color: #003366;
            color: #ffffff;
        }'''

import re
content = re.sub(css_old, css_new, content, flags=re.DOTALL)

# Update Plotly charts theme
content = content.replace("marker: {color: '#00b894'}", "marker: {color: '#1e8e3e'}")
content = content.replace("marker: {color: '#d63031'}", "marker: {color: '#d93025'}")
content = content.replace("font: {color: '#e0e0e0'", "font: {color: '#333333'")
content = content.replace("gridcolor: '#2d2d42'", "gridcolor: '#e9ecef'")
content = content.replace("line: {color: '#667eea', width: 3}", "line: {color: '#003366', width: 3}")
content = content.replace("marker: {size: 10, color: '#764ba2'}", "marker: {size: 10, color: '#003366'}")
content = content.replace("fillcolor: 'rgba(102,126,234,0.15)'", "fillcolor: 'rgba(0,51,102,0.1)'")
content = content.replace("colorscale: 'RdBu'", "colorscale: 'Blues'")

# Remove more emojis from the evaluation output
content = content.replace("↑", "+").replace("↓", "-")

# Add the generate_pdf_report function
pdf_func = """    def generate_pdf_report(self, html_filepath="feature_engine_report.html", pdf_filepath="feature_engine_report.pdf"):
        \"\"\"Generate a high-quality PDF report using Playwright.\"\"\"
        import os
        import time
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._logger.error("[Reporter] Playwright is not installed. To generate PDFs, run: pip install playwright && playwright install chromium")
            return

        self.generate_html_report(html_filepath)
        abs_path = os.path.abspath(html_filepath)
        file_url = f"file:///{abs_path.replace('\\\\', '/')}"

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
            
        self._logger.info(f"[Reporter] High-quality PDF generated: {pdf_filepath}")"""

content = content.replace('def print_report(self):', pdf_func + '\n\n    def print_report(self):')

with open('feature_engine_pro/reporter.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated reporter.py")
