import sys

def fix_reporter():
    with open("feature_engine_pro/reporter.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # We want to replace everything from "        # Column journey chart" down to the end of the file.
    start_str = "        # Column journey chart"
    start_index = content.find(start_str)
    if start_index == -1:
        print("Could not find start string")
        return
        
    new_tail = """        # Column journey chart
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
            content: ' \\u25BC';
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
</html>'''"""

    new_content = content[:start_index] + new_tail
    with open("feature_engine_pro/reporter.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Done")

if __name__ == "__main__":
    fix_reporter()
