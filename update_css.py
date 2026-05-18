import re

def update_reporter():
    with open("feature_engine_pro/reporter.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    start_index = content.find("return f'''<!DOCTYPE html>")
    end_index = content.find("</html>'''", start_index)
        
    if start_index == -1 or end_index == -1:
        print("Could not find HTML string bounds.")
        return
    
    end_index += len("</html>'''")

    new_html = """return '''<!DOCTYPE html>
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
            background: #f4f7f6;
            color: #333333;
            line-height: 1.6;
            padding: 0;
        }
        .header {
            background: #ffffff;
            padding: 48px 40px;
            text-align: center;
            border-bottom: 1px solid #e1e4e8;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #0d1b2a;
        }
        .header .subtitle {
            font-size: 1rem;
            opacity: 0.7;
            margin-top: 8px;
            font-weight: 500;
            color: #4a4e69;
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
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: box-shadow 0.2s, border-color 0.2s;
        }
        .stat-box:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-color: #1e6091;
        }
        .stat-box.accent {
            border-top: 4px solid #1e6091;
        }
        .stat-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e6091;
        }
        .stat-label {
            font-size: 0.85rem;
            color: #6c757d;
            margin-top: 4px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 28px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        .card h2 {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 6px;
            color: #0d1b2a;
            border-bottom: 2px solid #f1f3f5;
            padding-bottom: 8px;
        }
        .card .subtitle {
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 20px;
        }
        .data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.9rem;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #e1e4e8;
        }
        .data-table thead th {
            background: #f8f9fa;
            padding: 14px 16px;
            text-align: left;
            font-weight: 600;
            color: #495057;
            border-bottom: 1px solid #e1e4e8;
            position: sticky;
            top: 0;
            cursor: pointer;
        }
        .data-table thead th:hover {
            background: #e9ecef;
        }
        .data-table tbody td {
            padding: 12px 16px;
            border-bottom: 1px solid #f1f3f5;
            color: #343a40;
            max-width: 400px;
            word-wrap: break-word;
        }
        .data-table tbody tr:hover {
            background: #f8f9fa;
        }
        .data-table tbody tr.status-dropped {
            border-left: 4px solid #e63946;
        }
        .data-table tbody tr.status-kept {
            border-left: 4px solid #2a9d8f;
        }
        .badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge.status-kept {
            background: #e6fcf5;
            color: #2a9d8f;
            border: 1px solid #b2f2bb;
        }
        .badge.status-dropped {
            background: #fff5f5;
            color: #e63946;
            border: 1px solid #ffc9c9;
        }
        .step-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            background: #e9ecef;
            color: #495057;
            border: 1px solid #dee2e6;
        }
        .search-box {
            width: 100%;
            padding: 14px 16px;
            background: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 6px;
            color: #333;
            font-size: 0.95rem;
            margin-bottom: 16px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .search-box:focus {
            border-color: #1e6091;
            box-shadow: 0 0 0 3px rgba(30, 96, 145, 0.1);
        }
        .search-box::placeholder {
            color: #adb5bd;
        }
        .filter-buttons {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 8px 16px;
            background: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 6px;
            color: #495057;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        .filter-btn:hover, .filter-btn.active {
            background: #1e6091;
            border-color: #1e6091;
            color: #ffffff;
        }
        .footer {
            text-align: center;
            padding: 30px;
            color: #6c757d;
            font-size: 0.85rem;
            border-top: 1px solid #e1e4e8;
            margin-top: 40px;
        }
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        .collapsible::after {
            content: ' \\u25BC';
            font-size: 0.7rem;
            color: #adb5bd;
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
            <p class="subtitle">Generated on ''' + f'{time.strftime("%Y-%m-%d %H:%M:%S")}' + ''' • Pipeline runtime: ''' + f'{self.pipeline_runtime}' + '''s</p>
            
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
                    <div class="stat-value">''' + f'{reduction:.1f}' + '''%</div>
                    <div class="stat-label">Dimensionality Reduction</div>
                </div>
            </div>
        </div>

        ''' + f'{journey_js}' + '''

        ''' + f'{metrics_html}' + '''

        ''' + f'{correlation_html}' + '''

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
                    ''' + f'{tbody}' + '''
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
                    // Only show if it matches current status filter too
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
            // Update buttons
            let btns = document.getElementsByClassName('filter-btn');
            for (let b of btns) b.classList.remove('active');
            event.target.classList.add('active');

            // Filter rows
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

    # Also fix journey_chart colors
    content = content.replace("line: {color: '#667eea', width: 3},", "line: {color: '#1e6091', width: 3},")
    content = content.replace("marker: {size: 10, color: '#764ba2'},", "marker: {size: 10, color: '#184e77'},")
    content = content.replace("fillcolor: 'rgba(0,51,102,0.1)',", "fillcolor: 'rgba(30,96,145,0.1)',")
    content = content.replace("font: {color: '#e0e0e0', family: 'Inter, sans-serif'},", "font: {color: '#333333', family: 'Inter, sans-serif'},")
    content = content.replace("paper_bgcolor: 'rgba(0,0,0,0)',", "paper_bgcolor: 'rgba(255,255,255,1)',")
    content = content.replace("plot_bgcolor: 'rgba(0,0,0,0)',", "plot_bgcolor: 'rgba(255,255,255,1)',")
    
    # Also fix metrics_html
    content = content.replace("line: {color: '#d63031', width: 3}", "line: {color: '#e63946', width: 3}")
    content = content.replace("line: {color: '#00b894', width: 3}", "line: {color: '#2a9d8f', width: 3}")
    
    # Also fix correlation map colors
    content = content.replace("colorscale: 'RdBu',", "colorscale: 'Blues',")
    
    new_content = content[:start_index] + new_html + content[end_index:]
    with open("feature_engine_pro/reporter.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Done")

if __name__ == "__main__":
    update_reporter()
