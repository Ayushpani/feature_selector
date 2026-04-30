import pandas as pd

class Reporter:
    """
    Maintains an audit trail of the feature selection pipeline.
    Logs which features were dropped, kept, and the exact mathematical reason why.
    """
    def __init__(self):
        self.logs = []
        self.feature_status = {}  # {feature_name: {'status': 'kept'/'dropped', 'reason': '...'}}

    def log_event(self, feature, status, reason, step_name):
        """
        Logs a specific event for a feature.
        :param feature: str, feature name
        :param status: str, 'kept' or 'dropped'
        :param reason: str, explanation for the status
        :param step_name: str, which stage of the pipeline made this decision
        """
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

    def generate_summary(self):
        """
        Returns a DataFrame summarizing the audit trail for all features.
        """
        return pd.DataFrame(self.logs)

    def print_report(self):
        """
        Prints a formatted report to the console.
        """
        print("="*50)
        print(" FEATURE ENGINE PRO - AUDIT REPORT ")
        print("="*50)
        df_logs = self.generate_summary()
        if df_logs.empty:
            print("No features have been processed yet.")
            return

        dropped = df_logs[df_logs['status'] == 'dropped']
        kept = df_logs[df_logs['status'] == 'kept']

        print(f"\nTotal Features Kept: {len(kept)}")
        print(f"Total Features Dropped: {len(dropped)}")

        print("\n--- DROPPED FEATURES ---")
        for _, row in dropped.iterrows():
            print(f"[{row['step']}] {row['feature']} -> {row['reason']}")

        print("\n--- KEPT FEATURES ---")
        for _, row in kept.iterrows():
            print(f"[{row['step']}] {row['feature']} -> {row['reason']}")

        print("="*50)

    def generate_html_report(self, filepath="feature_report.html"):
        """
        Generates an HTML report of the audit trail.
        """
        df_logs = self.generate_summary()
        html_str = f"""
        <html>
        <head>
            <title>Feature Engine Pro Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .dropped {{ color: red; }}
                .kept {{ color: green; }}
            </style>
        </head>
        <body>
            <h2>Feature Engine Pro - Audit Report</h2>
            <p><strong>Total Kept:</strong> {len(df_logs[df_logs['status'] == 'kept'])}</p>
            <p><strong>Total Dropped:</strong> {len(df_logs[df_logs['status'] == 'dropped'])}</p>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Status</th>
                    <th>Step</th>
                    <th>Reason</th>
                </tr>
        """
        for _, row in df_logs.iterrows():
            status_class = "dropped" if row['status'] == 'dropped' else "kept"
            html_str += f"""
                <tr>
                    <td>{row['feature']}</td>
                    <td class="{status_class}"><strong>{row['status'].upper()}</strong></td>
                    <td>{row['step']}</td>
                    <td>{row['reason']}</td>
                </tr>
            """
        html_str += """
            </table>
        </body>
        </html>
        """
        with open(filepath, "w") as f:
            f.write(html_str)
        print(f"HTML report generated at {filepath}")
