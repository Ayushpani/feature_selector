import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

class Reporter:
    """
    Maintains an audit trail of the feature selection pipeline.
    Logs which features were dropped, kept, and generates an HTML report with visual plots.
    """
    def __init__(self):
        self.logs = []
        self.feature_status = {}
        # Stores visual plots
        self.correlation_matrix_before = None
        self.feature_importances = None

    def log_event(self, feature, status, reason, step_name):
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
        """Captures the correlation matrix plot before variables are dropped."""
        num_df = df.select_dtypes(include='number')
        if num_df.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(num_df.corr(), cmap='coolwarm', annot=False)
            plt.title("Correlation Matrix (Before Filtering)")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            self.correlation_matrix_before = base64.b64encode(buf.read()).decode('utf-8')

    def generate_summary(self):
        return pd.DataFrame(self.logs)

    def print_report(self):
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
        df_logs = self.generate_summary()

        # Summary statistics per step
        if not df_logs.empty:
            step_summary = df_logs.groupby(['step', 'status']).size().unstack(fill_value=0).reset_index()
            # Generate a bar chart of the funnel
            plt.figure(figsize=(8, 5))
            step_summary.plot(x='step', kind='bar', stacked=True, color=['red', 'green'])
            plt.title("Feature Filtering Funnel")
            plt.ylabel("Number of Features")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            funnel_chart = base64.b64encode(buf.read()).decode('utf-8')
        else:
            funnel_chart = None

        html_str = f"""
        <html>
        <head>
            <title>Feature Engine Pro Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }}
                .container {{ background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 14px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .dropped {{ color: #d9534f; font-weight: bold; }}
                .kept {{ color: #5cb85c; font-weight: bold; }}
                .plot-container {{ display: flex; justify-content: space-around; margin-top: 30px; margin-bottom: 30px; }}
                img {{ max-width: 45%; border: 1px solid #ddd; border-radius: 4px; padding: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>📊 Feature Engine Pro - Audit Report</h2>
                <p><strong>Total Features Kept:</strong> {len(df_logs[df_logs['status'] == 'kept']) if not df_logs.empty else 0}</p>
                <p><strong>Total Features Dropped:</strong> {len(df_logs[df_logs['status'] == 'dropped']) if not df_logs.empty else 0}</p>

                <div class="plot-container">
        """
        if funnel_chart:
            html_str += f"<img src='data:image/png;base64,{funnel_chart}' alt='Funnel Chart'>"
        if self.correlation_matrix_before:
            html_str += f"<img src='data:image/png;base64,{self.correlation_matrix_before}' alt='Correlation Matrix'>"

        html_str += """
                </div>

                <h3>Audit Trail</h3>
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
                        <td class="{status_class}">{row['status'].upper()}</td>
                        <td>{row['step']}</td>
                        <td>{row['reason']}</td>
                    </tr>
            """
        html_str += """
                </table>
            </div>
        </body>
        </html>
        """
        with open(filepath, "w") as f:
            f.write(html_str)
        print(f"HTML report generated at {filepath}")
