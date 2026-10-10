from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'


def train_model(events_csv=None, model_out=None):
    """Train the appliance-classification model and return summary metrics."""
    events_path = Path(events_csv) if events_csv else DATA_DIR / 'events.csv'
    model_out = Path(model_out) if model_out else DATA_DIR / 'model_summary.json'

    if not events_path.is_absolute():
        events_path = ROOT / events_path
    if not model_out.is_absolute():
        model_out = ROOT / model_out

    if not events_path.exists():
        raise FileNotFoundError(f'Events file not found: {events_path}')

    df = pd.read_csv(events_path)
    df_on = df[df['event_type'] == 'ON'].copy()

    X = df_on[['delta_p', 'delta_q', 'inrush_ratio', 'ramp_speed', 'hour_of_day']]
    y = df_on['appliance_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    summary = {
        'accuracy': float(report.get('accuracy', 0.0)),
        'report': report,
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    display_path = model_out.relative_to(ROOT) if model_out.is_relative_to(ROOT) else model_out
    print(f'Model summary saved to {display_path}')
    print(json.dumps(summary, indent=2))
    return {'model': model, 'report': report, 'accuracy': summary['accuracy']}


if __name__ == '__main__':
    train_model()
