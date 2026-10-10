from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'


def extract_events(csv_path=None, output_path=None):
    """Extract significant power-step events from the simulated smart meter data."""
    csv_path = Path(csv_path) if csv_path else DATA_DIR / 'sim_day.csv'
    output_path = Path(output_path) if output_path else DATA_DIR / 'events.csv'

    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if not csv_path.exists():
        raise FileNotFoundError(f'Simulation file not found: {csv_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    df['delta_p'] = df['total_p'].diff()
    df['delta_q'] = df['total_q'].diff()

    events = []
    threshold = 50.0
    n_rows = len(df)

    for i in range(1, n_rows - 15):
        d_p = df['delta_p'].iloc[i]
        if abs(d_p) > threshold:
            timestamp = df.index[i]
            d_q = df['delta_q'].iloc[i]
            hour = timestamp.hour

            window_3s = df['total_p'].iloc[i : i + 3]
            p_before = df['total_p'].iloc[i - 1]
            p_peak = window_3s.max() - p_before
            inrush_ratio = p_peak / d_p if d_p > 0 else 1.0

            p_15s_after = df['total_p'].iloc[i + 15] - p_before
            ramp_speed = (p_15s_after - d_p) / 15.0 if d_p > 0 else 0.0

            true_label = 'unknown'
            for app in ['fridge', 'ac', 'microwave', 'kettle']:
                app_diff = df[f'{app}_p'].diff().iloc[i]
                if abs(app_diff) > 30:
                    true_label = app
                    break

            events.append(
                {
                    'timestamp': timestamp,
                    'event_type': 'ON' if d_p > 0 else 'OFF',
                    'delta_p': round(d_p, 2),
                    'delta_q': round(d_q, 2),
                    'inrush_ratio': round(max(1.0, inrush_ratio), 2),
                    'ramp_speed': round(ramp_speed, 2),
                    'hour_of_day': hour,
                    'appliance_label': true_label,
                }
            )

    events_df = pd.DataFrame(events)
    events_df.to_csv(output_path, index=False)

    display_path = output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path
    print(f'Extracted {len(events_df)} events and saved to {display_path}')
    return events_df


if __name__ == '__main__':
    extract_events()
