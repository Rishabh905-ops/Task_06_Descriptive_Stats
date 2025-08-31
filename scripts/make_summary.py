import pandas as pd, json
from pathlib import Path
import argparse

def pick_col(df, options):
    for opt in options:
        if opt in df.columns:
            return opt
    return None

def main():
    ap = argparse.ArgumentParser(description="Summarize athletics CSV to JSON + truth table.")
    ap.add_argument("--csv", required=True, help="Path to your CSV (do NOT place inside repo).")
    ap.add_argument("--outdir", default="results", help="Where to write outputs.")
    args = ap.parse_args()

    data_path = Path(args.csv)
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    # Read and normalize columns
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df = df.rename(columns=lambda c: c.strip().lower())

    # Possible synonyms
    player_col  = pick_col(df, ["player","athlete","name","player_name"])
    game_col    = pick_col(df, ["gameid","game","match_id","match","game_no"])
    date_col    = pick_col(df, ["date","game_date","played_on"])
    goals_col   = pick_col(df, ["goals","goal"])
    assists_col = pick_col(df, ["assists","assist"])
    points_col  = pick_col(df, ["points","pts","total points"])

    # Coerce numerics
    for c in [goals_col, assists_col, points_col]:
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    n_rows, n_cols = df.shape
    metrics = {}
    if goals_col:   metrics["total_goals"] = int(df[goals_col].sum())
    if assists_col: metrics["total_assists"] = int(df[assists_col].sum())
    if points_col:  metrics["total_points"] = float(df[points_col].sum())

    # Games played
    games_played = None
    if game_col:
        games_played = int(df[game_col].nunique())
    elif date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
        games_played = int(dates.nunique())
        metrics["unique_dates"] = games_played

    # Top performers (points preferred, else goals)
    top_table = None
    if player_col:
        if points_col:
            top_table = (df.groupby(player_col, dropna=False)[points_col]
                           .sum().sort_values(ascending=False).head(10)
                           .reset_index()
                           .rename(columns={player_col:"Player", points_col:"TotalPoints"}))
        elif goals_col:
            top_table = (df.groupby(player_col, dropna=False)[goals_col]
                           .sum().sort_values(ascending=False).head(10)
                           .reset_index()
                           .rename(columns={player_col:"Player", goals_col:"TotalGoals"}))

    summary = {
        "shape": {"rows": n_rows, "columns": n_cols},
        "games_played": games_played,
        "metrics": metrics,
        "columns_detected": list(df.columns),
        "column_mapping": {
            "player": player_col, "game_id": game_col, "date": date_col,
            "goals": goals_col, "assists": assists_col, "points": points_col
        }
    }
    if top_table is not None:
        summary["top_performers"] = top_table.to_dict(orient="records")

    # Write files
    out_json = Path(out_dir) / "summary_for_llm.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    truth = top_table if top_table is not None else pd.DataFrame()
    truth_path = Path(out_dir) / "truth_table.csv"
    truth.to_csv(truth_path, index=False, encoding="utf-8")

    print(f"Wrote:\n  {out_json}\n  {truth_path}")

if __name__ == "__main__":
    main()
