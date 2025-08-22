from constants import XY_COLS, CONF_TH, MAX_GAP, SCORE_COLS
import numpy as np

def interpolate_track(df_part):
    """
    df_part: one (participant, date, camera) block sorted by frame_id.
    Replaces coords with NaN where score < CONF_TH,
    then linearly interpolates gaps ≤ MAX_GAP.
    Returns the same DataFrame with coords filled.
    """
    out = df_part.copy()

    # mask low-confidence samples
    low_conf = out[SCORE_COLS] < CONF_TH        # DataFrame of bools
    for xy in XY_COLS:
        i = int(xy[1:])                         # key-point index
        out.loc[low_conf[f"score{i}"], xy] = np.nan

    # interpolate inside small gaps only
    out[XY_COLS] = (
        out[XY_COLS]
          .interpolate(method="linear",
                       limit=MAX_GAP,
                       limit_area="inside",
                       axis=0)
    )
    return out

