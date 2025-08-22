FPS       = 30.0
dt        = 1 / FPS
SEC       = int(FPS)
JOINTS    = {"wr_L": 9,  "wr_R": 10,   # wrists
          "ank_L": 15, "ank_R": 16} # ankles
CONF_TH   = 0.10      # keep frames with score ≥ 0.10
MAX_GAP   = 4         # interpolate gaps up to four consecutive frames
SIGMA     = 2         # Gaussian σ (frames)
WIN       = 5         # effective window size (frames)

MAX_LAG_SEC = 3                 # look ±3 s around zero
MAX_LAG_FR  = int(MAX_LAG_SEC * FPS)

WINDOW_SEC = [1, 2, 3]                 # seconds
WINDOW_FR  = [int(FPS*s) for s in WINDOW_SEC]   # [30,60,90]

GAUSS_SIG = 2              # frames
GAUSS_WIN = 5              # approximate support (frames)

XY_COLS    = [f"{a}{i}" for i in range(17) for a in ("x", "y")]
SCORE_COLS = [f"score{i}" for i in range(17)]