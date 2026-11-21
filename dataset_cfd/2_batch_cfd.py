"""Step 2: batch COMSOL volume meshing and CFD on successful step-1 surfaces."""
from src.batch_stages import stage_cli

if __name__=='__main__':raise SystemExit(stage_cli('cfd'))
