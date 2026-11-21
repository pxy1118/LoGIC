"""Step 1: batch fluid-domain STL generation, without COMSOL volume meshing."""
from src.batch_stages import stage_cli

if __name__=='__main__':raise SystemExit(stage_cli('geometry'))
