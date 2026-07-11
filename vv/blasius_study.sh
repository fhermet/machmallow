#!/usr/bin/env bash
# Blasius Cf grid-convergence + Mach study — the experiment behind the
# "source of the Cf bias" analysis in vv/README.md. Expensive (several
# minutes, incl. ny=1024 and low-Mach runs), so it is NOT part of the
# default `vv/generate.py`; run it on demand to regenerate the two data
# files vv/data/blasius_refinement.csv and vv/data/blasius_mach.csv.
#
#   cmake --build build --target blasius -j
#   bash vv/blasius_study.sh
#
# The blasius driver takes optional knobs: `blasius <ny> [nx] [p0]`
#   ny     : wall-normal cells
#   nx     : streamwise cells (default 1.25*ny -> square cells)
#   p0     : stagnation pressure -> lowers Mach at FIXED Re_x
# Cf is read at the mid-plate station x ~ 0.55 from out/blasius_cf.csv.
set -e
cd "$(dirname "$0")/.."
mid() { awk -F, 'NR>1 && $1>0.53 && $1<0.57 {printf "%.1f",100*($3/$4-1)}' \
        out/blasius_cf.csv | head -1; }

R=vv/data/blasius_refinement.csv
echo "study,ny,nx,dy,cells,mach,cf_pct" > "$R"
# isotropic (square cells): refine both directions
for ny in 128 256 512; do nx=$((5*ny/4)); ./build/blasius $ny $nx >/dev/null 2>&1
  printf "iso,%d,%d,%.5f,%d,0.254,%s\n" $ny $nx $(python3 -c "print(1.0/$ny)") \
         $((nx*ny)) "$(mid)" >> "$R"; done
# anisotropic: fine wall-normal only (nx fixed = 320)
for ny in 256 512 1024; do ./build/blasius $ny 320 >/dev/null 2>&1
  printf "aniso,%d,320,%.5f,%d,0.254,%s\n" $ny $(python3 -c "print(1.0/$ny)") \
         $((320*ny)) "$(mid)" >> "$R"; done

M=vv/data/blasius_mach.csv
echo "p0,mach,rex,cf_pct" > "$M"
# low-Mach at fixed Re_x / resolution (ny=256, nx=320), varying p0
for p0 in 1 4 16; do ./build/blasius 256 320 $p0 > /tmp/_blm.txt 2>&1
  mach=$(grep -o 'Mach = [0-9.]*' /tmp/_blm.txt | grep -o '[0-9.]*')
  rex=$(grep -o 'Re_x = [0-9]*' /tmp/_blm.txt | grep -o '[0-9]*')
  printf "%d,%s,%s,%s\n" $p0 "$mach" "$rex" "$(mid)" >> "$M"; done
echo "wrote $R and $M"
