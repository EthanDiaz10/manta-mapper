#!/bin/sh
VSP="${VSP:-$HOME/OpenVSP}"
cd "$(dirname "$0")"
"$VSP/vspscript" -script F1_vlm_sweep.vspscript > F1_vlm_sweep.log 2>&1
"$VSP/vspscript" -script F1_vlm_stab.vspscript > F1_vlm_stab.log 2>&1
"$VSP/vspscript" -script F1_vlm_beta.vspscript > F1_vlm_beta.log 2>&1
"$VSP/vspscript" -script F1_panel_sweep.vspscript > F1_panel_sweep.log 2>&1
"$VSP/vspscript" -script F1_panel_stab.vspscript > F1_panel_stab.log 2>&1
"$VSP/vspscript" -script F1_panel_beta.vspscript > F1_panel_beta.log 2>&1
"$VSP/vspscript" -script F2_vlm_sweep.vspscript > F2_vlm_sweep.log 2>&1
"$VSP/vspscript" -script F2_vlm_stab.vspscript > F2_vlm_stab.log 2>&1
"$VSP/vspscript" -script F2_vlm_beta.vspscript > F2_vlm_beta.log 2>&1
"$VSP/vspscript" -script F2_panel_sweep.vspscript > F2_panel_sweep.log 2>&1
"$VSP/vspscript" -script F2_panel_stab.vspscript > F2_panel_stab.log 2>&1
"$VSP/vspscript" -script F2_panel_beta.vspscript > F2_panel_beta.log 2>&1
"$VSP/vspscript" -script F3_vlm_sweep.vspscript > F3_vlm_sweep.log 2>&1
"$VSP/vspscript" -script F3_vlm_stab.vspscript > F3_vlm_stab.log 2>&1
"$VSP/vspscript" -script F3_vlm_beta.vspscript > F3_vlm_beta.log 2>&1
"$VSP/vspscript" -script F3_panel_sweep.vspscript > F3_panel_sweep.log 2>&1
"$VSP/vspscript" -script F3_panel_stab.vspscript > F3_panel_stab.log 2>&1
"$VSP/vspscript" -script F3_panel_beta.vspscript > F3_panel_beta.log 2>&1
