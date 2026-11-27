@echo off
rem Runs every case with OpenVSP's vspscript. Set VSP to the OpenVSP folder if needed.
if "%VSP%"=="" set VSP=%USERPROFILE%\OpenVSP
cd /d %~dp0
"%VSP%\vspscript.exe" -script F1_vlm_sweep.vspscript > F1_vlm_sweep.log 2>&1
"%VSP%\vspscript.exe" -script F1_vlm_stab.vspscript > F1_vlm_stab.log 2>&1
"%VSP%\vspscript.exe" -script F1_panel_sweep.vspscript > F1_panel_sweep.log 2>&1
"%VSP%\vspscript.exe" -script F1_panel_stab.vspscript > F1_panel_stab.log 2>&1
"%VSP%\vspscript.exe" -script F2_vlm_sweep.vspscript > F2_vlm_sweep.log 2>&1
"%VSP%\vspscript.exe" -script F2_vlm_stab.vspscript > F2_vlm_stab.log 2>&1
"%VSP%\vspscript.exe" -script F2_panel_sweep.vspscript > F2_panel_sweep.log 2>&1
"%VSP%\vspscript.exe" -script F2_panel_stab.vspscript > F2_panel_stab.log 2>&1
"%VSP%\vspscript.exe" -script F3_vlm_sweep.vspscript > F3_vlm_sweep.log 2>&1
"%VSP%\vspscript.exe" -script F3_vlm_stab.vspscript > F3_vlm_stab.log 2>&1
"%VSP%\vspscript.exe" -script F3_panel_sweep.vspscript > F3_panel_sweep.log 2>&1
"%VSP%\vspscript.exe" -script F3_panel_stab.vspscript > F3_panel_stab.log 2>&1
