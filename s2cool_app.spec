from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH)
runtime_data = [
    (str(root / "s2cool_python_app" / "assets"), "s2cool_python_app/assets"),
    (str(root / "s2cool_python_app" / "config"), "s2cool_python_app/config"),
    (str(root / "M2_PVnowcasting_module" / "pv_hybrid_forecasting_multihorizon.py"), "M2_PVnowcasting_module"),
    (str(root / "M2_PVnowcasting_module" / "data"), "M2_PVnowcasting_module/data"),
    (str(root / "M3_CoolingLoad_prediction_module" / "cooling_hybrid_forecasting_multihorizon.py"), "M3_CoolingLoad_prediction_module"),
    (str(root / "M3_CoolingLoad_prediction_module" / "data"), "M3_CoolingLoad_prediction_module/data"),
]
hiddenimports = collect_submodules("pages") + collect_submodules("services") + collect_submodules("components") + collect_submodules("models")
a = Analysis([str(root / "s2cool_python_app" / "app.py")], pathex=[str(root / "s2cool_python_app")], binaries=[], datas=runtime_data, hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=["pytest", "notebook", "jupyter"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="S2Cool", debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name="S2Cool")
