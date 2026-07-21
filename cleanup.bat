@echo off
REM ==================================================================
REM   cleanup.bat - clean up the herb_bot project
REM   Keeps only the working bot omela_bg.py and needed files.
REM   Put this file (with README.md and .gitignore) into the
REM   project folder and run it:
REM        C:\Users\zakco\VS Code\herb_bot
REM ==================================================================
setlocal
cd /d "%~dp0"

echo.
echo === herb_bot cleanup ===
echo Folder: %CD%
echo.

if not exist "omela_bg.py" (
  echo [ERROR] omela_bg.py not found in this folder.
  echo Copy cleanup.bat into the herb_bot project folder and run again.
  echo.
  pause
  exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 goto plaindelete

REM -------- Git repo: remove files and stage for commit --------
echo --- Removing old bot herb_bot ...
git rm -f  --ignore-unmatch herb_bot.py herb_bot.log
git rm -rf --ignore-unmatch templates

echo --- Removing debug screenshots, page dumps and logs ...
git rm -f --ignore-unmatch omela_bg.log omela.png page_region.png
git rm -f --ignore-unmatch "page_full*.png" "page_dom_*.html" "screen_full.png" "screen_region.png" "debug_omela*.png" "debug_preview.png"

echo --- Removing old / broken instruction files (all .md) ...
git rm -f --ignore-unmatch "*.md"

echo --- Untracking browser_profile from git (files stay on disk) ...
git rm -r --cached --ignore-unmatch browser_profile >nul 2>&1

echo --- Adding clean files ...
if exist "README.md"   git add "README.md"
if exist ".gitignore"  git add ".gitignore"
git add -A

echo.
echo === Changes to be committed (git status) ===
git status --short
echo.
echo -------------------------------------------------------------
echo Review the list above. To save and push to GitHub:
echo     git commit -m "cleanup: remove old bot, debug artifacts, browser_profile"
echo     git push
echo -------------------------------------------------------------
goto end

:plaindelete
echo (git not found - deleting files directly)
del /q herb_bot.py herb_bot.log omela_bg.log omela.png page_region.png 2>nul
del /q page_full*.png page_dom_*.html screen_full.png screen_region.png debug_omela*.png debug_preview.png 2>nul
rmdir /s /q templates 2>nul
for %%f in (*.md) do if /i not "%%~nxf"=="README.md" del /q "%%f"
echo Done (without git). browser_profile was not touched.

:end
echo.
echo Done.
pause
