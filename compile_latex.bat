@echo off
REM Compile LaTeX to PDF

echo Compiling LaTeX document to PDF...
echo.

cd paperresults

REM First pass
pdflatex -interaction=nonstopmode research_paper.tex

REM Second pass for references
pdflatex -interaction=nonstopmode research_paper.tex

REM Third pass for final cross-references
pdflatex -interaction=nonstopmode research_paper.tex

echo.
echo Compilation complete! Check research_paper.pdf
echo.

pause
