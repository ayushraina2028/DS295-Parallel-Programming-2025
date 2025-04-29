# Parallel EM Algorithm Presentation

This project contains a LaTeX presentation for the Parallel EM Algorithm Implementation. The presentation is structured to provide a comprehensive overview of the algorithm, its implementation, and results.

## Project Structure

- **ppt.tex**: Contains the LaTeX code for the presentation, structured for a maximum of 14 slides.
- **images/**: Directory for storing images used in the presentation. (Contains a .gitkeep file to ensure the directory is tracked by Git.)
- **bibliography.bib**: Manages references and citations used in the presentation.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **README.md**: Documentation for the project.

## Compiling the Presentation

To compile the LaTeX presentation, follow these steps:

1. Ensure you have a LaTeX distribution installed (e.g., TeX Live, MiKTeX).
2. Open a terminal and navigate to the project directory.
3. Run the following command to compile the presentation:

   ```
   pdflatex ppt.tex
   ```

4. If you have citations, you may need to run BibTeX:

   ```
   bibtex ppt
   ```

5. Finally, run `pdflatex` again to ensure all references are updated:

   ```
   pdflatex ppt.tex
   pdflatex ppt.tex
   ```

## Usage

- Modify `ppt.tex` to update the content of the presentation.
- Add images to the `images/` directory and reference them in `ppt.tex`.
- Update `bibliography.bib` with any new references you wish to cite in the presentation.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.