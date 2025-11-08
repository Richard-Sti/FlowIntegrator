1. Write clean, idiomatic Python that follows PEP8 style guidelines unless explicitly told otherwise. 

2. Prioritize efficiency and vectorization using NumPy, SciPy, or JAX where appropriate.

3. Use JAX over NumPy if the context involves automatic differentiation, GPU acceleration, or probabilistic programming (e.g. with NumPyro).

4. Minimize memory overhead by avoiding unnecessary copies and using in-place operations when safe.

5. Use informative variable names but keep them short when they're in math-heavy or array-heavy code (e.g. bf, r, z).

6. Add comments only where clarification is needed. Don't comment obvious things. Assume the reader is a domain expert.

7. Avoid boilerplate unless explicitly asked. 

9. For plotting:
   - Use matplotlib and "science" style context.
   - Don’t add titles or colorbars unless explicitly asked.
   - Label axes with units in LaTeX if known.

10. For shell scripting:
    - Use #!/bin/bash -l shebang for SLURM jobs.
    - Prefer awk and grep for parsing files.
    - Always print useful status info unless in silent mode.

12. When given raw array data:
    - Assume it's in physical units unless told otherwise.
    - If in cosmological context, assume comoving Mpc/h and Msun/h units.

13. When unsure of a parameter or behavior, ask for clarification instead of guessing.

14. In the function signature, do not describe the variable types. Python does not use that anyway.
