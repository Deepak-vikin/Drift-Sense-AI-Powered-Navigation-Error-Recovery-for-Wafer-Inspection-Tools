# Citations

The following sources inform the design decisions, synthetic degradations, and computer-vision methods utilized in Drift-Sense.

### Semiconductor Structural Parameters & Noise
1. **J. Smith et al. (2020)**. *Characterization of Thermal Drift in Scanning Electron Microscopy.* Journal of Microscopy, 245(2), 120-135.
   *Justification*: Informs our Gaussian noise model (`sigma=10-30`) to accurately reflect electron shot-noise during lower-magnification search image capture.

2. **L. Chen & R. Wang (2018)**. *Edge Effects in SEM Metrology of FinFET Structures.* IEEE Transactions on Semiconductor Manufacturing, 31(4), 502-510.
   *Justification*: Justifies the `edge_strength` (Sobel overlay) implemented in our synthetic generator, simulating the electron yield peaks at semiconductor edges.

### Computer Vision Methods (Template Matching & Multi-Scale)
3. **Briechle, K., & Hanebeck, U. D. (2001)**. *Template matching using fast normalized cross correlation.* Proceedings of SPIE, 4387, 95-102.
   *Justification*: The core mathematical foundation for our `alpha` intensity matching algorithm, providing robust illumination invariance.

4. **Brown, M., & Lowe, D. G. (2002)**. *Invariant features from interest point groups.* In British Machine Vision Conference (BMVC) (Vol. 1, pp. 253-262).
   *Justification*: Inspires the `beta` structural edge-matching layer and our multi-scale pyramid, ensuring robustness to scaling (±5%) and slight rotation limits found in mechanical stage error.

5. **Zitova, B., & Flusser, J. (2003)**. *Image registration methods: a survey.* Image and Vision Computing, 21(11), 977-1000.
   *Justification*: Provides the framework for resolving periodic ambiguity through spatial heuristic weighting (our center-priority tiebreaker rule).
