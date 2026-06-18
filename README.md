# DiamFit
Fits a diameter and limb-darkening coefficient to interferometric visibilities and/or closure amplitudes.
Linear or power law limb darkening laws are both available. 
When fitting to visibilities, fits a scaling factor for each night+combiner. Uncertainties calculated by a block-bootstrap method.

Required Inputs
* data_dir - the relative path to the directory with the reduced+calibrated oifits that you want to use are located
* stars - a star or list of stars to fit
* diam_init - the initial guess for the diameter
* ldcK_init - the initial guess for the K-band limb-darkening coefficient
* ldcH_init - the initial guess for the H-band limb-darkening coefficient
* combiners - a beam combiner (or list of combiners) your data come from
* nights - the night or list of nights your data come from

*IMPORTANT NOTE - Your oifits files must be named in the following format to be recognized by diam_fit.py*

STAR.COMBINER.NIGHT*.oifits

Examples:
* altair.MIRCX.2026Jun16.oifits
* vega.MYSTIC.2025Oct03.01.oifits
* vega.MYSTIC.2025Oct03.02.oifits
* HD_27371.PIONIER.2027Jul01.extra_label.oifits

When you call diam_fit, the star, combiner, and nights must match the values in the filenames or else the files won't be read

Optional Inputs
* scenarios - accepted values are ['by_file','by_night','by_combiner','all'], 'all' is default. Multiple options allowed
* law - accepted values are ['power','linear'], 'power' is default
* boot_N - number of iterations for the bootstrapping. Default is 1000
* fix_ldcK - fix ldcK to ldcK_init (default = false)
* fix_ldcH - fix ldcH to ldcH_init (default = false)
* camp - fit closure amplitudes (default = false)
